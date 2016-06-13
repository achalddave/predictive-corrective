--[[
Helper class to load data and labels from an LMDB containing LabeledVideoFrames
as values.
]]--

local classic = require 'classic'
local lmdb = require 'lmdb'
local threads = require 'threads'

local video_frame_proto = require 'video_util.video_frames_pb'

local Sampler = classic.class('Sampler')
Sampler:mustHave('sample_keys')
Sampler:mustHave('num_samples')
function Sampler.static.permute(list)
    local permuted_list = {}
    local permutation = torch.randperm(#list)
    for i = 1, permutation:nElement() do
        table.insert(permuted_list, list[permutation[i]])
    end
    return permuted_list
end

local PermutedSampler = classic.class('PermutedSampler', Sampler)
function PermutedSampler:_init(lmdb_without_images_path, num_labels)
    --[[
    Returns consecutive batches from a permuted list of keys.

    Once the list has been exhausted, we generate a new permutation.
    ]]--
    self.imageless_path = lmdb_without_images_path
    self.keys = PermutedSampler.load_lmdb_keys(lmdb_without_images_path)
    self.permuted_keys = Sampler.permute(self.keys)
    self.key_index = 1
end

function PermutedSampler:sample_keys(num_keys)
    --[[
    Sample the next set of keys.
    ]]--
    local batch_keys = {}
    for _ = 1, num_keys do
        if self.key_index > #self.permuted_keys then
            print(string.format('%s: Finished pass through data, repermuting!',
                                os.date('%X')))
            self.permuted_keys = Sampler.permute(self.keys)
            self.key_index = 1
        end
        table.insert(batch_keys, self.permuted_keys[self.key_index])
        self.key_index = self.key_index + 1
    end
    return batch_keys
end

function PermutedSampler:num_samples()
    return #self.keys
end

function PermutedSampler.static.load_lmdb_keys(lmdb_path)
    --[[
    Loads keys from LMDB, using the LMDB that doesn't contain images.

    Returns:
        keys: List of keys.
    ]]--

    -- Get LMDB cursor.
    local db = lmdb.env { Path = lmdb_path }
    db:open()
    local transaction = db:txn(true --[[readonly]])
    local cursor = transaction:cursor()

    local keys = {}
    for i = 1, db:stat().entries do
        local key, _ = cursor:get('MDB_GET_CURRENT')
        table.insert(keys, key)
        if i ~= db:stat().entries then cursor:next() end
    end

    -- Cleanup.
    cursor:close()
    transaction:abort()
    db:close()

    return keys
end

local BalancedSampler = classic.class('BalancedSampler', Sampler)
function BalancedSampler:_init(lmdb_without_images_path, num_labels)
    --[[
    Samples from each class a balanced number of times, so that the model should
    see approximately the same amount of data from each class.
    ]]--
    self.imageless_path = lmdb_without_images_path
    self.num_labels = num_labels
    self.label_keys, self.num_keys = self:_load_label_key_mapping()
    -- For each label, maintain an index of the next data point to output.
    self.label_indices = {}
    self:_permute_keys()
end

function BalancedSampler:sample_keys(num_keys)
    local keys = {}
    for i = 1, num_keys do
        local label = math.random(self.num_labels)
        table.insert(keys, self.label_keys[label][self.label_indices[label]])
        self:_advance_label_index(label)
    end
    return keys
end

function BalancedSampler:num_samples()
    return self.num_keys
end

function BalancedSampler:_advance_label_index(label)
    if self.label_indices[label] + 1 <= #self.label_keys[label] then
        self.label_indices[label] = self.label_indices[label] + 1
    else
        self.label_indices[label] = 1
    end
end

function BalancedSampler:_permute_keys()
    for i = 1, self.num_labels do
        self.label_keys[i] = Sampler.permute(self.label_keys[i])
        self.label_indices[i] = 1
    end
end

function BalancedSampler:_load_label_key_mapping()
    -- Get LMDB cursor.
    local db = lmdb.env { Path = self.imageless_path }
    db:open()
    local transaction = db:txn(true --[[readonly]])
    local cursor = transaction:cursor()

    local label_keys = {}
    for i = 1, self.num_labels do
        label_keys[i] = {}
    end

    local num_keys = db:stat().entries
    for i = 1, num_keys do
        local key, value = cursor:get('MDB_GET_CURRENT')
        local video_frame = video_frame_proto.LabeledVideoFrame()
        video_frame:ParseFromString(value:storage():string())
        for _, label in ipairs(video_frame.label) do
            -- Label ids start at 0.
            table.insert(label_keys[label.id + 1], key)
        end
        if i ~= db:stat().entries then cursor:next() end
    end

    -- Cleanup.
    cursor:close()
    transaction:abort()
    db:close()

    return label_keys, num_keys
end

local DataLoader = classic.class('DataLoader')

function DataLoader:_init(
    lmdb_path, lmdb_without_images_path, sampler_class, num_labels)
    --[[
    Args:
        lmdb_path (str): Path to LMDB containing LabeledVideoFrames as values.
        lmdb_without_images_path (str): Path to LMDB containing
            LabeledVideoFrames as values, but without any raw image data. This
            is easy to iterate over, and can be used to decide which images to
            sample.
        sampler_class (Sampler): Type of sampler used for batches.
        num_labels (num): Number of total labels.
    ]]--
    self.path = lmdb_path
    self.sampler = sampler_class(lmdb_without_images_path, num_labels)
    self.num_labels = num_labels
    self._prefetched_data = {
        batch_images = nil,
        batch_labels = nil
    }
    self._prefetching_thread = threads.Threads(1)
end

function DataLoader:num_samples()
    return self.sampler:num_samples()
end

function DataLoader:load_batch(batch_size, return_keys)
    --[[
    Load a batch of images and labels.

    Args:
        batch_size (num)
        return_keys (bool): Whether to return the keys from the batch.
            Default false.

    Returns:
        batch_images (Array of ByteTensors)
        batch_labels (Array of arrays): Containing label ids.
        batch_keys (Array of strings): Only returned if return_keys is True.
    ]]--
    return_keys = return_keys == nil and false or return_keys
    if self._prefetched_data.batch_images == nil then
        -- Thread has not fetched data
        self._prefetching_thread:synchronize()
        if self._prefetched_data.batch_images == nil then
            self:fetch_batch_async(batch_size)
            self._prefetching_thread:synchronize()
        end
    end

    local batch_images = self._prefetched_data.batch_images
    local batch_labels = self._prefetched_data.batch_labels
    local batch_keys = self._prefetched_data.batch_keys
    self._prefetched_data.batch_images = nil
    self._prefetched_data.batch_labels = nil
    self._prefetched_data.batch_keys = nil
    if return_keys then
        return batch_images, batch_labels, batch_keys
    else
        return batch_images, batch_labels
    end
end

function DataLoader:fetch_batch_async(batch_size)
    --[[ Load a batch, store it for returning in next call to load_batch. ]]--
    local batch_keys = self.sampler:sample_keys(batch_size)

    self._prefetching_thread:addjob(
        DataLoader._load_image_labels_from_path,
        function(output)
            self._prefetched_data = output
        end,
        self.path, batch_keys, self.num_labels)
end

function DataLoader.static._load_image_labels_from_proto(video_frame_proto)
    --[[
    Loads an image tensor and labels for a given key.

    Returns:
        img: (num_channels, height, width) tensor
        labels: Table containing numeric id for each label.
    ]]

    local img = DataLoader._image_proto_to_tensor(video_frame_proto.frame.image)

    -- Load labels in an array.
    local labels = {}
    for _, label in ipairs(video_frame_proto.label) do
        table.insert(labels, label.id)
    end

    return img, labels
end

-- ###
-- Private methods.
-- ###

function DataLoader.static._labels_to_tensor(labels, num_labels)
    --[[
    Convert an array of label ids into a 1-hot encoding in a binary Tensor.
    ]]--
    local labels_tensor = torch.ByteTensor(num_labels):zero()
    for _, label in ipairs(labels) do
        -- Label ids start at 0.
        labels_tensor[label + 1] = 1
    end
    return labels_tensor
end
function DataLoader.static._image_proto_to_tensor(image_proto)
    local image_storage = torch.ByteStorage()
    image_storage:string(image_proto.data)
    return torch.ByteTensor(image_storage):reshape(
        image_proto.channels, image_proto.height, image_proto.width)
end

function DataLoader.static._load_image_labels_from_path(
    lmdb_path, keys, num_labels)
    --[[
    Load images and labels for a set of keys from the LMDB.

    Args:
        lmdb_path (str): Path to an LMDB of LabeledVideoFrames
        keys (list): List of string keys.
        num_labels (num): Number of total labels.

    Returns:
        images_and_labels (table): Contains
            batch_images: Array of ByteTensors
            batch_labels: Array of ByteTensors
            batch_keys: Same as keys argument.
    ]]--
    -- Open database
    local torch = require 'torch'
    local lmdb = require 'lmdb'
    local video_frame_proto = require 'video_util.video_frames_pb'
    local DataLoader = require('data_loader').DataLoader

    local db = lmdb.env { Path = lmdb_path }
    db:open()
    local transaction = db:txn(true --[[readonly]])

    local batch_labels = {}
    local batch_images = {}
    for i = 1, #keys do
        -- Load LabeledVideoFrame.
        local key = keys[i]
        local video_frame = video_frame_proto.LabeledVideoFrame()
        video_frame:ParseFromString(
            transaction:get(key):storage():string())

        -- Load image and labels.
        local img, labels = DataLoader._load_image_labels_from_proto(
            video_frame)
        labels = DataLoader._labels_to_tensor(labels, num_labels)
        table.insert(batch_images, img)
        table.insert(batch_labels, labels)
    end

    transaction:abort()
    db:close()

    return {
        batch_images = batch_images,
        batch_labels = batch_labels,
        batch_keys = keys
    }
end

return {
    DataLoader = DataLoader,
    PermutedSampler = PermutedSampler,
    BalancedSampler = BalancedSampler
}
