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

function Sampler.static.next_frame_key(frame_key)
    --[[ Return the key for the next frame. ]]--
    -- Keys are of the form '<filename>-<frame_number>'.
    -- Find the index of the '-'
    local _, split_index = string.find(frame_key, '.*-')
    local filename = string.sub(frame_key, 1, split_index - 1)
    local frame_number = tonumber(string.sub(frame_key, split_index + 1, -1))
    frame_number = frame_number + 1
    return string.format('%s-%d', filename, frame_number)
end

local PermutedSampler = classic.class('PermutedSampler', Sampler)
function PermutedSampler:_init(
        lmdb_without_images_path, _ --[[num_labels]],
        sequence_length, _ --[[options]])
    --[[
    Returns consecutive batches from a permuted list of keys.

    Once the list has been exhausted, we generate a new permutation.

    Args:
        lmdb_without_images_path (str): Path to LMDB containing
            LabeledVideoFrames as values, but without any raw image data. This
            is easy to iterate over, and can be used to decide which images to
            sample.
        num_labels (num)
        sequence_length (num): If provided, sample sequences of length
            sequence_length for each training sample.
    ]]--
    self.imageless_path = lmdb_without_images_path
    self.sequence_length = sequence_length == nil and 1 or sequence_length
    self.keys = PermutedSampler.filter_end_frames(
        PermutedSampler.load_lmdb_keys(lmdb_without_images_path),
        self.sequence_length)
    self.permuted_keys = Sampler.permute(self.keys)
    self.key_index = 1
end

function PermutedSampler:sample_keys(num_sequences)
    --[[
    Sample the next set of keys.

    Returns:
        batch_keys (Array of array of strings): Each element contains
            num_sequences arrays, each of which contains sequence_length keys.
    ]]--
    local batch_keys = {}
    for _ = 1, self.sequence_length do
        table.insert(batch_keys, {})
    end
    for _ = 1, num_sequences do
        if self.key_index > self:num_samples() then
            print(string.format('%s: Finished pass through data, repermuting!',
                                os.date('%X')))
            self.permuted_keys = Sampler.permute(self.keys)
            self.key_index = 1
        end
        local sampled_key = self.permuted_keys[self.key_index]
        for step = 1, self.sequence_length do
            table.insert(batch_keys[step], sampled_key)
            sampled_key = Sampler.next_frame_key(sampled_key)
        end
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
        keys: Array of keys.
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

function PermutedSampler.static.filter_end_frames(frame_keys, sequence_length)
    --[[ Filter out the last sequence_length frames in the video.
    --
    -- Args:
    --     frame_keys (array)
    --]]
    local frame_keys_set = {}
    for _, key in ipairs(frame_keys) do
        frame_keys_set[key] = true
    end
    local filtered_frame_keys = {}
    for key, _ in pairs(frame_keys_set) do
        local frame_to_check = key
        local frame_valid = true
        -- Check if next sequence_length frames exist.
        for _ = 2, sequence_length do
            frame_to_check = Sampler.next_frame_key(frame_to_check)
            if frame_keys_set[frame_to_check] == nil then
                frame_valid = false
                break
            end
        end
        if frame_valid then
            table.insert(filtered_frame_keys, key)
        end
    end
    return filtered_frame_keys
end

local BalancedSampler = classic.class('BalancedSampler', Sampler)
function BalancedSampler:_init(
        lmdb_without_images_path, num_labels, sequence_length, options)
    --[[
    Samples from each class a balanced number of times, so that the model should
    see approximately the same amount of data from each class.

    Args:
        lmdb_without_images_path (str): Path to LMDB containing
            LabeledVideoFrames as values, but without any raw image data. This
            is easy to iterate over, and can be used to decide which images to
            sample.
        num_labels (num)
        sequence_length (num): If provided, sample sequences of length
            sequence_length for each training sample.
        options (table):
            include_bg: Whether to output a balanced number of frames that
                have none of the labels.
    ]]--
    -- BalancedSampler is not yet implemented for sequences.
    error("BalancedSampler is currently broken.")
    self.imageless_path = lmdb_without_images_path
    options = options == nil and {} or options
    self.include_bg = options.include_bg == nil and false or include_bg
    self.num_labels = self.include_bg and num_labels + 1 or num_labels

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
        self.label_keys[i] = Sampler.permute(self.label_keys[i])
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

    -- Create mapping from label index to keys.
    local label_keys = {}
    for i = 1, self.num_labels do
        label_keys[i] = {}
    end

    local num_keys = db:stat().entries
    for i = 1, num_keys do
        local key, value = cursor:get('MDB_GET_CURRENT')
        local video_frame = video_frame_proto.LabeledVideoFrame()
        video_frame:ParseFromString(value:storage():string())
        local num_frame_labels = #video_frame.label
        if num_frame_labels == 0 and self.include_bg then
            -- Add frame to list of background frames.
            table.insert(label_keys[#label_keys], key)
        else
            for _, label in ipairs(video_frame.label) do
                -- Label ids start at 0.
                table.insert(label_keys[label.id + 1], key)
            end
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

function DataLoader:_init(lmdb_path, sampler, num_labels)
    --[[
    Args:
        lmdb_path (str): Path to LMDB containing LabeledVideoFrames as values.
        sampler (Sampler): Sampler used for batches
        num_labels (num): Number of total labels.
    ]]--
    self.path = lmdb_path
    self.sampler = sampler
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
        images (Array of array of ByteTensors): Contains image sequences for
            the batch. Each element is a step in the sequence, so that images is
            an array of length sequence_length, whose elements are arrays of
            length batch_size.
        labels (ByteTensor): Contains label ids. Size is
            (sequence_length, batch_size, num_labels)
        keys (Array of array of strings): Only returned if return_keys is
            True.
    ]]--
    return_keys = return_keys == nil and false or return_keys
    if not self:_data_fetched() then
        self:fetch_batch_async(batch_size)
        self._prefetching_thread:synchronize()
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
    if self:_data_fetched() then
        return
    end

    local batch_keys = self.sampler:sample_keys(batch_size)

    self._prefetching_thread:addjob(
        DataLoader._load_images_labels_for_keys,
        function(output)
            self._prefetched_data = output
        end,
        self.path, batch_keys, self.num_labels)
end

function DataLoader:_data_fetched()
    -- Wait for possible fetching thread to finish.
    self._prefetching_thread:synchronize()
    return self._prefetched_data.batch_images ~= nil
end

function DataLoader.static._load_image_labels_from_proto(video_frame_proto)
    --[[
    Loads an image tensor and labels for a given key.

    Returns:
        img (ByteTensor): Image of size (num_channels, height, width).
        labels (Array): Contains numeric id for each label.
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

function DataLoader.static._load_images_labels_for_keys(
    lmdb_path, keys, num_labels)
    --[[
    Load images and labels for a set of keys from the LMDB.

    Args:
        lmdb_path (str): Path to an LMDB of LabeledVideoFrames
        keys (array): Array of array of string keys. Each element must be
            an array of the same length as every element, and contains keys for
            one step of the image sequence.
        num_labels (num): Number of total labels.

    Returns:
        images_and_labels (table): See load_batch for detailed information.
            Contains
            - batch_images: Array of array of ByteTensors
            - batch_labels: ByteTensor
            - batch_keys: Same as keys argument.
    ]]--
    -- Open database
    local torch = require 'torch'
    local lmdb = require 'lmdb'
    local video_frame_proto = require 'video_util.video_frames_pb'
    local DataLoader = require('data_loader').DataLoader

    local db = lmdb.env { Path = lmdb_path }
    db:open()
    local transaction = db:txn(true --[[readonly]])

    local num_steps = #keys
    local batch_size = #keys[1]
    local batch_labels = torch.ByteTensor(num_steps, batch_size, num_labels)
    local batch_images = {}
    for step = 1, num_steps do
        batch_images[step] = {}
        for i = 1, batch_size do
            -- Load LabeledVideoFrame.
            local video_frame = video_frame_proto.LabeledVideoFrame()
            video_frame:ParseFromString(
                transaction:get(keys[step][i]):storage():string())

            -- Load image and labels.
            local img, labels = DataLoader._load_image_labels_from_proto(
                video_frame)
            labels = DataLoader._labels_to_tensor(labels, num_labels)
            table.insert(batch_images[step], img)
            batch_labels[{step, i}] = labels
        end
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
    -- TODO(achald): Implement sequence sampling for BalancedSampler.
    -- BalancedSampler = BalancedSampler
}
