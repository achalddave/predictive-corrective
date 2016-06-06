--[[
Helper class to load data and labels from an LMDB containing LabeledVideoFrames
as values.
]]--

local classic = require 'classic'
local lmdb = require 'lmdb'
local threads = require 'threads'

local video_frame_proto = require 'video_util.video_frames_pb'

local DataLoader = classic.class('DataLoader')

function DataLoader:_init(lmdb_path, lmdb_without_images_path, num_labels)
    --[[
    Args:
        lmdb_path (str): Path to LMDB containing LabeledVideoFrames as values.
        lmdb_without_images_path (str): Path to LMDB containing
            LabeledVideoFrames as values, but without any raw image data. This
            is easy to iterate over, and can be used to decide which images to
            sample.
    ]]--
    self.path = lmdb_path
    self.imageless_path = lmdb_without_images_path
    self.keys = self:_load_keys()
    self._prefetched_data = {
        batch_images = nil,
        batch_labels = nil
    }
    self._permuted_keys = self:_permute_keys()
    self._key_index = 1
    self._prefetching_thread = threads.Threads(1)
end

function DataLoader:num_samples()
    return #self.keys
end

function DataLoader:load_batch(batch_size)
    --[[
    Load a batch of images and labels.

    Returns:
        batch_images: Array of ByteTensors
        batch_labels: Array of arrays containing label ids.
    ]]--
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
    self._prefetched_data.batch_images = nil
    self._prefetched_data.batch_labels = nil
    return batch_images, batch_labels
end

function DataLoader:fetch_batch_async(batch_size)
    --[[ Load a batch, store it for returning in next call to load_batch. ]]--
    local batch_keys = self:_get_batch_keys(batch_size)

    self._prefetching_thread:addjob(
        function(path, batch_keys)
            -- Open database
            local torch = require 'torch'
            local lmdb = require 'lmdb'
            local video_frame_proto = require 'video_util.video_frames_pb'
            local DataLoader = require('data_loader').DataLoader

            local db = lmdb.env { Path = path }
            db:open()
            local transaction = db:txn(true --[[readonly]])

            local batch_labels = {}
            local batch_images = {}
            for i = 1, batch_size do
                -- Load LabeledVideoFrame.
                local key = batch_keys[i]
                local video_frame = video_frame_proto.LabeledVideoFrame()
                video_frame:ParseFromString(
                    transaction:get(key):storage():string())

                -- Load image and labels.
                local img, labels = DataLoader.load_image_labels(video_frame)
                table.insert(batch_images, img)
                table.insert(batch_labels, labels)
            end

            transaction:abort()
            db:close()

            output = {}
            output.batch_images = batch_images
            output.batch_labels = batch_labels
            return output
        end,
        function(output)
            self._prefetched_data = output
        end, self.path, batch_keys)
end

function DataLoader.static.labels_to_tensor(labels, num_labels)
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

function DataLoader.static.load_image_labels(video_frame_proto)
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

function DataLoader:_get_batch_keys(batch_size)
    local batch_keys = {}
    for _ = 1, batch_size do
        if self._key_index > #self._permuted_keys then
            print(string.format('%s: Finished pass through data, repermuting!',
                                os.date('%X')))
            self._permuted_keys = self:_permute_keys()
            self._key_index = 1
        end
        table.insert(batch_keys, self._permuted_keys[self._key_index])
        self._key_index = self._key_index + 1
    end
    return batch_keys
end

function DataLoader:_permute_keys()
    local permuted_keys = {}
    local permutation = torch.randperm(#self.keys)
    for i = 1, permutation:nElement() do
        table.insert(permuted_keys, self.keys[permutation[i]])
    end
    return permuted_keys
end

function DataLoader:_load_keys()
    --[[
    Loads keys from LMDB, using the LMDB that doesn't contain images.

    Returns:
        keys: List of keys.
    ]]--

    -- Get LMDB cursor.
    local db = lmdb.env { Path = self.imageless_path }
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

function DataLoader.static._image_proto_to_tensor(image_proto)
    local image_storage = torch.ByteStorage()
    image_storage:string(image_proto.data)
    return torch.ByteTensor(image_storage):reshape(
        image_proto.channels, image_proto.height, image_proto.width)
end

return {DataLoader = DataLoader}
