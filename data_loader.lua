--[[
Helper class to load data and labels from an LMDB containing LabeledVideoFrames
as values.
]]--

local class = require 'class'
local lmdb = require 'lmdb'
local video_frame_proto = require 'video_util.video_frames_pb'

local DataLoader = class('DataLoader')

function DataLoader:__init(lmdb_path, lmdb_without_images_path, num_labels)
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
end

function DataLoader:load_batch(batch_size)
    --[[
    Load a batch of images and labels.

    Returns:
        batch_images: Array of ByteTensors
        batch_labels: Array of arrays containing label ids.
    ]]--
    -- Open database
    local db = lmdb.env { Path = self.path }
    db:open()
    local transaction = db:txn(true --[[readonly]])

    local batch_labels = {}
    local batch_images = {}
    local batch_keys = self:_sample_keys_batch(batch_size)
    for i = 1, batch_size do
        -- Load LabeledVideoFrame.
        local key = batch_keys[i]
        local video_frame = video_frame_proto.LabeledVideoFrame()
        video_frame:ParseFromString(transaction:get(key):storage():string())

        -- Load image and labels.
        local image, labels = self:load_image_labels(video_frame)
        table.insert(batch_images, image)
        table.insert(batch_labels, labels)
    end

    transaction:abort()
    db:close()
    return batch_images, batch_labels
end

function DataLoader:load_image_labels(video_frame_proto)
    --[[
    Loads an image tensor and labels for a given key.

    Returns:
        image: (num_channels, height, width) tensor
        labels: Table containing numeric id for each label.
    ]]

    local image = self:_image_proto_to_tensor(video_frame_proto.frame.image)

    -- Load labels in an array.
    local labels = {}
    for _, label in ipairs(video_frame_proto.label) do
        table.insert(labels, label.id)
    end

    return image, labels
end

function DataLoader:_image_proto_to_tensor(image_proto)
    local image_storage = torch.ByteStorage()
    image_storage:string(image_proto.data)
    return torch.ByteTensor(image_storage):reshape(
        image_proto.channels, image_proto.height, image_proto.width)
end

function DataLoader:_sample_keys_batch(batch_size)
    -- TODO(achald): Allow other ways of sampling keys.
    local batch_keys = {}
    for _ = 1, batch_size do
        table.insert(batch_keys, self.keys[math.random(#self.keys)])
    end
    return batch_keys
end

function DataLoader:_load_keys()
    --[[
    Loads keys from LMDB, using the LMDB that doesn't contain images.

    Returns:
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

return {DataLoader = DataLoader}
