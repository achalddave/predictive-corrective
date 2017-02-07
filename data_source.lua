local classic = require 'classic'
local lmdb = require 'lmdb'
local torch = require 'torch'
require 'classic.torch'

local video_frame_proto = require 'video_util.video_frames_pb'

local DataSource = classic.class('DataSource')
DataSource:mustHave('num_samples')
-- Given an array of keys, load the associated images and labels.
DataSource:mustHave('load_data')
-- Return a table mapping data sample keys to labels for the sample.
DataSource:mustHave('key_label_map')
DataSource:mustHave('num_labels')

local VideoDataSource = classic.class('VideoDataSource', 'DataSource')
-- Return a table mapping video name to an ordered array of keys for frames in
-- the video.
VideoDataSource:mustHave('video_keys')
VideoDataSource:mustHave('frame_video_offset')
VideoDataSource.END_OF_SEQUENCE = -1

local LabeledVideoFramesLmdbSource = classic.class(
    'LabeledVideoFramesLmdbSource', 'VideoDataSource')
function LabeledVideoFramesLmdbSource:_init(
        lmdb_path, lmdb_without_images_path, num_labels)
    self.lmdb_path = lmdb_path
    self.lmdb_without_images_path = lmdb_without_images_path
    self.num_labels_ = num_labels

    self.key_labels = self:load_key_label_map()

    self.num_keys = 0
    self.video_keys_ = {}
    for key, _ in pairs(self.key_labels) do
        local video, frame = LabeledVideoFramesLmdbSource.parse_frame_key(key)
        if self.video_keys_[video] == nil then
            self.video_keys_[video] = {}
        end
        self.video_keys_[video][frame] = key
        self.num_keys = self.num_keys + 1
    end
end

function LabeledVideoFramesLmdbSource:num_samples() return self.num_keys end

function LabeledVideoFramesLmdbSource:key_label_map() return self.key_labels end

function LabeledVideoFramesLmdbSource:video_keys() return self.video_keys_ end

function LabeledVideoFramesLmdbSource:num_labels() return self.num_labels_ end

function LabeledVideoFramesLmdbSource:frame_video_offset(key)
    return LabeledVideoFramesLmdbSource.static.parse_frame_key(key)
end

function LabeledVideoFramesLmdbSource:load_data(keys)
    --[[
    Load images and labels for a set of keys from the LMDB.

    Args:
        keys (array): Array of array of string keys. Each element must be
            an array of the same length as every element, and contains keys for
            one step of the image sequence.

    Returns:
        batch_images: Array of array of ByteTensors
        batch_labels: ByteTensor
    ]]--
    local db = lmdb.env { Path = self.lmdb_path }
    db:open()
    local transaction = db:txn(true --[[readonly]])

    local num_steps = #keys
    local batch_size = #keys[1]
    local batch_labels = torch.ByteTensor(
        num_steps, batch_size, self.num_labels_)
    local batch_images = {}
    for step = 1, num_steps do
        batch_images[step] = {}
    end
    for i = 1, batch_size do
        for step = 1, num_steps do
            if keys[step][i] == VideoDataSource.END_OF_SEQUENCE then
                table.insert(batch_images[step],
                             VideoDataSource.END_OF_SEQUENCE)
                batch_labels[{step, i}]:zero()
            else
                -- Load LabeledVideoFrame.
                local video_frame = video_frame_proto.LabeledVideoFrame()
                video_frame:ParseFromString(
                    transaction:get(keys[step][i]):storage():string())

                -- Load image and labels.
                local img, labels =
                    LabeledVideoFramesLmdbSource._load_image_labels_from_proto(
                        video_frame)
                labels = LabeledVideoFramesLmdbSource._labels_to_tensor(
                    labels, self.num_labels_)
                table.insert(batch_images[step], img)
                batch_labels[{step, i}] = labels
            end
        end
    end

    transaction:abort()
    db:close()

    return batch_images, batch_labels
end

function LabeledVideoFramesLmdbSource.static._labels_to_tensor(
    labels, num_labels)
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

function LabeledVideoFramesLmdbSource.static._image_proto_to_tensor(image_proto)
    local image_storage = torch.ByteStorage()
    if image_proto.data:len() == 0 then
        return torch.ByteTensor()
    end
    image_storage:string(image_proto.data)
    return torch.ByteTensor(image_storage):reshape(
        image_proto.channels, image_proto.height, image_proto.width)
end

function LabeledVideoFramesLmdbSource.static._load_image_labels_from_proto(
        frame_proto)
    --[[
    Loads an image tensor and labels for a given key.

    Returns:
        img (ByteTensor): Image of size (num_channels, height, width).
        labels (Array): Contains numeric id for each label.
    ]]

    local img = LabeledVideoFramesLmdbSource._image_proto_to_tensor(
        frame_proto.frame.image)

    -- Load labels in an array.
    local labels = {}
    for _, label in ipairs(frame_proto.label) do
        table.insert(labels, label.id)
    end

    return img, labels
end

function LabeledVideoFramesLmdbSource:load_key_label_map()
    --[[
    Load mapping from frame keys to labels array.

    Returns:
        key_labels: Table mapping frame keys to array of label indices.
    ]]--
    -- Get LMDB cursor.
    local db = lmdb.env { Path = self.lmdb_without_images_path }
    db:open()
    local transaction = db:txn(true --[[readonly]])
    local cursor = transaction:cursor()

    -- Create mapping from keys to labels.
    local key_label_map = {}

    local num_keys = db:stat().entries
    for i = 1, num_keys do
        local key, value = cursor:get('MDB_GET_CURRENT')
        local video_frame = video_frame_proto.LabeledVideoFrame()
        video_frame:ParseFromString(value:storage():string())
        local num_frame_labels = #video_frame.label
        if num_frame_labels == 0 then
            -- Add frame to list of background frames.
            key_label_map[key] = {self.num_labels_ + 1}
        else
            local labels = {}
            for _, label in ipairs(video_frame.label) do
                -- Label ids start at 0.
                table.insert(labels, label.id + 1)
            end
            key_label_map[key] = labels
        end
        if i ~= db:stat().entries then cursor:next() end
    end

    -- Cleanup.
    cursor:close()
    transaction:abort()
    db:close()

    return key_label_map
end

function LabeledVideoFramesLmdbSource.static.parse_frame_key(frame_key)
    -- Keys are of the form '<filename>-<frame_number>'.
    -- Find the index of the '-'
    local _, split_index = string.find(frame_key, '[^-]*-')
    local filename = string.sub(frame_key, 1, split_index - 1)
    local frame_number = tonumber(string.sub(frame_key, split_index + 1, -1))
    return filename, frame_number
end

return {
    DataSource = DataSource,
    VideoDataSource = VideoDataSource,
    LabeledVideoFramesLmdbSource = LabeledVideoFramesLmdbSource
}
