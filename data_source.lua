local classic = require 'classic'
local hdf5 = require 'hdf5'
local image = require 'image'
local torch = require 'torch'
local __ = require 'moses'
require 'classic.torch'  -- Necessary for serializing classic classes.

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

local DiskFramesHdf5LabelsDataSource = classic.class(
    'DiskFramesHdf5LabelsDataSource', 'VideoDataSource')
function DiskFramesHdf5LabelsDataSource:_init(options)
    --[[
    Data source for loading frames from disk and labels from HDF5.

    Args (in options):
        frames_root (str): Contains subdirectories titled <video_name> for each
            video, which in turn contain frames of the form frame%04d.png
        labels_hdf5 (str or num): If str, specifies path to HDF5 file
            containing <video_name> keys, with (num_frames, num_labels) binary
            label matrices as values. If num, specifies number of labels, and
            we will set the labels matrix to be a matrix of all 1s. This is
            useful for running on images without labels.
    ]]--
    self.frames_root = options.frames_root
    local labels_hdf5 = options.labels_hdf5

    if type(labels_hdf5) == "number" then
        self.num_labels_ = labels_hdf5
        self.video_keys_ =
            DiskFramesHdf5LabelsDataSource.static.collect_video_frames(
                self.frames_root)
        self.num_samples_ = 0
        self.labels = {}
        for video, video_keys in pairs(self.video_keys_) do
            self.num_samples_ = self.num_samples_ + #video_keys
            self.labels[video] = torch.ones(#video_keys, self.num_labels_)
        end
    else
        local hdf5_labels_file = hdf5.open(labels_hdf5, 'r')
        self.labels = hdf5_labels_file:all()
        self.num_labels_ = self.labels[__.keys(self.labels)[1]]:size(2)

        self.video_keys_ = {}
        self.num_samples_ = 0
        for video_name, video_labels in pairs(self.labels) do
            local num_frames = video_labels:size(1)
            self.video_keys_[video_name] = {}
            for i = 1, num_frames do
                table.insert(self.video_keys_[video_name], video_name .. '-' .. i)
            end
            self.num_samples_ = self.num_samples_ + num_frames
        end
    end
end

function DiskFramesHdf5LabelsDataSource.static.collect_video_frames(path)
    local paths = require 'paths'
    local video_keys = {}
    for video in paths.iterdirs(path) do
        video_keys[video] = {}
        for frame in paths.iterfiles(path .. '/' .. video) do
            if string.match(frame, 'frame[0-9]+') ~= nil then
                local index = string.match(frame, '[0-9]+')
                video_keys[video][tonumber(index)] = video .. '-' .. index
            end
        end
    end
    return video_keys
end

function DiskFramesHdf5LabelsDataSource:num_labels() return self.num_labels_ end

function DiskFramesHdf5LabelsDataSource:video_keys()
    return self.video_keys_
end

function DiskFramesHdf5LabelsDataSource:key_label_map(return_label_map)
    --[[
    Load mapping from frame keys to labels array.

    Note: This is a giant array, and should be destroyed as soon as it is no
    longer needed. If this array is stored permanently (e.g. globally or as an
    object attribute), it will slow down *all* future calls to collectgarbage().

    Args:
        return_label_map (bool): If true, return a map from label names to label
            id.

    Returns:
        key_labels: Table mapping frame keys to array of label indices.
        (optional) label_map: See doc for return_label_map arg.

    ]]--
    assert(not return_label_map)
    local key_labels = {}
    for video_name, video_labels in pairs(self.labels) do
        for i = 1, video_labels:size(1) do
            local key = video_name .. '-' .. i
            local squeezed = video_labels[{i, {}}]:nonzero():squeeze()
            if torch.isTensor(squeezed) then
                squeezed = squeezed:totable()
            else
                squeezed = {squeezed}
            end
            local labels = squeezed
            key_labels[key] = labels
        end
    end
    return key_labels
end

function DiskFramesHdf5LabelsDataSource:load_data(keys, load_images)
    --[[
    Load images and labels for a set of keys.

    Args:
        keys (array): Array of array of string keys. Each element must be
            an array of the same length as every element, and contains keys for
            one step of the image sequence.
        load_images (bool): Defaults to true. If false, load only labels,
            not images. The ByteTensors in batch_images will simply be empty.

    Returns:
        batch_images: Array of array of ByteTensors
        batch_labels: ByteTensor of shape (num_steps, batch_size, num_labels)
    ]]--
    load_images = load_images == nil and true or load_images
    local num_steps = #keys
    local batch_size = #keys[1]
    local batch_labels = torch.ByteTensor(
        num_steps, batch_size, self.num_labels_)
    local batch_images = {}
    for step = 1, num_steps do
        batch_images[step] = {}
    end

    for step, step_keys in ipairs(keys) do
        for sequence, key in ipairs(step_keys) do
            if key == VideoDataSource.END_OF_SEQUENCE then
                table.insert(batch_images[step],
                             VideoDataSource.END_OF_SEQUENCE)
                batch_labels[{step, sequence}]:zero()
            else
                local video_name, frame_number = self:frame_video_offset(key)
                if load_images then
                    local frame_path = string.format('%s/%s/frame%04d.png',
                                                    self.frames_root,
                                                    video_name,
                                                    frame_number)
                    local frame = image.load(
                        frame_path, 3 --[[depth]], 'byte' --[[type]])
                    -- For backwards compatibility, use BGR images.
                    frame = frame:index(1, torch.LongTensor{3, 2, 1})
                    batch_images[step][sequence] = frame
                else
                    batch_images[step][sequence] = torch.ByteTensor()
                end
                batch_labels[{step, sequence, {}}] =
                    self.labels[video_name][frame_number]
            end
        end
    end
    return batch_images, batch_labels
end

-- luacheck: push no unused args
function DiskFramesHdf5LabelsDataSource:frame_video_offset(key)
    return DiskFramesHdf5LabelsDataSource.static.parse_frame_key(key)
end
-- luacheck: pop

function DiskFramesHdf5LabelsDataSource:num_samples()
    return self.num_samples_
end

function DiskFramesHdf5LabelsDataSource.static.parse_frame_key(frame_key)
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
    DiskFramesHdf5LabelsDataSource = DiskFramesHdf5LabelsDataSource,
    END_OF_SEQUENCE = VideoDataSource.END_OF_SEQUENCE
}
