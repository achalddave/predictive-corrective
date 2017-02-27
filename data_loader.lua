--[[
Helper class to load data and labels from an LMDB containing LabeledVideoFrames
as values.
]]--

local classic = require 'classic'
local threads = require 'threads'
local torch = require 'torch'
require 'classic.torch'

local data_source = require 'data_source'
local log = require 'util/log'

local END_OF_SEQUENCE = data_source.VideoDataSource.END_OF_SEQUENCE

local Sampler = classic.class('Sampler')
Sampler:mustHave('sample_keys')
Sampler:mustHave('num_samples')
Sampler:mustHave('num_labels')

function Sampler.static.permute(list)
    local permuted_list = {}
    local permutation = torch.randperm(#list)
    for i = 1, permutation:nElement() do
        table.insert(permuted_list, list[permutation[i]])
    end
    return permuted_list
end

local PermutedSampler = classic.class('PermutedSampler', Sampler)
function PermutedSampler:_init(
        data_source_obj, sequence_length, step_size, use_boundary_frames,
        options)
    --[[
    Sample frames randomly, with or without replacement.

    Args:
        data_source_obj (DataSource)
        sequence_length (num): If provided, sample sequences of length
            sequence_length for each training sample.
        step_size (num): If provided, elements in the sequence should be
            separated by this step_size. If step_size is 2, a sequence of length
            5 starting at x_1 is {x_1, x_3, x_5, x_7, x_9}.
        use_boundary_frames (bool): Default false. If false, avoid sequences
            that go outside the video temporally. Otherwise, for sequences at
            the boundary, we replicate the first or last frame of the video.
        options:
            replace (bool): If true, sample each frame i.i.d. with replacement.
                If false, do not re-sample a frame until all other frames have
                been sampled.
    ]]--
    options = options == nil and {} or options
    self.sequence_length = sequence_length == nil and 1 or sequence_length
    self.step_size = step_size == nil and 1 or step_size
    self.use_boundary_frames = use_boundary_frames == nil and
                               false or use_boundary_frames

    self.video_keys = data_source_obj:video_keys()
    self.data_source = data_source_obj
    self.replace = options.replace == nil and false or options.replace

    -- TODO: Don't store two copies of keys.
    self.keys = {}
    for _, keys in pairs(self.video_keys) do
        for _, key in ipairs(keys) do
            table.insert(self.keys, key)
        end
    end

    if not self.use_boundary_frames then
        self.keys = PermutedSampler.filter_boundary_frames(
            self.video_keys, self.sequence_length, self.step_size)
    end

    self:_refresh_keys()
end

function PermutedSampler:_refresh_keys()
    if self.replace then
        self.key_order = torch.multinomial(
            torch.ones(#self.keys), #self.keys, true --[[replace]])
    else
        -- Using torch.multinomial here with replace set to false is incredibly
        -- slow, for some reason.
        self.key_order = torch.randperm(#self.keys)
    end
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
            log.info(string.format(
                '%s: Finished pass through data, repermuting!', os.date('%X')))
            self:_refresh_keys()
        end
        local sampled_key = self.keys[self.key_order[self.key_index]]
        local video, offset = self.data_source:frame_video_offset(sampled_key)
        local last_valid_key
        for step = 1, self.sequence_length do
            -- If the key exists, use it. Otherwise, use the last frame we have.
            if self.video_keys[video][offset] ~= nil then
                last_valid_key = sampled_key
            elseif not self.use_boundary_frames then
                -- If we aren't using boundary frames, we shouldn't run into
                -- missing keys!
                error('Missing key:', sampled_key)
            end
            table.insert(batch_keys[step], last_valid_key)
            offset = offset + self.step_size
            sampled_key = self.video_keys[video][offset]
        end
        self.key_index = self.key_index + 1
    end
    return batch_keys
end

function PermutedSampler:num_labels()
    return self.data_source:num_labels()
end

function PermutedSampler:num_samples()
    return #self.keys
end

function PermutedSampler.static.filter_boundary_frames(
        video_keys, sequence_length, step_size)
    --[[ Filter out the last sequence_length frames in the video.
    --
    -- Args:
    --     video_keys (array of arrays): Maps video name to array of keys for
    --         frames in the video.
    --]]
    local keys = {}
    for _, keys_in_video in pairs(video_keys) do
        if step_size > 0 then
            -- Remove the last ((sequence_length - 1) * step_size) keys.
            for i = 1, #keys_in_video - (sequence_length - 1) * step_size do
                local key = keys_in_video[i]
                table.insert(keys, key)
            end
        elseif step_size < 0 then
            -- Remove the first ((sequence_length - 1) * step_size) keys.
            for i = 1 - (sequence_length - 1) * step_size, #keys_in_video do
                local key = keys_in_video[i]
                table.insert(keys, key)
            end
        end
    end
    return keys
end

local BalancedSampler = classic.class('BalancedSampler', Sampler)
function BalancedSampler:_init(
        data_source_obj,
        sequence_length,
        step_size,
        use_boundary_frames,
        options)
    --[[
    Samples from each class a balanced number of times, so that the model should
    see approximately the same amount of data from each class.

    If sequence_length is >1, then the label for the _last_ frame in the
    sequence is used for balancing classes.

    Args:
        data_source_obj (DataSource)
        sequence_length (num): If provided, sample sequences of length
            sequence_length for each training sample.
        step_size (num): If provided, elements in the sequence should be
            separated by this step_size. If step_size is 2, a sequence of length
            5 starting at x_1 is {x_1, x_3, x_5, x_7, x_9}.
        use_boundary_frames (bool): Default false. If false, avoid sequences
            that go outside the video temporally. Otherwise, for sequences at
            the boundary, we replicate the first or last frame of the video.
        options (table):
            background_weight (int): Indicates weight for sampling background frames.
                If this is 1, for example, , we sample background frames as
                often as frames from any particular label
                (i.e. with probability 1/(num_labels + 1)).
            DEPRECATED include_bg (bool): If true, background_weight is set to
                1; if false, background_weight is set to 0.
    ]]--
    self.data_source = data_source_obj
    self.num_labels_ = data_source_obj:num_labels()
    options = options == nil and {} or options
    self.sequence_length = sequence_length
    self.step_size = step_size
    self.use_boundary_frames = use_boundary_frames == nil and
                               false or use_boundary_frames

    assert(options.include_bg == nil,
           'include_bg is deprecated; use background_weight instead.')

    -- List of all valid keys.
    self.video_keys = self.data_source:video_keys()
    local valid_keys = {}
    if not self.use_boundary_frames then
        valid_keys = PermutedSampler.filter_boundary_frames(
            self.video_keys, sequence_length, -step_size)
    else
        for _, keys in pairs(self.video_keys) do
            for _, key in ipairs(keys) do
                table.insert(valid_keys, key)
            end
        end
    end
    self.num_keys = #valid_keys

    -- Map labels to list of keys containing that label.
    local key_label_map = self.data_source:key_label_map()
    self.label_key_map = {}
    for i = 1, self.num_labels_+1 do self.label_key_map[i] = {} end
    for _, key in ipairs(valid_keys) do
        for _, label in ipairs(key_label_map[key]) do
            table.insert(self.label_key_map[label], key)
        end
    end

    self.label_weights = torch.ones(self.num_labels_ + 1)
    self.label_weights[self.num_labels_ + 1] =
        options.background_weight == nil and 0 or options.background_weight

    -- For each label, maintain an index of the next data point to output.
    self.label_indices = {}
    self:_permute_keys()
end

function BalancedSampler:sample_keys(num_sequences)
    --[[
    Returns:
        batch_keys (Array of array of strings): Each element contains
            num_sequences arrays, each of which contains sequence_length keys.
    ]]--
    local batch_keys = {}
    for _ = 1, self.sequence_length do
        table.insert(batch_keys, {})
    end
    local sampled_labels = torch.multinomial(
        self.label_weights, num_sequences, true --[[replace]])
    for sequence = 1, num_sequences do
        local label = sampled_labels[sequence]
        local label_key_index = self.label_indices[label]
        -- We sample the _end_ of the sequence based on the labels, and build
        -- the sequence backwards.
        local sampled_key = self.label_key_map[label][label_key_index]
        local video, offset = self.data_source:frame_video_offset(sampled_key)
        local last_valid_key
        for step = self.sequence_length, 1, -1 do
            -- If the key exists, use it. Otherwise, use the last frame we have.
            if self.video_keys[video][offset] ~= nil then
                last_valid_key = sampled_key
            elseif not self.use_boundary_frames then
                -- If we aren't using boundary frames, we shouldn't run into
                -- missing keys!
                error('Missing key:', sampled_key)
            end
            table.insert(batch_keys[step], last_valid_key)
            offset = offset - self.step_size
            sampled_key = self.video_keys[video][offset]
        end
        self:_advance_label_index(label)
    end
    return batch_keys
end

function BalancedSampler:num_labels()
    return self.data_source:num_labels()
end

function BalancedSampler:num_samples()
    return self.num_keys
end

function BalancedSampler:_advance_label_index(label)
    if self.label_indices[label] + 1 <= #self.label_key_map[label] then
        self.label_indices[label] = self.label_indices[label] + 1
    else
        self.label_key_map[label] = Sampler.permute(self.label_key_map[label])
        self.label_indices[label] = 1
    end
end

function BalancedSampler:_permute_keys()
    for i = 1, self.num_labels_ + 1 do
        self.label_key_map[i] = Sampler.permute(self.label_key_map[i])
        self.label_indices[i] = 1
    end
end

-- TODO(achald): Implement sequential sampler. This will choose
-- `batch_size` videos, and then emit consecutive sequences from these videos.
-- So each batch will contain sequence_length frames from batch_size videos.
-- If a video has less than batch_size frames left, the batch will be padded
-- with 'nil' keys.
local SequentialSampler = classic.class('SequentialSampler', Sampler)
function SequentialSampler:_init(
        data_source_obj, sequence_length, step_size,
        _ --[[use_boundary_frames]], options)
    --[[
    Returns consecutives sequences of frames from videos.

    Args:
        data_source_obj (DataSource)
        sequence_length (num): If provided, sample sequences of length
            sequence_length for each training sample.
        step_size (num): If provided, elements in the sequence should be
            separated by this step_size. If step_size is 2, a sequence of length
            5 starting at x_1 is {x_1, x_3, x_5, x_7, x_9}.
        use_boundary_frames (bool): Ignored for SequentialSampler.
        options:
            batch_size (int): Must be specified a-priori and cannot be changed.
            sample_once (bool): If true, only do one pass through the videos.
                Useful for evaluating.
    ]]--
    assert(options.batch_size ~= nil)
    self.sequence_length = sequence_length == nil and 1 or sequence_length
    self.step_size = step_size == nil and 1 or step_size
    self.batch_size = options.batch_size
    self.sample_once = options.sample_once
    self.sampled_all_videos = false

    self.video_keys = data_source_obj:video_keys()
    self.data_source = data_source_obj

    -- TODO(achald): Should we sort these by length of videos?
    self.video_start_keys = {}
    for _, keys_for_video in pairs(self.video_keys) do
        table.insert(self.video_start_keys, keys_for_video[1])
    end
    self.video_start_keys = Sampler.permute(self.video_start_keys)

    self.next_frames = {}
    for i = 1, self.batch_size do
        self.next_frames[i] = self.video_start_keys[i]
    end
    -- Set to the last video that we are currently outputting; when a video
    -- ends, this will be advanced by 1 and a new video will be output.
    self.video_index = self.batch_size
end

function SequentialSampler:advance_video_index(offset)
    if offset == nil then offset = 1 end
    self.video_index = self.video_index + offset
    if self.video_index > #self.video_start_keys then
        self.sampled_all_videos = true
        if not self.sample_once then
            log.info(string.format(
                '%s: Finished pass through videos, repermuting!',
                os.date('%X')))
            self.video_start_keys = Sampler.permute(
                self.video_start_keys)
            self.video_index = 1
        end
    end
end

function SequentialSampler:update_start_frame(sequence)
    if self.sample_once and self.sampled_all_videos then
        -- Don't sample any more frames.
        self.next_frames[sequence] = nil
    else
        self.next_frames[sequence] =
            self.video_start_keys[self.video_index]
    end
end

function SequentialSampler:sample_keys(num_sequences)
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
    assert(num_sequences == self.batch_size,
           string.format('Expected batch size %s, received %s',
                         self.batch_size, num_sequences))
    for sequence = 1, num_sequences do
        local sampled_key = self.next_frames[sequence]
        local sequence_valid = true
        local video, offset
        -- Add steps from the sequence to batch_keys until the sequence ends.
        for step = 1, self.sequence_length do
            if sampled_key ~= nil then
                video, offset = self.data_source:frame_video_offset(sampled_key)
                sequence_valid = self.video_keys[video][offset] ~= nil and
                                 sequence_valid
            else
                sequence_valid = false
            end
            if sequence_valid then
                table.insert(batch_keys[step], sampled_key)
            else
                table.insert(batch_keys[step], END_OF_SEQUENCE)
            end
            if sampled_key ~= nil then
                offset = offset + self.step_size
                sampled_key = self.video_keys[video][offset]
            end
        end
        local last_video, last_offset = self.data_source:frame_video_offset(
            batch_keys[#batch_keys][sequence])
        if sequence_valid then
            -- The sequence filled the batch with valid keys, so we want to
            -- output the sampled_key as the next sample.
            -- Note that sampled_key may be nil if the sequence just ended, in
            -- which case we will use the next batch to report the end of the
            -- sequence.
            self.next_frames[sequence] = sampled_key
        else
            -- Move to the next video.
            if not (self.sample_once and self.sampled_all_videos) then
                self:advance_video_index()
            end
            self:update_start_frame(sequence)
        end
    end
    return batch_keys
end

function SequentialSampler:num_labels()
    return self.data_source:num_labels()
end


function SequentialSampler:num_samples()
    return self.data_source:num_samples()
end

function SequentialSampler.static.get_start_frames(keys)
    local start_frame_keys = {}
    for _, key in ipairs(keys) do
        local _, frame_number = Sampler.parse_frame_key(key)
        if frame_number == 1 then
            table.insert(start_frame_keys, key)
        end
    end
    return start_frame_keys
end

local DataLoader = classic.class('DataLoader')

function DataLoader:_init(data_source_obj, sampler)
    --[[
    Args:
        lmdb_path (str): Path to LMDB containing LabeledVideoFrames as values.
        sampler (Sampler): Sampler used for batches
    ]]--
    self.data_source = data_source_obj
    self.sampler = sampler
    self._prefetched_data = {
        batch_images = nil,
        batch_labels = nil
    }
    self._prefetching_thread = threads.Threads(1)
end

function DataLoader:num_labels()
    return self.sampler:num_labels()
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
    local data_source_obj = self.data_source

    self._prefetching_thread:addjob(
        function()
            require 'torch'
            require 'classic'
            require 'classic.torch'
            require 'data_source'
        end,
        function()
            local batch_images, batch_labels = data_source_obj:load_data(
                batch_keys)
            self._prefetched_data = {
                batch_images = batch_images,
                batch_labels = batch_labels,
                batch_keys = batch_keys
            }
        end)
end

function DataLoader:_data_fetched()
    -- Wait for possible fetching thread to finish.
    self._prefetching_thread:synchronize()
    return self._prefetched_data.batch_images ~= nil
end

return {
    DataLoader = DataLoader,
    Sampler = Sampler,
    BalancedSampler = BalancedSampler,
    PermutedSampler = PermutedSampler,
    SequentialSampler = SequentialSampler,
    END_OF_SEQUENCE = END_OF_SEQUENCE
}
