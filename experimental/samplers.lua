local classic = require 'classic'
local threads = require 'threads'
local torch = require 'torch'
require 'classic.torch'

local data_loader = require 'data_loader'
local log = require 'util/log'

local Sampler = data_loader.Sampler
local BalancedSampler = data_loader.BalancedSampler

-- TODO(achald): Implement greedy balancing sampler.
local GreedyBalancingSampler, GreedyBalancingSamplerSuper =
    classic.class('GreedyBalancingSampler', BalancedSampler)
function GreedyBalancingSampler:_init(
        data_source_obj,
        sequence_length,
        step_size,
        use_boundary_frames,
        options)

    GreedyBalancingSamplerSuper._init(self,
        data_source_obj,
        sequence_length,
        step_size,
        use_boundary_frames,
        options)

    self.key_label_map = self.data_source:key_label_map()
end

function GreedyBalancingSampler:sample_keys(num_sequences)
    --[[
    Returns:
        batch_keys (Array of array of strings): Each element contains
            num_sequences arrays, each of which contains sequence_length keys.
    ]]--
    local batch_keys = {}
    for _ = 1, self.sequence_length do
        table.insert(batch_keys, {})
    end
    local batch_keys = {}
    for _ = 1, self.sequence_length do
        table.insert(batch_keys, {})
    end
    local weights = self.label_weights:clone()
    for sequence = 1, num_sequences do
        local label = torch.multinomial(weights, 1)[1]
        local label_key_index = self.label_indices[label]
        -- We sample the _end_ of the sequence based on the labels, and build
        -- the sequence backwards.
        local sampled_key = self.label_key_map[label][label_key_index]
        local video, offset = self.data_source:frame_video_offset(sampled_key)
        for _, label in ipairs(self.key_label_map[sampled_key]) do
            weights[label] = weights[label] * 0.1
        end
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
    -- local desired_distribution = self.label_weights:clone()
    -- desired_distribution = (
    --     desired_distribution / desired_distribution:sum())
    -- local batch_counts = torch.zeros(self.num_labels + 1)
    -- local batch_distribution = batch_counts:clone()

    -- for sequence = 1, num_sequences do
    --     --[[ We want to sample from a distribution p^s such that
    --     --   k/n * p^b + (1 - k/n) * p^s = p^d
    --     --   => p^s = (n p^d - k p^b) / (n - k)
    --     --  where
    --     --      k: number of elements in our batch so far (sequence - 1)
    --     --      n: total size of the batch (num_sequences)
    --     --      p^b: current batch distribution (batch_distribution)
    --     --      p^d: desired distribution (desired_distribution)
    --     --
    --     --  TODO(achald): Something about this seems fishy. It assumes that each
    --     --  data point in the batch has the same influence on the distribution,
    --     --  which isn't true since the influence is proportional to the number
    --     --  of labels. But I'm not sure what the right way to fix this is.
    --     --]]
    --     print(desired_distribution)
    --     print(batch_distribution)
    --     print(desired_distribution * num_sequences)
    --     local weights = (desired_distribution * num_sequences
    --         - (sequence - 1) * batch_distribution
    --         ) / (num_sequences - sequence + 1)
    --     local label = torch.multinomial(weights, 1)[1]
    --     local label_key_index = self.label_indices[label]
    --     -- We sample the _end_ of the sequence based on the labels, and build
    --     -- the sequence backwards.
    --     local sampled_key = self.label_key_map[label][label_key_index]
    --     for _, label in ipairs(self.key_label_map[sampled_key]) do
    --         batch_counts[label] = batch_counts[label] + 1
    --     end
    --     batch_distribution = batch_counts / batch_counts:sum()

    --     local video, offset = self.data_source:frame_video_offset(sampled_key)
    --     local last_valid_key
    --     for step = self.sequence_length, 1, -1 do
    --         -- If the key exists, use it. Otherwise, use the last frame we have.
    --         if self.video_keys[video][offset] ~= nil then
    --             last_valid_key = sampled_key
    --         elseif not self.use_boundary_frames then
    --             -- If we aren't using boundary frames, we shouldn't run into
    --             -- missing keys!
    --             error('Missing key:', sampled_key)
    --         end
    --         table.insert(batch_keys[step], last_valid_key)
    --         offset = offset - self.step_size
    --         sampled_key = self.video_keys[video][offset]
    --     end
    --     self:_advance_label_index(label)
    -- end
end


--[[
--
-- For each label, maintain a set of frames that have that label, and a
-- separate set of 'seen' frames. At each iteration, do
-- the following until enough frames have been sampled.
--  1. L <- Sample a label uniformly
--  2. F <- Sample a frame containing label L
--  3. If F in seen_L: return to step 1
--  4. Else: Add F to seen_{L'} for all L' assigned to F.
--  5. Add F to sampled frames, repeat from 1.
-- Keep a table marking the 'seen' frames for each label. When we sample a
-- frame, mark it as 'seen' for all the labels it has. Then, if that frame
-- is ever sampled, ignore it and resample the label.
--
-- TODO(achald): Explain this better.
-- ]]
local MarkSeenBalancingSampler, MarkSeenBalancingSamplerSuper =
    classic.class('MarkSeenBalancingSampler', BalancedSampler)
function MarkSeenBalancingSampler:_init(
        data_source_obj,
        sequence_length,
        step_size,
        use_boundary_frames,
        options)

    MarkSeenBalancingSamplerSuper._init(self,
        data_source_obj,
        sequence_length,
        step_size,
        use_boundary_frames,
        options)

    self.key_label_map = self.data_source:key_label_map()
    self.seen_frames = {}
    for i = 1, self.num_labels + 1 do
        self.seen_frames[i] = {}
    end
end

function MarkSeenBalancingSampler:sample_keys(num_sequences)
    --[[
    Returns:
        batch_keys (Array of array of strings): Each element contains
            num_sequences arrays, each of which contains sequence_length keys.
    ]]--
    local batch_keys = {}
    for _ = 1, self.sequence_length do
        table.insert(batch_keys, {})
    end
    local batch_keys = {}
    for _ = 1, self.sequence_length do
        table.insert(batch_keys, {})
    end
    local sampled_sequences = 0
    while sampled_sequences < num_sequences do
        local label = torch.multinomial(self.label_weights, 1)[1]
        local label_key_index = self.label_indices[label]
        -- We sample the _end_ of the sequence based on the labels, and build
        -- the sequence backwards.
        local sampled_key = self.label_key_map[label][label_key_index]
        if self.seen_frames[label][sampled_key] == nil then
            local video, offset = self.data_source:frame_video_offset(
                sampled_key)
            for _, label in ipairs(self.key_label_map[sampled_key]) do
                self.seen_frames[label][sampled_key] = true
            end
            local last_valid_key
            for step = self.sequence_length, 1, -1 do
                -- If the key exists, use it. Otherwise, use the last frame we
                -- have.
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
            sampled_sequences = sampled_sequences + 1
        end
        self:_advance_label_index(label)
    end
    return batch_keys
end

function MarkSeenBalancingSampler:_advance_label_index(label)
    if self.label_indices[label] + 1 <= #self.label_key_map[label] then
        self.label_indices[label] = self.label_indices[label] + 1
    else
        log.info('Refreshing seen_frames')
        local num = 0
        for _, _ in pairs(self.seen_frames[label]) do num = num + 1 end
        -- log.info(string.format(
        --     'Saw %d frames, had %d.', num, #self.label_key_map[label]))
        assert(num == #self.label_key_map[label])
        self.seen_frames[label] = {}
        self.label_key_map[label] = Sampler.permute(self.label_key_map[label])
        self.label_indices[label] = 1
    end
end

data_loader.GreedyBalancingSampler = GreedyBalancingSampler
data_loader.MarkSeenBalancingSampler = MarkSeenBalancingSampler
