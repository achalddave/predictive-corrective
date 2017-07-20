--[[ Hacky script I use for debugging samplers. ]]--

local argparse = require 'argparse'
local torch = require 'torch'

local data_source = require 'data_source'
local data_loader = require 'data_loader'
local log = require 'util/log'
local samplers = require 'samplers'
require 'lmdb_data_source'

local parser = argparse() {
    description = 'Compute distribution of labels from samples.'
}
parser:argument('groundtruth_without_images_lmdb')
parser:option('--num_labels'):default(65):convert(tonumber)

local args = parser:parse()

-- log.info('Creating source')
-- local source = data_source.SubsampledLmdbSource(
--     args.groundtruth_without_images_lmdb,
--     args.groundtruth_without_images_lmdb,
--     args.num_labels,
--     {subsample_rate=128})
-- log.info('Created source')

for _, rate in ipairs({132, 133, 134, 135, 136}) do
    log.info('Creating source with subsample rate', rate)
    local source = data_source.SubsampledLmdbSource(
        args.groundtruth_without_images_lmdb,
        args.groundtruth_without_images_lmdb,
        args.num_labels,
        {subsample_rate=rate})
    log.info('Created source')
    samplers.BalancedSampler(
        source,
        1 --[[sequence_length]],
        1 --[[step_size]],
        true --[[use_boundary_frames]],
        {background_weight = 20})
end


-- local sampler = samplers.AdaptiveMemorySampler(source,
--                                                1 --[[sequence_length]],
--                                                1 --[[step_size]],
--                                                true --[[use_boundary_frames]],
--                                                {memory_size=8})
-- local sampler = samplers.ReplayMemorySampler(source,
--                                              1 --[[sequence_length]],
--                                              1 --[[step_size]],
--                                              true --[[use_boundary_frames]])
-- sampler:sample_keys(source:num_samples())

-- local sampler = samplers.SequentialBatchSampler(
--     source,
--     2 --[[sequence_length]],
--     1 --[[step_size]],
--     true --[[use_boundary_frames]])

local sampler = samplers.BalancedSampler(
    source,
    1 --[[sequence_length]],
    1 --[[step_size]],
    true --[[use_boundary_frames]],
    {background_weight = 20})

-- local sampler = samplers.PermutedSampler(
--     source,
--     1 --[[sequence_length]],
--     1 --[[step_size]],
--     true --[[use_boundary_frames]],
--     {replace = False})

-- local sampler = samplers.UniformlySpacedSampler(
--     source,
--     1 --[[sequence_length]],
--     nil --[[step_size]],
--     nil --[[use_boundary_frames]],
--     {num_frames_per_video = 25})

sampler:sample_keys(8)
log.info(sampler.sample_rate)
for _ = 1, 2 do
    log.info('sample rate')
    log.info(sampler.sample_rate)
    log.info('sampled')
    log.info(sampler:sample_keys(1))
    log.info('memory')
    log.info(sampler.memory)
end
