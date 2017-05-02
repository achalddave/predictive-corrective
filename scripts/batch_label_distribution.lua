--[[ Compute distribution of labels after a number of samples. --]]

local argparse = require 'argparse'
local torch = require 'torch'
local gnuplot = require 'gnuplot'

local data_source = require 'data_source'
local data_loader = require 'data_loader'
local log = require 'util/log'

local parser = argparse() {
    description = 'Compute distribution of labels from samples.'
}
parser:argument('groundtruth_without_images_lmdb')
parser:option('--num_labels'):default(65):convert(tonumber)
parser:option('--num_samples'):default(100):convert(tonumber)
parser:option('--output_counts')

local args = parser:parse()

log.info('Creating source')
local source = data_source.LabeledVideoFramesLmdbSource(
    args.groundtruth_without_images_lmdb,
    args.groundtruth_without_images_lmdb,
    args.num_labels)
log.info('Created source')

-- local sampler = data_loader.BalancedSampler(
--     source,
--     1 --[[sequence_length]],
--     1 --[[step_size]],
--     true --[[use_boundary_frames]],
--     {background_weight = 20})
local sampler = data_loader.PermutedSampler(
    source,
    1 --[[sequence_length]],
    1 --[[step_size]],
    true --[[use_boundary_frames]],
    {replace = False})
-- local sampler = data_loader.UniformlySpacedSampler(
--     source,
--     1 --[[sequence_length]],
--     nil --[[step_size]],
--     nil --[[use_boundary_frames]],
--     {num_frames_per_video = 25})

log.info('Created sampler')
local sampled_keys = sampler:sample_keys(sampler:num_samples())
log.info('Sampled keys')
local _, sampled_labels = source:load_data(sampled_keys)
log.info('Loaded labels')
sampled_labels = sampled_labels[1 --[[first step]]]

local counts = torch.zeros(args.num_labels + 1)
counts[{{1, args.num_labels}}] = torch.sum(sampled_labels:float(), 1)
-- Get number of background frames (rows which are all 0)
counts[args.num_labels + 1] = torch.sum(
    torch.eq(torch.sum(sampled_labels:float(), 2), 0))
-- gnuplot.bar(counts)
if args.output_counts ~= nil then
    torch.save(args.output_counts, counts)
end
