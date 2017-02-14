--[[ Compute distribution of labels after a number of samples. --]]

local argparse = require 'argparse'
local torch = require 'torch'

local data_source = require 'data_source'
local data_loader = require 'data_loader'

local parser = argparse() {
    description = 'Compute distribution of labels from samples.'
}
parser:argument('groundtruth_without_images_lmdb')
parser:option('--num_labels'):default(65)
parser:option('--num_samples'):default(100):convert(tonumber)
parser:option('--background_weight'):default(20)
parser:option('--output_counts')

local args = parser:parse()

local source = data_source.LabeledVideoFramesLmdbSource(
    args.groundtruth_without_images_lmdb,
    args.groundtruth_without_images_lmdb,
    args.num_labels)
local sampler = data_loader.BalancedSampler(
    source,
    1 --[[sequence_length]],
    1 --[[step_size]],
    true --[[use_boundary_frames]],
    {background_weight = args.background_weight})

local sampled_keys = sampler:sample_keys(args.num_samples)
local _, sampled_labels = source:load_data(sampled_keys)
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
