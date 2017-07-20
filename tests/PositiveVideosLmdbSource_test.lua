-- Not really a test, just a simple check for helping with debugging.

local data_loader = require 'data_loader'
local data_source = require 'data_source'
local samplers = require 'samplers'
local lyaml = require 'lyaml'
require 'lmdb_data_source

local data_paths = lyaml.load(io.open('config/data_paths.yaml', 'r'):read('*a'))

local source = data_source.PositiveVideosLmdbSource(
    data_paths.train_split.without_images,
    data_paths.train_split.without_images,
    65 --[[num_labels]],
    {labels = {'VolleyballSpiking'}})

local sampler = samplers.PermutedSampler(source, 1, 1, false)
local keys = sampler:sample_keys(5)
print(keys)

local _, labels = source:load_data(keys, false --[[ load_images ]])
print(labels)
print(labels:size())
