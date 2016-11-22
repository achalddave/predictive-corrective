--[[ Computes activations for a given video.
--
-- Outputs a tensor which contains the output feature_map for each frame. The
-- first dimension is of size #frames.
--]]

local argparse = require 'argparse'
local classic = require 'classic'
local cunn = require 'cunn'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local image = require 'image'
local lmdb = require 'lmdb'
local torch = require 'torch'
local nn = require 'nn'
rnn = require 'rnn'
require 'layers/init'

local data_loader = require 'data_loader'

local parser = argparse() {
    description = 'Computes activations for a given video for a VGG network.'
}
parser:option('--model', 'Torch model')
parser:option('--layer_spec',
              'Layer specification. This is a ">" separated list of '  ..
              'specifications of the form "layer_type:index". ' ..
              'For example, ' ..
              '"nn.ParallelTable:2>cudnn.SpatialConvolution:13"' ..
              'will select the 13th SpatialConvolution layer in the 2nd ' ..
              'ParallelTable module.')
parser:option('--groundtruth_lmdb')
parser:option('--video_name')
parser:option('--output_activations')
parser:option('--sequence_length', 'Number of input frames.')
    :count('?'):default(1):convert(tonumber)
parser:option('--step_size', 'Size of step between frames.')
    :count('?'):default(1):convert(tonumber)
parser:option('--num_labels', 'Number of labels. Default: 65 (MultiTHUMOS)')
    :count('?'):default(65):convert(tonumber)
parser:option('--batch_size'):convert(tonumber):default(64)
parser:flag('--decorate_sequencer')

local args = parser:parse()

---
-- Configuration
---
-- Only supports one GPU. I can't figure out how to get activations from an
-- arbitrary layer on multiple GPUs easily.
local GPU = 1
nn.DataParallelTable.deserializeNGPUs = 1
local MEANS = {96.8293, 103.073, 101.662}
local CROP_SIZE = 224
-- Unfortunately, this is necessary due to the way DataLoader is implemented.

math.randomseed(0)
torch.manualSeed(0)
cutorch.manualSeedAll(0)
cutorch.setDevice(GPU)
torch.setdefaulttensortype('torch.FloatTensor')

---
-- Load list of frame keys for video.
---
local VideoSampler = classic.class('VideoSampler', data_loader.Sampler)
function VideoSampler:_init(
    frames_lmdb, _ --[[num_labels]], sequence_length, step_size,
    use_boundary_frames, options)
    --[[ Return consecutive frames from a given video.

    Args:
        options:
            - video_name
    ]]--
    self.video_name = options.video_name
    self.video_keys = VideoSampler.get_video_keys(frames_lmdb, self.video_name)
    if not use_boundary_frames then
        self.video_keys = data_loader.PermutedSampler.filter_boundary_frames(
            self.video_keys, sequence_length, step_size)
    end
    self.sequence_length = sequence_length
    self.step_size = step_size
    self.key_index = 1
end

function VideoSampler:num_samples()
    return #self.video_keys
end

function VideoSampler:sample_keys(num_sequences)
    local batch_keys = {}
    for _ = 1, self.sequence_length do
        table.insert(batch_keys, {})
    end
    for _ = 1, num_sequences do
        if self.key_index > self:num_samples() then
            self.key_index = 1
        end
        local sampled_key = self.video_keys[self.key_index]
        for step = 1, self.sequence_length do
            table.insert(batch_keys[step], sampled_key)
            sampled_key = data_loader.Sampler.frame_offset_key(
                sampled_key, self.step_size)
        end
        self.key_index = self.key_index + 1
    end
    return batch_keys
end

function VideoSampler.static.get_video_keys(frames_lmdb, video_name)
    local video_keys = {}
    local db = lmdb.env { Path = frames_lmdb }
    db:open()
    local transaction = db:txn(true --[[readonly]])
    local key = video_name .. '-1'
    while transaction:get(key) ~= nil do
        table.insert(video_keys, key)
        key = data_loader.Sampler.next_frame_key(key)
    end
    return video_keys
end

---
-- Load model.
---
print('Loading model.')
local model = torch.load(args.model)
if torch.isTypeOf(model, 'nn.DataParallelTable') then
    model = model:get(1)
end
if args.decorate_sequencer then
    if torch.isTypeOf(model, 'nn.Sequencer') then
        print('WARNING: --decorate_sequencer on model that is already ' ..
              'nn.Sequencer!')
    end
    model = nn.Sequencer(model)
end

model:evaluate()
print('Loaded model.')

---
-- Get requested layer.
---
print('Layer specification:', args.layer_spec)
-- Find specifications between ">" characters.
local search_module = model
for specification in string.gmatch(args.layer_spec, "[^>]+") do
    -- Split on "," character, returns an iterator.
    local spec_parser = string.gmatch(specification, "[^,]+")
    local layer_type = spec_parser()
    local layer_index = tonumber(spec_parser())
    print(layer_type, layer_index)
    search_module = search_module:findModules(layer_type)[layer_index]
end
local layer_to_extract = search_module
print('Extracting from layer:', layer_to_extract)

---
-- Pass frames through model
---
local sampler = VideoSampler(
    args.groundtruth_lmdb,
    nil --[[num_labels]],
    args.sequence_length,
    args.step_size,
    true --[[use boundary frames]],
    {video_name = args.video_name})
local loader = data_loader.DataLoader(
    args.groundtruth_lmdb, sampler, args.num_labels)

local gpu_inputs = torch.CudaTensor()
local samples_complete = 0
local relus = model:findModules('cudnn.ReLU')
-- Disable in-place ReLUs so that we don't accidentally compute the ReLU'd
-- version of activations.
for _, relu in ipairs(relus) do
    relu.inplace = false
end
print('Disabled in-place ReLUs.')

local frame_activations = nil
while samples_complete ~= sampler:num_samples() do
    local to_load = args.batch_size
    if samples_complete + args.batch_size > sampler:num_samples() then
        to_load = sampler:num_samples() - samples_complete
    end
    local images_table, _, batch_keys = loader:load_batch(
        to_load, true --[[return_keys]])
    -- TODO(achald): This assumes that the layer we extract maps to the last
    -- frame. That's not necessarily true!
    batch_keys = batch_keys[args.sequence_length]

    local batch_size = #images_table[1]
    local num_channels = images_table[1][1]:size(1)
    local images = torch.Tensor(args.sequence_length, batch_size,
                                num_channels, CROP_SIZE, CROP_SIZE)
    for step, step_images in ipairs(images_table) do
        for batch_index, img in ipairs(step_images) do
            -- Process image after converting to the default Tensor type.
            -- (Originally, it is a ByteTensor).
            img = img:typeAs(images)
            for channel = 1, 3 do
                img[{{channel}, {}, {}}]:add(-MEANS[channel])
            end
            images[{step, batch_index}] = image.crop(
                img, 'c', CROP_SIZE, CROP_SIZE)
        end
    end

    gpu_inputs:resize(images:size()):copy(images)
    model:forward(gpu_inputs)
    if frame_activations == nil then
        local activation_map_size = torch.totable(
            layer_to_extract.output[1]:size())
        frame_activations = torch.Tensor(
            sampler:num_samples() + args.sequence_length - 1,
            unpack(activation_map_size))
    end
    for i = 1, batch_size do
        local _, frame_number = data_loader.Sampler.parse_frame_key(
            batch_keys[i])
        frame_activations[frame_number] = layer_to_extract.output[i]:float()
    end
    samples_complete = samples_complete + to_load
    print(string.format('Computed activations for %d/%d.',
                        samples_complete, sampler:num_samples()))
    collectgarbage()
    collectgarbage()
end
print('Finished computing activations.')

---
-- Create activations tensor.
---
print('Saving activations to disk.')
torch.save(args.output_activations, frame_activations)
