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

local data_source = require 'data_source'
local data_loader = require 'data_loader'
local log = require 'util/log'
local samplers = require 'samplers'
require 'lmdb_data_source'

local parser = argparse() {
    description = 'Computes activations for a given video for a VGG network.'
}
parser:option('--model', 'Torch model')
parser:option('--layer_spec',
              'Layer specification. This is a ">" separated list of '  ..
              'specifications of the form "layer_type,index". ' ..
              'For example, ' ..
              '"nn.ParallelTable,2>cudnn.SpatialConvolution,13"' ..
              'will select the 13th SpatialConvolution layer in the 2nd ' ..
              'ParallelTable module.')
parser:option('--groundtruth_lmdb')
parser:option('--groundtruth_without_images_lmdb')
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
-- local MEANS = {103.939, 116.779, 123.68} -- Gunnar's Charades model mean
-- local MEANS = {86.18377069, 93.74071911, 105.4525389} -- Charades
local MEANS = {94.57184865, 100.78170151, 101.76892795} -- train+val MultiTHUMOS
-- local MEANS = {96.8293, 103.073, 101.662} -- train MultiTHUMOS
local CROP_SIZE = 224
local output_log = args.output_activations .. '.log'
print(output_log)
log.outfile = output_log
log.info('Args:', args)

math.randomseed(0)
torch.manualSeed(0)
cutorch.manualSeedAll(0)
cutorch.setDevice(GPU)
torch.setdefaulttensortype('torch.FloatTensor')

---
-- Load list of frame keys for video.
---
local SingleVideoSampler = classic.class('SingleVideoSampler', samplers.Sampler)
function SingleVideoSampler:_init(
    data_source_obj, sequence_length, step_size, use_boundary_frames, options)
    --[[ Return consecutive frames from a given video.

    Args:
        options:
            - video_name
    ]]--
    self.video_name = options.video_name
    self.data_source = data_source_obj
    self.all_video_keys = self.data_source:video_keys()
    if not use_boundary_frames then
        self.all_video_keys =
            samplers.PermutedSampler.filter_boundary_frames(
                self.all_video_keys, sequence_length, step_size)
    end
    self.video_keys = self.all_video_keys[self.video_name]
    self.sequence_length = sequence_length
    self.step_size = step_size
    self.key_index = 1
end

function SingleVideoSampler:num_labels()
    return self.data_source:num_labels()
end

function SingleVideoSampler:num_samples()
    return #self.video_keys
end

function SingleVideoSampler:sample_keys(num_sequences)
    local batch_keys = {}
    for _ = 1, self.sequence_length do
        table.insert(batch_keys, {})
    end
    for _ = 1, num_sequences do
        if self.key_index > self:num_samples() then
            self.key_index = 1
        end
        local sampled_key = self.video_keys[self.key_index]
        local offset = self.key_index
        for step = 1, self.sequence_length do
            table.insert(batch_keys[step], sampled_key)
            offset = offset + self.step_size
            sampled_key = self.video_keys[offset]
        end
        self.key_index = self.key_index + 1
    end
    return batch_keys
end

function SingleVideoSampler.static.get_video_keys(frames_lmdb, video_name)
    local video_keys = {}
    local db = lmdb.env { Path = frames_lmdb }
    db:open()
    local transaction = db:txn(true --[[readonly]])
    local offset = 1
    local key = string.format('%s-%d', video_name, offset)
    while transaction:get(key) ~= nil do
        table.insert(video_keys, key)
        offset = offset + 1
        key = string.format('%s-%d', video_name, offset)
    end
    return video_keys
end

---
-- Load model.
---
log.info('Loading model.')
local model = torch.load(args.model)
model:cuda()
if torch.isTypeOf(model, 'nn.DataParallelTable') then
    model = model:get(1)
end
if args.decorate_sequencer then
    if torch.isTypeOf(model, 'nn.Sequencer') then
        log.info('WARNING: --decorate_sequencer on model that is already ' ..
                 'nn.Sequencer!')
    end
    model = nn.Sequencer(model)
end

model:evaluate()
log.info('Loaded model.')

-- Ensure that dropout is properly turned off. Sometimes with modules such as
-- nn.MapTable and nn.PeriodicResidualTable, we have issues where dropout is not
-- properly turned off even when :evaluate() is called.
log.info('Testing that dropout is properly turned off.')
local test_input = torch.rand(
    args.sequence_length, 1, 3, CROP_SIZE, CROP_SIZE):cuda()
local output1 = model:forward(test_input):clone()
model:forget()
local output2 = model:forward(test_input):clone()
model:forget()
assert(torch.all(torch.eq(output1, output2)),
       'Model produced two different outputs for the same input!')
output1 = nil
output2 = nil
test_input = nil
log.info('Successfully tested dropout.')
collectgarbage(); collectgarbage()

log.info('Executing command: git --no-pager diff scripts/compute_video_activations.lua')
os.execute('git --no-pager diff scripts/compute_video_activations.lua | tee -a ' ..
           output_log)
log.info('Executing command: git --no-pager rev-parse HEAD')
os.execute('git --no-pager rev-parse HEAD | tee -a ' .. output_log)
---
-- Get requested layer.
---
log.info('Layer specification:', args.layer_spec)
-- Find specifications between ">" characters.
-- local search_module = model:findModules('nn.PeriodicResidualTable')[1].modules[1]:findModules('cudnn.SpatialConvolution')[1]
local search_module = model
for specification in string.gmatch(args.layer_spec, "[^>]+") do
    -- Split on "," character, returns an iterator.
    local spec_parser = string.gmatch(specification, "[^,]+")
    local layer_type = spec_parser()
    local layer_index = spec_parser()
    log.info(layer_index)
    layer_index = tonumber(layer_index)
    log.info(layer_type, layer_index)
    log.info(search_module:findModules(layer_type))
    search_module = search_module:findModules(layer_type)[layer_index]
end
local layer_to_extract = search_module
log.info('Extracting from layer:', layer_to_extract)

---
-- Pass frames through model
---
local source = data_source.LabeledVideoFramesLmdbSource(
    args.groundtruth_lmdb, args.groundtruth_without_images_lmdb, args.num_labels)
local sampler = SingleVideoSampler(
    source,
    args.sequence_length,
    args.step_size,
    true --[[use boundary frames]],
    {video_name = args.video_name})
local loader = data_loader.DataLoader(source, sampler)

local gpu_inputs = torch.CudaTensor()
local samples_complete = 0
local relus, relu_containers = model:findModules('cudnn.ReLU')
log.info('# of ReLUs:', #relus)
-- Disable in-place ReLUs so that we don't accidentally compute the ReLU'd
-- version of activations.
for i, relu in ipairs(relus) do
    for j, layer in ipairs(relu_containers[i].modules) do
        if layer == relu then
            relu_containers[i].modules[j] = cudnn.ReLU(false):cuda()
        end
    end
    -- For some reason, simply setting relu.inplace to false doesn't do
    -- anything. Maybe when a ReLU is loaded from disk, it's already hardcoded
    -- the "inplace" setting somewhere?
    -- relu.inplace = false
end
log.info('Disabled in-place ReLUs.')

local frame_activations = nil
while samples_complete + args.sequence_length - 1 ~= sampler:num_samples() do
    local to_load = args.batch_size
    if samples_complete + args.batch_size + args.sequence_length > sampler:num_samples() then
        to_load = sampler:num_samples() - samples_complete - args.sequence_length + 1
    end
    local images_table, _, batch_keys = loader:load_batch(
        to_load, true --[[return_keys]])
    -- TODO(achald): This assumes that the layer we extract maps to the last
    -- frame. That's not necessarily true!
    batch_keys = batch_keys[1]

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
    -- Should be of shape
    -- (batch_size, activation_size_1, activation_size_2, ...)
    local outputs = layer_to_extract.output
    if torch.any(torch.eq(outputs, 0)) then
        error('Output was exactly 0, check relu/dropout')
    end
    if frame_activations == nil then
        local activation_map_size = torch.totable(outputs[1]:size())
        frame_activations = torch.Tensor(
            sampler:num_samples() + args.sequence_length - 1,
            unpack(activation_map_size))
    end
    for i = 1, batch_size do
        local _, frame_number = source:frame_video_offset(batch_keys[i])
        frame_activations[frame_number] = outputs[i]:float()
    end
    samples_complete = samples_complete + to_load
    log.info(string.format('Computed activations for %d/%d.',
                        samples_complete, sampler:num_samples()))
    collectgarbage()
    collectgarbage()
end
log.info('Finished computing activations.')

---
-- Create activations tensor.
---
log.info('Saving activations to disk.')
torch.save(args.output_activations, frame_activations)
