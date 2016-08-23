--[[ Evaluate a trained torch model on an LMDB.
--
-- Note that this should be run from the root directory, not from within
-- scripts/.
--]]

package.path = package.path .. ";../?.lua"
package.path = package.path .. ";layers/?.lua"

local argparse = require 'argparse'
local cutorch = require 'cutorch'
local hdf5 = require 'hdf5'
local image = require 'image'
local nn = require 'nn'
local torch = require 'torch'
require 'rnn'

local data_loader = require 'data_loader'
local evaluator = require 'evaluator'
require 'CAvgTable'

local parser = argparse() {
    description = 'Evaluate a Torch model on MultiTHUMOS.'
}
parser:argument('model', 'Model file.')
parser:argument('labeled_video_frames_lmdb',
                'LMDB containing LabeledVideoFrames to evaluate.')
parser:argument('labeled_video_frames_without_images_lmdb',
                'LMDB containing LabeledVideoFrames without images.')
parser:argument('output_hdf5', 'HDF5 to output predictions to')
parser:flag('--decorate_sequencer',
            'If specified, decorate model with nn.Sequencer.' ..
            'This is necessary if the model does not expect a table as ' ..
            'input.')

local args = parser:parse()

-- More config variables.
local NUM_LABELS = 65
local NETWORK_BATCH_SIZE = 384
local GPUS = {1, 2, 3, 4}
local MEANS = {96.8293, 103.073, 101.662}
local CROP_SIZE = 224
local CROPS = {'c'}
local SEQUENCE_LENGTH = 2
local FRAME_TO_PREDICT = 1
local STEP_SIZE = 1
local IMAGES_IN_BATCH = math.floor(NETWORK_BATCH_SIZE / #CROPS)

math.randomseed(0)
torch.manualSeed(0)
cutorch.manualSeedAll(0)
cutorch.setDevice(GPUS[1])
torch.setdefaulttensortype('torch.FloatTensor')

print('Git status information:')
print('===')
os.execute('git --no-pager diff scripts/evaluate_model.lua')
os.execute('git --no-pager rev-parse HEAD')
print('===')
print('Evaluating model. Args:')
print(args)
print('STEP_SIZE: ', STEP_SIZE)
print('SEQUENCE_LENGTH: ', SEQUENCE_LENGTH)
print('FRAME_TO_PREDICT: ', FRAME_TO_PREDICT)
print('CROPS: ', CROPS)

-- Load model.
print('Loading model.')
nn.DataParallelTable.deserializeNGPUs = #GPUS
local single_model = torch.load(args.model)
if torch.isTypeOf(single_model, 'nn.DataParallelTable') then
    single_model = single_model:get(1)
end
if args.decorate_sequencer then
    single_model = nn.Sequencer(single_model)
end
-- DataParallel across the 2nd dimension, which will be batch size. Our 1st
-- dimension is a step in the sequence.
local model = nn.DataParallelTable(2 --[[dimension]])
for _, gpu in ipairs(GPUS) do
    cutorch.setDevice(gpu)
    model:add(single_model:clone():cuda(), gpu)
end
cutorch.setDevice(GPUS[1])
model:evaluate()
print('Loaded model.')

-- Open database.
local sampler = data_loader.PermutedSampler(
    args.labeled_video_frames_without_images_lmdb,
    NUM_LABELS,
    SEQUENCE_LENGTH,
    STEP_SIZE)
local loader = data_loader.DataLoader(
    args.labeled_video_frames_lmdb, sampler, NUM_LABELS)
print('Initialized sampler.')

-- Pass each image in the database through the model, collect predictions and
-- groundtruth.
local gpu_inputs = torch.CudaTensor()
local all_predictions
local all_labels
local predictions_by_keys = {}
local samples_complete = 0

while true do
    if samples_complete == loader:num_samples() then
        break
    end
    local to_load = IMAGES_IN_BATCH
    if samples_complete + IMAGES_IN_BATCH > loader:num_samples() then
        to_load = loader:num_samples() - samples_complete
    end
    local images_table, labels, batch_keys = loader:load_batch(
        to_load, true --[[return_keys]])
    if loader.sampler.key_index ~= samples_complete + to_load + 1 then
        print('Data loader key index does not match samples_complete')
        print(loader.sampler.key_index, samples_complete + to_load + 1)
        os.exit(1)
    end
    -- Prefetch the next batch.
    if samples_complete + to_load < loader:num_samples() then
        -- Figure out how many images we will need in the next batch, and
        -- prefetch them.
        local next_samples_complete = samples_complete + to_load
        local next_to_load = IMAGES_IN_BATCH
        if next_samples_complete + next_to_load > loader:num_samples() then
            next_to_load = loader:num_samples() - next_samples_complete
        end
        loader:fetch_batch_async(next_to_load)
    end

    local batch_size = #images_table[1]
    local num_channels = images_table[1][1]:size(1)
    local images = torch.Tensor(SEQUENCE_LENGTH, batch_size * #CROPS,
                                num_channels, CROP_SIZE, CROP_SIZE)
    for step, step_images in ipairs(images_table) do
        for batch_index, img in ipairs(step_images) do
            -- Process image after converting to the default Tensor type.
            -- (Originally, it is a ByteTensor).
            img = img:typeAs(images)
            for channel = 1, 3 do
                img[{{channel}, {}, {}}]:add(-MEANS[channel])
            end
            for i, crop in ipairs(CROPS) do
                images[{step, batch_index + i - 1}] = image.crop(
                    img, crop, CROP_SIZE, CROP_SIZE)
            end
        end
    end

    gpu_inputs:resize(images:size()):copy(images)

    -- (SEQUENCE_LENGTH, #images_table, num_labels) array
    local crop_predictions = model:forward(gpu_inputs):type(
        torch.getdefaulttensortype())
    -- We only care about the predictions for the last step of the sequence.
    if torch.isTensor(crop_predictions) and
        crop_predictions:size(1) == SEQUENCE_LENGTH then
        -- If we are using nn.Sequencer
        crop_predictions = crop_predictions[SEQUENCE_LENGTH]
    elseif torch.isTensor(crop_predictions) and crop_predictions:size(1) == 1
        then
        -- If we are using, e.g. pyramid model.
        crop_predictions = crop_predictions[1]
    else
        error('Unknown output predictions shape.')
    end
    labels = labels[FRAME_TO_PREDICT]
    batch_keys = batch_keys[SEQUENCE_LENGTH]

    local predictions = torch.Tensor(batch_size, NUM_LABELS)
    for i = 1, batch_size do
        local start_crops = (i - 1) * #CROPS + 1
        local end_crops = i * #CROPS
        predictions[i] = torch.sum(
            crop_predictions[{{start_crops, end_crops}, {}}], 1) / #CROPS
    end
    for i = 1, batch_size do
        predictions_by_keys[batch_keys[i]] = predictions[i]
    end

    if all_predictions == nil then
        all_predictions = predictions
        all_labels = labels
    else
        all_predictions = torch.cat(all_predictions, predictions, 1)
        all_labels = torch.cat(all_labels, labels, 1)
    end

    local num_labels_with_groundtruth = 0
    for i = 1, NUM_LABELS do
        if torch.any(all_labels[{{}, {i}}]) then
            num_labels_with_groundtruth = num_labels_with_groundtruth + 1
        end
    end

    samples_complete = samples_complete + to_load
    local map_so_far = evaluator.compute_mean_average_precision(
        all_predictions, all_labels)
    local batch_map = evaluator.compute_mean_average_precision(
        predictions, labels)
    print(string.format(
        '%s: Finished %d/%d. mAP so far: %.5f, batch mAP: %.5f',
        os.date('%X'), samples_complete, loader:num_samples(), map_so_far,
        batch_map))
    collectgarbage()
end

-- Compute AP for each class.
local aps = torch.zeros(all_predictions:size(2))
for i = 1, all_predictions:size(2) do
    local ap = evaluator.compute_mean_average_precision(
        all_predictions[{{}, {i}}], all_labels[{{}, {i}}])
    aps[i] = ap
    print(string.format('Class %d\t AP: %.5f', i, ap))
end

local map = torch.mean(aps[torch.ne(aps, -1)])
print('mAP: ', map)

if SEQUENCE_LENGTH == 1 then
    -- Save predictions to HDF5.
    local output_file = hdf5.open(args.output_hdf5, 'w')
    -- Map filename to a table of predictions by frame number.
    local predictions_by_filename = {}
    for key, prediction in pairs(predictions_by_keys) do
        -- Keys are of the form '<filename>-<frame_number>'.
        -- Find the index of the '-'
        local _, split_index = string.find(key, '.*-')
        local filename = string.sub(key, 1, split_index - 1)
        local frame_number = tonumber(string.sub(key, split_index + 1, -1))
        if predictions_by_filename[filename] == nil then
            predictions_by_filename[filename] = {}
        end
        predictions_by_filename[filename][frame_number] = prediction
    end
    -- TODO(achald): This is currently broken when SEQUENCE_LENGTH > 1.
    -- The torch.cat call will fail because the first frame we have predictions for
    -- will be frame SEQUENCE_LENGTH, and so the predictions_table will not have a
    -- key for indices 1..SEQUENCE_LENGTH-1. Unclear how to fix...
    for filename, predictions_table in pairs(predictions_by_filename) do
        output_file:write(filename, torch.cat(predictions_table, 2):t())
    end
else
    print('Cannot save predictions to HDF5 for SEQUENCE_LENGTH ~= 1')
end
