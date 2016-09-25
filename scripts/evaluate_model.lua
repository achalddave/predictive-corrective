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
local string_util = require 'util/strings'
require 'CAvgTable'

local parser = argparse() {
    description = 'Evaluate a Torch model on MultiTHUMOS.'
}
parser:argument('model', 'Model file.')
parser:argument('labeled_video_frames_lmdb',
                'LMDB containing LabeledVideoFrames to evaluate.')
parser:argument('labeled_video_frames_without_images_lmdb',
                'LMDB containing LabeledVideoFrames without images.')
parser:option('--output_hdf5', 'HDF5 to output predictions to'):count('?')
parser:option('--sequence_length', 'Number of input frames.')
    :count('?'):default(2):convert(tonumber)
parser:option('--step_size', 'Size of step between frames.')
    :count('?'):default(1):convert(tonumber)
-- E.g. 'val1\nval2\n\nval3\n\nval4' denotes 3 groups.
parser:option('--val_groups',
              'Text file denoting groups of validation videos. ' ..
              'Groups are delimited using a blank line.')
      :count(1)
      :default('/data/achald/MultiTHUMOS/val_split/val_val_groups.txt')
parser:option('--batch_size', 'Batch size'):convert(tonumber):default(64)
parser:flag('--decorate_sequencer',
            'If specified, decorate model with nn.Sequencer.' ..
            'This is necessary if the model does not expect a table as ' ..
            'input.')
parser:flag('--c3d_input',
            'If specified, use C3D input format, which is of size ' ..
            '(batch_size, num_channels, sequence_length, width, height)')

local args = parser:parse()

-- More config variables.
local NUM_LABELS = 65
local GPUS = {1, 2, 3, 4}
local MEANS = {96.8293, 103.073, 101.662}
local CROP_SIZE = 224
local CROPS = {'c'}
local FRAME_TO_PREDICT = args.sequence_length

local num_images_in_batch = math.floor(args.batch_size / #CROPS)

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
    if torch.isTypeOf(single_model, 'nn.Sequencer') then
        print('WARNING: --decorate_sequencer on model that is already ' ..
              'nn.Sequencer!')
    end
    single_model = nn.Sequencer(single_model)
end
-- DataParallel across the 2nd dimension, which will be batch size. Our 1st
-- dimension is a step in the sequence.
local batch_dimension = args.c3d_input and 1 or 2
local model = nn.DataParallelTable(batch_dimension --[[dimension]])
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
    args.sequence_length,
    args.step_size,
    false --[[ use_boundary_frames ]])
local loader = data_loader.DataLoader(
    args.labeled_video_frames_lmdb, sampler, NUM_LABELS)
print('Initialized sampler.')

-- Pass each image in the database through the model, collect predictions and
-- groundtruth.
local gpu_inputs = torch.CudaTensor()
local all_predictions -- (num_samples, num_labels) tensor
local all_labels -- (num_samples, num_labels) tensor
local all_keys = {} -- (num_samples) length table
local samples_complete = 0

while true do
    if samples_complete == loader:num_samples() then
        break
    end
    local to_load = num_images_in_batch
    if samples_complete + num_images_in_batch > loader:num_samples() then
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
        local next_to_load = num_images_in_batch
        if next_samples_complete + next_to_load > loader:num_samples() then
            next_to_load = loader:num_samples() - next_samples_complete
        end
        loader:fetch_batch_async(next_to_load)
    end

    local batch_size = #images_table[1]
    local num_channels = images_table[1][1]:size(1)
    local images = torch.Tensor(args.sequence_length, batch_size * #CROPS,
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

    if args.c3d_input then
        -- Permute
        -- from (sequence_length,   batch_size,    num_channels, width, height)
        -- to   (     batch_size, num_channels, sequence_length, width, height)
        images = images:permute(2, 3, 1, 4, 5)
    end

    gpu_inputs:resize(images:size()):copy(images)

    -- (sequence_length, #images_table, num_labels) array
    local crop_predictions = model:forward(gpu_inputs):type(
        torch.getdefaulttensortype())
    -- We only care about the predictions for the last step of the sequence.
    if torch.isTensor(crop_predictions) and
        crop_predictions:dim() == 3 and
        crop_predictions:size(1) == args.sequence_length then
        -- If we are using nn.Sequencer
        crop_predictions = crop_predictions[args.sequence_length]
    elseif torch.isTensor(crop_predictions) and crop_predictions:size(1) == 1
        then
        -- If we are using, e.g. pyramid model.
        crop_predictions = crop_predictions[1]
    end

    if not (crop_predictions:dim() == 2
            and crop_predictions:size(1) == batch_size
            and crop_predictions:size(2) == NUM_LABELS) then
        error('Unknown output predictions shape.')
    end
    labels = labels[FRAME_TO_PREDICT]
    batch_keys = batch_keys[args.sequence_length]

    local predictions = torch.Tensor(batch_size, NUM_LABELS)
    for i = 1, batch_size do
        local start_crops = (i - 1) * #CROPS + 1
        local end_crops = i * #CROPS
        predictions[i] = crop_predictions[{{start_crops, end_crops},
                                           {}}]:mean(1)
    end
    -- Concat batch_keys to all_keys.
    for i = 1, batch_size do table.insert(all_keys, batch_keys[i]) end

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

do -- Compute accuracy across validation groups.
    local groups_file = torch.DiskFile(args.val_groups, 'r'):quiet()
    local file_groups = {{}}
    while true do
        local line = string_util.trim(groups_file:readString('*l'))
        if groups_file:hasError() then break end
        if line == '' then
            table.insert(file_groups, {})
        else
            table.insert(file_groups[#file_groups], line)
        end
    end

    local group_predictions = {}
    local group_labels = {}
    for key_index, key in ipairs(all_keys) do
        local key_group = nil
        for group, group_keys in ipairs(file_groups) do
            for _, group_key in ipairs(group_keys) do
                if string_util.starts(key, group_key) then
                    key_group = group
                    break
                end
            end
            if key_group ~= nil then break end
        end
        if group_predictions[key_group] == nil then
            group_predictions[key_group] = all_predictions[{{key_index}}]
            group_labels[key_group] = all_labels[{{key_index}}]
        else
            group_predictions[key_group] = torch.cat(
                group_predictions[key_group], all_predictions[{{key_index}}], 1)
            group_labels[key_group] = torch.cat(
                group_labels[key_group], all_labels[{{key_index}}], 1)
        end
    end

    local mAPs = torch.zeros(#file_groups)
    for group_index = 1, #file_groups do
        if group_predictions[group_index] ~= nil then
            mAPs[group_index] = evaluator.compute_mean_average_precision(
                group_predictions[group_index],
                group_labels[group_index])
        end
        print(string.format('Group %d mAP:', group_index), mAPs[group_index])
    end
    print('Group mAPs STD:', mAPs:std())
end

if args.output_hdf5 ~= nil then
    local predictions_by_filename = {}
    for i, key in pairs(all_keys) do
        local prediction = all_predictions[i]
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
    for filename, predictions_table in pairs(predictions_by_filename) do
        -- We don't have predictions for the first `FRAME_TO_PREDICT-1` frames.
        -- So, set them to be -1.
        for i = 1, FRAME_TO_PREDICT-1 do
            predictions_by_filename[filename][i] = torch.zeros(NUM_LABELS) - 1
        end
        predictions_by_filename[filename] = torch.cat(predictions_table, 2):t()
    end

    -- Save predictions to HDF5.
    local output_file = hdf5.open(args.output_hdf5, 'w')
    for filename, file_predictions in pairs(predictions_by_filename) do
        output_file:write(filename, file_predictions)
    end
end
