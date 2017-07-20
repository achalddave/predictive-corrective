--[[ Evaluate a trained torch model on an LMDB.
--
-- Note that this should be run from the root directory, not from within
-- scripts/.
--
-- Example usage:
--  th scripts/evaluate_model.lua \
--      /data/achald/MultiTHUMOS/models/balanced_without_bg_sampling_vgg_new/from_scratch/model_30.t7 \
--      /data/achald/MultiTHUMOS/frames@10fps/labeled_video_frames/valval.lmdb/ \
--      /data/achald/MultiTHUMOS/frames@10fps/labeled_video_frames/valval_without_images.lmdb \
--      /data/achald/MultiTHUMOS/models/balanced_without_bg_sampling_vgg_new/from_scratch/model_30_valval_evaluation.log \
--      --sequence_length 1 \
--      --step_size 1 \
--      --batch_size 128 \
--      --val_groups /data/achald/MultiTHUMOS/val_split/val_val_groups.txt \
--      --output_hdf5 /data/achald/MultiTHUMOS/models/balanced_without_bg_sampling_vgg_new/from_scratch/model_30_valval_predictions.h5
--]]

package.path = package.path .. ";../?.lua"
package.path = package.path .. ";layers/?.lua"

local argparse = require 'argparse'
local cutorch = require 'cutorch'
require 'cunn'
local hdf5 = require 'hdf5'
local image = require 'image'
local __ = require 'moses'
local nn = require 'nn'
local torch = require 'torch'
require 'rnn'

local data_source = require 'data_source'
local data_loader = require 'data_loader'
local evaluator = require 'evaluator'
local experiment_saver = require 'util/experiment_saver'
local image_util = require 'util/image_util'
local log = require 'util/log'
local samplers = require 'samplers'
local string_util = require 'util/strings'
require 'layers/init'

local parser = argparse() {
    description = 'Evaluate a Torch model on MultiTHUMOS.'
}
parser:argument('model', 'Model file.')
parser:argument('frames_root',
                'Root directory containing video frames.')
parser:argument('labels_hdf5',
                'HDF5 file containing labels for videos.')
parser:argument('output_log', 'File to log output to')
parser:option('--output_hdf5', 'HDF5 to output predictions to'):count('?')
parser:option('--num_labels', 'Number of labels')
    :count('?'):default(65):convert(tonumber)
parser:option('--sequence_length', 'Number of input frames.')
    :count('?'):default(1):convert(tonumber)
parser:option('--step_size', 'Size of step between frames.')
    :count('?'):default(1):convert(tonumber)
parser:option('--experiment_id_file',
              'Path to text file containing the experiment id for this run.' ..
              'The id in this file will be incremented by this program.')
              :count(1)
              :default('/data/achald/MultiTHUMOS/models/next_experiment_id.txt')
-- E.g. 'val1\nval2\n\nval3\n\nval4' denotes 3 groups.
parser:option('--val_groups',
              'Text file denoting groups of validation videos. ' ..
              'Groups are delimited using a blank line.')
      :count(1)
      :default('/data/achald/MultiTHUMOS/val_split/val_val_groups.txt')
parser:option('--batch_size', 'Batch size'):convert(tonumber):default(64)
parser:option('--reinit_rate', 'Reinit rate'):convert(tonumber)
parser:option('--min_reinit_rate',
              'Min reinit rate (for PredictiveCorrectiveBlock)')
      :convert(tonumber)
parser:option('--reinit_threshold', 'Threshold over which to reinitialize')
      :convert(tonumber)
parser:option('--ignore_threshold', 'Threshold under which to ignore frames')
      :convert(tonumber)
parser:option('--charades_submission_out',
              'Path to output Charades submission file to. If specified, ' ..
              'evaluate only on 25 frames uniformly spaced over ' ..
              'the video, as used for Charades evaluation.')
parser:flag('--decorate_sequencer',
            'If specified, decorate model with nn.Sequencer.' ..
            'This is necessary if the model does not expect a table as ' ..
            'input.')
parser:flag('--c3d_input',
            'If specified, use C3D input format, which is of size ' ..
            '(batch_size, num_channels, sequence_length, width, height)')
parser:flag('--recurrent',
            'If specified, evaluate "recurrently"; i.e. with a sliding ' ..
            'window with a stride = sequence_length instead of stride = 1, ' ..
            'which is the default. Assumes that the model outputs ' ..
            'sequence_length predictions for sequence_length inputs.')

local args = parser:parse()

if args.charades_submission_out ~= nil and args.recurrent ~= nil then
    log.info('Disabling recurrent evaluation as ' ..
             '--charades_submission_out specified.')
    args.recurrent = nil
end

-- More config variables.
local GPUS = {1, 2, 3, 4}
-- TODO(achald): Allow specifying dataset from cmd line; use this to determine
-- pixel mean, num labels, etc.
-- local PIXEL_MEAN = {103.939, 116.779, 123.68} -- Gunnar's Charades model mean
-- local PIXEL_MEAN = {86.18377069, 93.74071911, 105.4525389} -- Charades
-- local PIXEL_MEAN = {92.4318769, 99.46975121, 100.62499024} -- train+val MultiTHUMOS
local PIXEL_MEAN = {96.8293, 103.073, 101.662} -- train MultiTHUMOS
local CROP_SIZE = 224
local CROPS = {'c'}

local max_batch_size_images = math.floor(args.batch_size / #CROPS)

math.randomseed(0)
torch.manualSeed(0)
cutorch.manualSeedAll(0)
cutorch.setDevice(GPUS[1])
torch.setdefaulttensortype('torch.FloatTensor')
torch.setheaptracking(true)

log.outfile = args.output_log
log.info('Git status information:')
log.info('===')

log.info('Executing command: git --no-pager diff scripts/evaluate_model.lua')
os.execute('git --no-pager diff scripts/evaluate_model.lua | tee -a ' ..
           args.output_log)

log.info('Executing command: git --no-pager diff layers/')
os.execute('git --no-pager diff layers/ | tee -a ' .. args.output_log)

log.info('Executing command: git --no-pager rev-parse HEAD')
os.execute('git --no-pager rev-parse HEAD | tee -a ' .. args.output_log)

log.info('===')

log.info('Evaluating model. Args:')
log.info(args)
log.info('CROPS: ', CROPS)

local experiment_id = experiment_saver.read_and_increment_experiment_id(
    args.experiment_id_file)
log.info('===')
log.info('Experiment id:', experiment_id)
log.info('===')

-- Load model.
log.info('Loading model.')
nn.DataParallelTable.deserializeNGPUs = #GPUS
local single_model = torch.load(args.model)
if torch.isTypeOf(single_model, 'nn.DataParallelTable') then
    single_model = single_model:get(1)
end

if args.decorate_sequencer then
    log.info('Decorating sequencer')
    if torch.isTypeOf(single_model, 'nn.Sequencer') then
        log.warn('--decorate_sequencer on model that is already ' ..
                 'nn.Sequencer!')
    end
    single_model = nn.Sequencer(single_model)
end

if args.reinit_rate ~= nil then
    log.info('Resetting reinit rate to', args.reinit_rate)
    local reinit_types = {
        'nn.CRollingDiffTable', 'nn.PeriodicResidualTable', 'nn.CCumSumTable'}
    for _, reinit_type in ipairs(reinit_types) do
        local reinit_layers, _ = single_model:findModules(reinit_type)
        for _, reinit_layer in ipairs(reinit_layers) do
            reinit_layer:set_reinitialize_rate(args.reinit_rate)
        end
    end
end

local pcbs, _ = single_model:findModules('nn.PredictiveCorrectiveBlock')
if args.min_reinit_rate ~= nil then
    for _, pcb in ipairs(pcbs) do
        pcb.max_update = args.min_reinit_rate
    end
end
if args.ignore_threshold ~= nil then
    for _, pcb in ipairs(pcbs) do
        pcb.ignore_threshold = args.ignore_threshold
    end
end
if args.reinit_threshold ~= nil then
    for _, pcb in ipairs(pcbs) do
        pcb.init_threshold = args.reinit_threshold
    end
end

-- DataParallel across the 2nd dimension, which will be batch size. Our 1st
-- dimension is a step in the sequence.
local batch_dimension = args.c3d_input and 1 or 2
local model = nn.DataParallelTable(batch_dimension --[[dimension]])
model:add(single_model, GPUS)
model:cuda()
single_model = nil
collectgarbage()
collectgarbage()

cutorch.setDevice(GPUS[1])
local layers, _ = model:findModules('nn.MapTable')
for _, layer in ipairs(layers) do
    if layer.modules[1] ~= layer.module then
        -- This shouldn't happen! It happened for a few models trained using
        -- commits between c6b93fe and 12f5b4b. See the commit message for
        -- 12f5b4b for details.
        log.warn('MapTable module not in modules! Adding it.')
        layer.module = layer.modules[1]
    end
end
collectgarbage()
collectgarbage()
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

local function closest_multiple(target, divisor)
    return torch.round(target / divisor) * divisor
end

local function crop_and_zero_center_images(
    images_table, crops, crop_size, image_mean)
    --[[
    Args:
        images (Array of array of ByteTensors): Contains image sequences for
            the batch. Each element is a step in the sequence, so that images is
            an array of length sequence_length, whose elements are arrays of
            length batch_size.
        crops (Array of crop formats): E.g. {'c', 'tl', 'tr', 'bl' or 'br'}
    ]]--
    local sequence_length = #images_table
    local num_crops = #crops * #images_table[1]
    local num_channels = 3
    local images = torch.Tensor(
        sequence_length, num_crops, num_channels, crop_size, crop_size)
    for step, step_images in ipairs(images_table) do
        for batch_index, img in ipairs(step_images) do
            -- Process image after converting to the default Tensor type.
            -- (Originally, it is a ByteTensor).
            if img == samplers.END_OF_SEQUENCE then
                for i, _ in ipairs(crops) do
                    local crop_index = ((batch_index - 1) * #crops) + i
                    images[{step, crop_index}] = torch.zeros(
                        num_channels, crop_size, crop_size)
                end
            else
                img = img:typeAs(images)
                img = image_util.subtract_pixel_mean(img, image_mean)

                for i, crop in ipairs(crops) do
                    local crop_index = ((batch_index - 1) * #crops) + i
                    images[{step, crop_index}] = image.crop(
                        img, crop, crop_size, crop_size)
                end
            end
        end
    end
    return images
end

local function average_crop_predictions(crop_predictions, num_crops)
    --[[
        Args:
            crop_predictions (sequence_length, num_images*num_crops, num_labels)
            num_crops (int)

        Returns:
            predictions (sequence_length, num_images, num_labels)
    ]]--
    assert(crop_predictions:size(2) % num_crops == 0)
    local num_images = crop_predictions:size(2) / num_crops
    local predictions = torch.Tensor(
        crop_predictions:size(1), num_images, crop_predictions:size(3))
    for i = 1, num_images do
        local start_crops = (i - 1) * num_crops + 1
        local end_crops = i * num_crops
        predictions[{{}, i}] = crop_predictions[
            {{}, {start_crops, end_crops}}]:mean(2)
    end
    return predictions
end

local function evaluate_model_sequential(options)
    --[[
        Args:
            model
            source
            sequence_length
            step_size
            max_batch_size_images
            num_labels
            crops
            crop_size
            pixel_mean
    ]]--
    local model = options.model
    local source = options.source
    local sequence_length = options.sequence_length
    local step_size = options.step_size
    local batch_size_sequences = options.max_batch_size_images
    local num_labels = options.num_labels
    local crops = options.crops
    local crop_size = options.crop_size
    local pixel_mean = options.pixel_mean

    local sampler_options = {
        batch_size = batch_size_sequences,
        sample_once = true
    }
    local sampler = samplers.SequentialSampler(
        source, sequence_length, step_size,
        nil --[[ use boundary frames]], sampler_options)
    local loader = data_loader.DataLoader(source, sampler)
    log.info('Initialized sampler.')

    -- Pass each image in the database through the model, collect predictions
    -- and groundtruth.
    local gpu_inputs = torch.CudaTensor()
    local all_predictions = nil -- (num_samples, num_labels) tensor
    local all_labels = nil -- (num_samples, num_labels) tensor
    local all_keys = {} -- (num_samples) length table

    local num_iter = 0
    while true do
        local images_table, labels, batch_keys = loader:load_batch(
            batch_size_sequences, true --[[return_keys]])
        loader:fetch_batch_async(batch_size_sequences)
        local images = crop_and_zero_center_images(
            images_table, crops, crop_size, pixel_mean)
        gpu_inputs:resize(images:size()):copy(images)

        -- (sequence_length, batch_size_crops, num_labels) array
        local crop_predictions = model:forward(gpu_inputs):type(
            torch.getdefaulttensortype())
        -- (sequence_length, batch_size_sequences, num_labels) array
        local predictions = average_crop_predictions(crop_predictions, #crops)

        -- Concat batch_keys to all_keys.
        local valid_up_to_step = {}
        for sequence = 1, batch_size_sequences do
            valid_up_to_step[sequence] = 0
            for step = 1, sequence_length do
                if batch_keys[step][sequence] ~= samplers.END_OF_SEQUENCE
                    then
                    valid_up_to_step[sequence] = step
                    table.insert(all_keys, batch_keys[step][sequence])
                else
                    break
                end
            end
        end

        -- We can't put this all into the same loop because repeated torch.cat
        -- calls cause excessive memory usage that is not cleared unless
        -- collectgarbage() is called twice in every step of the inner loop
        -- above; doing this slows down the code drastically.
        local batch_predictions, batch_groundtruth
        for sequence = 1, batch_size_sequences do
            if valid_up_to_step[sequence] >= 1 then
                if batch_predictions == nil then
                    batch_predictions = predictions[{
                        {1, valid_up_to_step[sequence]}, sequence}]
                    batch_labels = labels[{
                        {1, valid_up_to_step[sequence]}, sequence}]
                else
                    batch_predictions = torch.cat(
                        batch_predictions, predictions[{
                            {1, valid_up_to_step[sequence]}, sequence}], 1)
                    batch_labels = torch.cat(batch_labels, labels[{
                        {1, valid_up_to_step[sequence]}, sequence}], 1)
                end
            end
        end

        if all_predictions == nil then
            all_predictions = batch_predictions
            all_labels = batch_labels
        else
            if batch_predictions ~= nil then
                all_predictions = torch.cat(
                    all_predictions, batch_predictions, 1)
                all_labels = torch.cat(all_labels, batch_labels, 1)
            end
        end

        if num_iter % 10 == 0 then
            local log_string = string.format(
                'Finished %d/%d.', #all_keys, loader:num_samples())
            if num_iter % 100 == 0 then
                local map_so_far = evaluator.compute_mean_average_precision(
                    all_predictions, all_labels)
                local thumos_map_so_far =
                    evaluator.compute_mean_average_precision(
                        all_predictions[{{}, {1, 20}}],
                        all_labels[{{}, {1, 20}}])
                log_string = string.format('%s mAP: %.5f, THUMOS mAP: %.5f',
                    log_string, map_so_far, thumos_map_so_far)
            end
            log.info(log_string)
        end

        images_table = nil
        images = nil
        predictions = nil
        batch_keys = nil
        collectgarbage()
        collectgarbage()
        if #all_keys == sampler:num_samples() then
            break
        end
        num_iter = num_iter + 1
    end

    return all_predictions, all_labels, all_keys
end

local function evaluate_model(options)
    --[[
        Args:
            model
            source
            sequence_length
            step_size
            max_batch_size_images
            num_labels
            crops
            crop_size
            pixel_mean
            c3d_input
    ]]--
    local model = options.model
    local source = options.source
    local sequence_length = options.sequence_length
    local step_size = options.step_size
    local max_batch_size_images = options.max_batch_size_images
    local num_labels = options.num_labels
    local crops = options.crops
    local crop_size = options.crop_size
    local pixel_mean = options.pixel_mean
    local c3d_input = options.c3d_input
    local uniformly_spaced_eval = options.uniformly_spaced_eval or false
    local progress_every = closest_multiple(
        math.max(100, max_batch_size_images), max_batch_size_images)
    local average_precision_every = closest_multiple(10000, progress_every)

    if torch.isTypeOf(model, 'nn.DataParallelTable') then
        -- https://github.com/Element-Research/rnn/issues/404
        model.impl:exec(function(m) m:remember('neither') end)
    else
        model:forget('neither')
    end

    -- Open database.
    local sampler

    if not uniformly_spaced_eval then
        sampler = samplers.PermutedSampler(
            source, sequence_length, step_size, true --[[use_boundary_frames]])
    else
        log.info('Using UniformlySpacedSampler!')
        sampler = samplers.UniformlySpacedSampler(
            source, sequence_length, nil, nil, {num_frames_per_video=25})
    end
    local loader = data_loader.DataLoader(source, sampler)
    log.info('Initialized sampler.')

    -- Pass each image in the database through the model, collect predictions
    -- and groundtruth.
    local gpu_inputs = torch.CudaTensor()
    local all_predictions -- (num_samples, num_labels) tensor
    local all_labels -- (num_samples, num_labels) tensor
    local all_keys = {} -- (num_samples) length table
    local samples_complete = 0

    while true do
        if samples_complete == loader:num_samples() then
            break
        end
        local to_load = max_batch_size_images
        if samples_complete + max_batch_size_images > loader:num_samples() then
            to_load = loader:num_samples() - samples_complete
        end
        local images_table, labels, batch_keys = loader:load_batch(
            to_load, true --[[return_keys]])
        if loader.sampler.key_index ~= samples_complete + to_load + 1 then
            log.info('Data loader key index does not match samples_complete')
            log.info(loader.sampler.key_index, samples_complete + to_load + 1)
            os.exit(1)
        end
        -- Prefetch the next batch.
        if samples_complete + to_load < loader:num_samples() then
            -- Figure out how many images we will need in the next batch, and
            -- prefetch them.
            local next_samples_complete = samples_complete + to_load
            local next_to_load = max_batch_size_images
            if next_samples_complete + next_to_load > loader:num_samples() then
                next_to_load = loader:num_samples() - next_samples_complete
            end
            loader:fetch_batch_async(next_to_load)
        end

        -- This may not be equal to max_batch_size_images in the last batch!
        local batch_size_images = to_load

        local images = crop_and_zero_center_images(
            images_table, crops, crop_size, pixel_mean)

        if c3d_input then
            -- Permute
            -- from (sequence_length, batch_size, num_channels, width, height)
            -- to   (batch_size, num_channels, sequence_length, width, height)
            images = images:permute(2, 3, 1, 4, 5)
        end

        gpu_inputs:resize(images:size()):copy(images)

        -- (sequence_length, #images_table, num_labels) array
        local crop_predictions = model:forward(gpu_inputs):type(
            torch.getdefaulttensortype())
        local predictions = average_crop_predictions(crop_predictions, #crops)

        -- We only care about the predictions for the last step of the sequence.
        if torch.isTensor(predictions) and
            predictions:dim() == 3 and
            predictions:size(1) == sequence_length then
            -- If there is an output for each step of the sequence; e.g. if we
            -- are using a recurrent model.
            predictions = predictions[sequence_length]
        elseif torch.isTensor(predictions) and predictions:size(1) == 1
            then
            -- If there is one output for the sequence; e.g. for the old
            -- hierarchical model.
            predictions = predictions[1]
        end
        assert(predictions:dim() == 2
               and predictions:size(1) == batch_size_images
               and predictions:size(2) == num_labels,
               string.format('Unknown output predictions shape: %s',
                             predictions:size()))

        if all_predictions == nil then
            all_predictions = predictions
            all_labels = labels[sequence_length]
        else
            all_predictions = torch.cat(all_predictions, predictions, 1)
            all_labels = torch.cat(all_labels, labels[sequence_length], 1)
        end

        -- Concat batch_keys to all_keys.
        all_keys = __.append(all_keys, batch_keys[sequence_length])

        -- The first (sequence_length - 1) frames of a video do not get
        -- predictions from the network. If these frames are in our batch,
        -- collect their labels and set their predictions to a large negative
        -- value.
        if sequence_length > 1 and not uniformly_spaced_eval then
            for batch_index, key in ipairs(batch_keys[1]) do
                local filename, frame_number = source:frame_video_offset(key)
                if frame_number == 1 then
                    local batch_labels = labels[
                        {{1, sequence_length-1}, batch_index}]
                    local batch_predictions = torch.zeros(
                        sequence_length-1, num_labels) - 1e9
                    all_labels = torch.cat(all_labels, batch_labels, 1)
                    all_predictions = torch.cat(
                        all_predictions, batch_predictions, 1)
                    local video_keys = source:video_keys()
                    for i = 1, sequence_length - 1 do
                        table.insert(all_keys, video_keys[filename][i])
                    end
                end
            end
        end

        samples_complete = samples_complete + to_load
        if samples_complete % progress_every == 0 then
            local log_string = string.format(
                'Finished %d/%d', samples_complete, loader:num_samples())
            if samples_complete % average_precision_every == 0 then
                local map_so_far = evaluator.compute_mean_average_precision(
                    all_predictions, all_labels)
                local thumos_map_so_far =
                    evaluator.compute_mean_average_precision(
                        all_predictions[{{}, {1, 20}}],
                        all_labels[{{}, {1, 20}}])
                log_string = string.format('%s mAP: %.5f, THUMOS mAP: %.5f',
                    log_string, map_so_far, thumos_map_so_far)
            end
            log.info(log_string)
        end
    end
    collectgarbage()
    collectgarbage()
    return all_predictions, all_labels, all_keys
end

local source = data_source.DiskFramesHdf5LabelsDataSource(
    args.frames_root,
    args.labels_hdf5)
local eval_options = {
    model = model,
    source = source,
    sequence_length = args.sequence_length,
    step_size = args.step_size,
    max_batch_size_images = max_batch_size_images,
    num_labels = args.num_labels,
    crops = CROPS,
    crop_size = CROP_SIZE,
    pixel_mean = PIXEL_MEAN,
    uniformly_spaced_eval = args.charades_submission_out ~= nil
}
local all_predictions = nil
local all_labels = nil
local all_keys = nil
if args.recurrent then
    log.info('NOTE: --recurrent specified; evaluating recurrently.')
    all_predictions, all_labels, all_keys = evaluate_model_sequential(
        eval_options)
else
    eval_options.c3d_input = args.c3d_input
    all_predictions, all_labels, all_keys = evaluate_model(
        eval_options)
end

-- Compute AP for each class.
local aps = torch.zeros(all_predictions:size(2))
for i = 1, all_predictions:size(2) do
    local ap = evaluator.compute_mean_average_precision(
        all_predictions[{{}, {i}}], all_labels[{{}, {i}}])
    aps[i] = ap
    log.info(string.format('Class %d\t AP: %.5f', i, ap))
end

assert(torch.all(torch.ne(aps, -1)))

log.info('mAP:', torch.mean(aps))

local group_mAPs
if args.val_groups ~= '' then -- Compute accuracy across validation groups.
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
    local groups_valid = true
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
        if key_group == nil then
            log.warn('Key missing from val group, not evaluating groups')
            groups_valid = false
            break
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

    if groups_valid then
        group_mAPs = torch.zeros(#file_groups)
        for group_index = 1, #file_groups do
            if group_predictions[group_index] ~= nil then
                group_mAPs[group_index] =
                    evaluator.compute_mean_average_precision(
                        group_predictions[group_index],
                        group_labels[group_index])
            end
            log.info(string.format(
                'Group %d mAP:', group_index), group_mAPs[group_index])
        end
        log.info(string.format(
            'Group %d mAP:', group_index), group_mAPs[group_index])
    end
    log.info('Group mAPs STD:', group_mAPs:std())
end

if args.charades_submission_out ~= nil then
    -- XXX HACK XXX
    torch.save('all_keys.t7', all_keys)
    torch.save('all_predictions.t7', all_predictions)
    local function tensor2str(x)
        str = ""
        for i=1,x:size(1) do
            if i == x:size(1) then
                str = str .. x[i]
            else
                str = str .. x[i] .. " "
            end
        end
        return str
    end
    local output = assert(io.open(args.charades_submission_out, 'w'))
    local current_filename, current_index
    for i, key in ipairs(all_keys) do
        local filename, frame_number = source:frame_video_offset(key)
        if current_filename == nil or current_filename ~= filename then
            current_index = 1
            current_filename = filename
        end
        if (i - 1) % 1000 == 0 then
            log.info(string.format(
                '%s %s %s\n', filename, current_index,
                tensor2str(all_predictions[{i, {}}])))
        end
        output:write(string.format(
            '%s %s %s\n', filename, current_index,
            tensor2str(all_predictions[{i, {}}])))
        current_index = current_index + 1
    end
    output:close()
elseif args.output_hdf5 ~= nil then
    log.info('Saving to output hdf5')
    local predictions_by_filename = {}
    for i, key in pairs(all_keys) do
        local prediction = all_predictions[i]
        -- Keys are of the form '<filename>-<frame_number>'.
        -- Find the index of the '-'
        local filename, frame_number = source:frame_video_offset(key)
        if predictions_by_filename[filename] == nil then
            predictions_by_filename[filename] = {}
        end
        predictions_by_filename[filename][frame_number] = prediction
    end

    -- Save predictions to HDF5.
    local output_file = hdf5.open(args.output_hdf5, 'w')
    for filename, predictions_table in pairs(predictions_by_filename) do
        if not args.recurrent then
            -- Ensure we have predictions for all the frames. If we don't, print
            -- a warning and set the prediction to a very small value.
            for i = 1, args.sequence_length-1 do
                if predictions_table[i] == nil then
                    log.warn('Missing prediction for', filename, i)
                    predictions_table[i] = torch.zeros(args.num_labels) - 1e9
                end
            end
        end
        output_file:write(filename, torch.cat(predictions_table, 2):t())
    end
end
