--[[ Evaluate a trained torch model on an LMDB.
--
-- Note that this should be run from the root directory, not from within
-- scripts/.
--]]

package.path = package.path .. ";../?.lua"

local argparse = require 'argparse'
local hdf5 = require 'hdf5'
local lmdb = require 'lmdb'
local torch = require 'torch'

local data_loader = require 'data_loader'
local evaluator = require 'evaluator'
local video_frame_proto = require 'video_util.video_frames_pb'

local parser = argparse() {
    description = 'Evaluate a Torch model on MultiTHUMOS.'
}
parser:argument('model', 'Model file.')
parser:argument('labeled_video_frames_lmdb',
                'LMDB containing LabeledVideoFrames to evaluate.')
parser:argument('labeled_video_frames_without_images_lmdb',
                'LMDB containing LabeledVideoFrames without images.')
parser:argument('output_hdf5', 'HDF5 to output predictions to')

local args = parser:parse()

-- More config variables.
local NUM_LABELS = 65
local BATCH_SIZE = 96
local GPU = 2
local MEANS = {96.8293, 103.073, 101.662}
local CROP_SIZE = 224

cutorch.setDevice(GPU)
torch.setdefaulttensortype('torch.FloatTensor')

-- Load model.
local model = torch.load(args.model)

-- Open database.
local data_loader = data_loader.DataLoader(
    args.labeled_video_frames_lmdb,
    args.labeled_video_frames_without_images_lmdb,
    data_loader.PermutedSampler,
    NUM_LABELS)

-- Pass each image in the database through the model, collect predictions and
-- groundtruth.
local gpu_inputs = torch.CudaTensor()
local all_predictions
local all_labels
local predictions_by_keys = {}
local samples_complete = 0

while true do
    if samples_complete == data_loader:num_samples() then
        break
    end
    local to_load = BATCH_SIZE
    if samples_complete + BATCH_SIZE > data_loader:num_samples() then
        to_load = data_loader:num_samples() - samples_complete
    end
    images_table, labels_table, batch_keys = data_loader:load_batch(
        to_load, true --[[return_keys]])
    data_loader:fetch_batch_async(to_load)

    local images = torch.Tensor(
        #images_table, images_table[1]:size(1), CROP_SIZE, CROP_SIZE)
    local labels = torch.ByteTensor(#labels_table, NUM_LABELS)
    for i, img in ipairs(images_table) do
        -- Process image after converting to the default Tensor type.
        -- (Originally, it is a ByteTensor).
        img = img:typeAs(images)
        local x_origin = (img:size(3) - CROP_SIZE) / 2
        local y_origin = (img:size(2) - CROP_SIZE) / 2
        img = image.crop(img, x_origin, y_origin, x_origin + CROP_SIZE, y_origin + CROP_SIZE)
        for channel = 1, 3 do
            img[{{channel}, {}, {}}]:add(-MEANS[channel])
        end
        images[i] = img
        labels[i] = labels_table[i]
    end

    gpu_inputs:resize(images:size()):copy(images)

    local predictions = model:forward(gpu_inputs):type(
        torch.getdefaulttensortype())
    labels = labels:type('torch.ByteTensor')
    for i = 1, #images_table do
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

    print('Number of labels with groundtruth:', num_labels_with_groundtruth)
    local map_so_far = evaluator.compute_mean_average_precision(
        all_predictions, all_labels)
    local batch_map = evaluator.compute_mean_average_precision(
        predictions, labels)
    print(string.format(
        '%s: Finished %d/%d. mAP so far: %.5f, batch mAP: %.5f',
        os.date('%X'), samples_complete, data_loader:num_samples(), map_so_far,
        batch_map))
    collectgarbage()
    samples_complete = samples_complete + to_load
end
for i = 1, all_predictions:size(2) do
    local ap = evaluator.compute_mean_average_precision(
        all_predictions[{{}, {i}}], all_labels[{{}, {i}}])
    print(string.format('Class %d\t AP: %.5f', i, ap))
end

local map = evaluator.compute_mean_average_precision(
    all_predictions, all_labels)
print('mAP: ', map)

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
for filename, predictions_table in pairs(predictions_by_filename) do
    output_file:write(filename, torch.cat(predictions_table, 1))
end
