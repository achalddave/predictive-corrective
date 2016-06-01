package.path = package.path .. ";../?.lua"

local argparse = require 'argparse'
local lmdb = require 'lmdb'
local torch = require 'torch'

local evaluator = require 'evaluator'
local video_frame_proto = require 'video_util.video_frames_pb'

local parser = argparse() {
    description = 'Evaluate a Torch model on MultiTHUMOS.'
}
parser:argument('model', 'Model file.')
parser:argument('labeled_video_frames_lmdb',
                'LMDB containing LabeledVideoFrames to evaluate.')

local args = parser:parse()

-- More config variables.
local NUM_LABELS = 65
local BATCH_SIZE = 64
local GPU = 1
local MEANS = {96.8293, 103.073, 101.662}
local CROP_SIZE = 224

cutorch.setDevice(GPU)
torch.setdefaulttensortype('torch.FloatTensor')

-- Load model.
local model = torch.load(args.model)

-- Open database.
local db = lmdb.env { Path = args.labeled_video_frames_lmdb }
db:open()
local transaction = db:txn(true --[[readonly]])
local cursor = transaction:cursor()
local num_samples = db:stat().entries

-- Pass each image in the database through the model, collect predictions and
-- groundtruth.
local gpu_inputs = torch.CudaTensor()
local finished = false
local all_predictions
local all_labels
local samples_complete = 0
-- TODO(achald): Remove this
while true do
    if finished then break end
    local batch_images = {}
    local batch_labels = {}
    for i = 1, BATCH_SIZE do
        local video_frame = video_frame_proto.LabeledVideoFrame()
        local _, video_frame_proto_tensor = cursor:get()
        video_frame:ParseFromString(video_frame_proto_tensor:storage():string())

        local image_proto = video_frame.frame.image
        local image_storage = torch.ByteStorage()
        image_storage:string(image_proto.data)
        local img = torch.ByteTensor(image_storage):reshape(
            image_proto.channels, image_proto.height, image_proto.width)
        for channel = 1, 3 do
            img[{{channel}, {}, {}}]:add(-MEANS[channel])
        end
        img = image.crop(img, "c" --[[center crop]], CROP_SIZE, CROP_SIZE)
        table.insert(batch_images, img)

        -- Load labels in an array.
        local labels = torch.ByteTensor(1, NUM_LABELS):zero()
        for _, label in ipairs(video_frame.label) do
            -- Label ids start at 0.
            labels[{1, label.id + 1}] = 1
        end
        table.insert(batch_labels, labels)

        if not cursor:next() then
            finished = true
            break
        end
        samples_complete = samples_complete + 1
    end
    gpu_inputs:resize(#batch_images,
                      batch_images[1]:size(1),
                      batch_images[1]:size(2),
                      batch_images[1]:size(3))
    local predictions = model:forward(gpu_inputs):type(
        torch.getdefaulttensortype())
    local batch_labels_tensor = torch.cat(batch_labels, 1):type('torch.ByteTensor')

    if all_predictions == nil then
        all_predictions = predictions
        all_labels = batch_labels_tensor
    else
        all_predictions = torch.cat(all_predictions, predictions, 1)
        all_labels = torch.cat(all_labels, batch_labels_tensor, 1)
    end

    local map_so_far = evaluator.compute_mean_average_precision(
        all_predictions, all_labels)
    print(string.format(
        '%s: Finished %d/%d. mAP so far: %.5f',
        os.date('%X'), samples_complete, num_samples, map_so_far))
    collectgarbage()
end
local map = evaluator.compute_mean_average_precision(
    all_predictions, all_labels)
print('mAP: ', map)
