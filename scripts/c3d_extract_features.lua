local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local cv = require 'cv'
local torch = require 'torch'
local nn = require 'nn'
require 'rnn'
require 'cunn'
require 'image'
require 'cv.imgcodecs'
require 'cv.imgproc'

local parser = argparse() {
    description = 'Evaluate a Torch model on MultiTHUMOS.'
}
parser:argument('model', 'Model file')
parser:argument('frames', 'File with 16 frame paths separated by new lines')

local RESIZE_HEIGHT = 128
local RESIZE_WIDTH = 171
local CROP_SIZE = 112
local NUM_CHANNELS = 3
local SEQUENCE_LENGTH = 16
local VOL_MEAN = torch.load('/data/achald/pretrained_models/c3d/torch/caffemodel2json/volmean_sports1m.t7')

local args = parser:parse()
local file = torch.DiskFile(args.frames, 'r')

print('Loading frames')
input = torch.zeros(1, NUM_CHANNELS, SEQUENCE_LENGTH, CROP_SIZE, CROP_SIZE)
VOL_MEAN_CENTER = torch.zeros(NUM_CHANNELS, SEQUENCE_LENGTH,
                              CROP_SIZE, CROP_SIZE)
for i = 1, SEQUENCE_LENGTH do
    local path = file:readString('*l')
    local frame = cv.imread({path, cv.IMREAD_COLOR}):byte()
    frame = cv.resize({frame, {RESIZE_WIDTH, RESIZE_HEIGHT}})
    -- Convert from hxwxc to cxhxw
    frame = frame:permute(3, 1, 2)

    local cropped = image.crop(frame --[[src]],
                               'c' --[[center crop]],
                               CROP_SIZE --[[width]],
                               CROP_SIZE --[[height]])
    input[{{}, {}, i}] = cropped
    image.crop(VOL_MEAN_CENTER[{{}, i}] --[[dst]],
               VOL_MEAN[{{}, i}] --[[src]],
               'c' --[[center crop]],
               CROP_SIZE --[[width]],
               CROP_SIZE --[[height]])
end
print(input:double():mean())
input[1] = input[1] - VOL_MEAN_CENTER
print(input:double():mean())

print('Loading model')
model = torch.load(args.model):cuda()
model:evaluate()
output = model:forward(input:cuda())
max_val, argmax = output:max(1)
print(argmax, max_val)
_, sorted_indices = output:sort()
-- print(sorted_indices)
print(output:max())
