--[[ Reset convolutional, fully-connected layer weights. ]]--

local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local torch = require 'torch'
local nn = require 'nn'
require 'rnn'
require 'cunn'

local nninit = require 'nninit'

local parser = argparse() {
    description = 'Reset weights in convolutional, fully-connected layers.'
}
parser:option('--model', 'Torch model'):count(1)
parser:option('--output', 'Output model'):count(1)

local args = parser:parse()

local model = torch.load(args.model)

local layer_types = {'cudnn.SpatialConvolution', 'nn.Linear'}

for _, layer_type in ipairs(layer_types) do
    print('Resetting layers of type:', layer_type)
    local layers = model:findModules(layer_type)
    for i, layer in ipairs(layers) do
        print('Resetting', layer_type, ':', i)
        layer:init('weight', nninit.xavier)
        layer:init('bias', nninit.constant, 0)
    end
end
torch.save(args.output, model)
