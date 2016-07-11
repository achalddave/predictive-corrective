--[[
Reinitializes the last (fully connected) layer of a pre-trained ResNet network.
]]--

local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local torch = require 'torch'
local nn = require 'nn'

local parser = argparse() {
    description = 'Replaces the last layer of a pre-trained model.'
}
parser:argument('model', 'Torch model')
parser:argument('num_classes', 'Number of outputs the last layer should have.')
      :convert(tonumber)
parser:argument('output_model', 'Output MultiTHUMOS model')

local args = parser:parse()

model = torch.load(args.model)
-- ResNet model provided by torch has no softmax layer.
local fc8 = model:get(model:size())
model:remove() -- Remove FC8 layer.

-- Add new FC8 layer with args.num_classes outputs.
local fc8_output_size = fc8.weight:size(2)
print(fc8_output_size)
local new_fc8 = nn.Linear(fc8_output_size, args.num_classes)
new_fc8.name = fc8.name
model:add(new_fc8)

torch.save(args.output_model, model)
