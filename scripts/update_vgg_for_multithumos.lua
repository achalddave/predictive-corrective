--[[
Updates a pre-trained VGG-16 network by chopping off the last layer and adding
a cross entropy layer.
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
model:remove() -- Remove softmax layer.
model:remove() -- Remove FC8 layer.

-- Add new FC8 layer with args.num_classes outputs.
local new_fc8 = nn.Linear(4096, args.num_classes)
new_fc8.name = 'fc8'
model:add(new_fc8)

-- Add sigmoid layer.
local sigmoid = nn.Sigmoid()
sigmoid.name = 'sigmoid'
model:add(sigmoid)

torch.save(args.output_model, model)
