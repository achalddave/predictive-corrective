--[[
Updates a pre-trained VGG-16 network by
1. Removing the softmax layer
2. (Optionally) Chopping off the last FC layer
3. Adding an FC layer with a specified output size.
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
parser:flag('--keep_last_fc',
            'If specified, the last fully connected layer is not removed. ' ..
            'Instead, one ReLU and one Dropout layer is added ' ..
            '(in that order), and the new fully connected layer is added on ' ..
            'top of that.'):default(false)
local args = parser:parse()

model = torch.load(args.model)
model:remove() -- Remove softmax layer.
if args.keep_last_fc then
    local rectifiers = model:findModules('cudnn.ReLU')
    local dropouts = model:findModules('nn.Dropout')
    -- Add a clone of the last ReLU and Dropout layers.
    model:add(rectifiers[#rectifiers]:clone())
    model:add(dropouts[#dropouts]:clone())
else
    model:remove() -- Remove FC8 layer.
end

-- Should be either 4096 (FC7 output) or 1000 (FC8 output)
local linear_layers = model:findModules('nn.Linear')
local last_layer_output_size = linear_layers[#linear_layers].weight:size(1)
-- Add new output layer with args.num_classes outputs.
local new_output_layer = nn.Linear(last_layer_output_size, args.num_classes)
model:add(new_output_layer)

torch.save(args.output_model, model)
