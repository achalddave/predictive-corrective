--[[
Replace last fully connected layer with a recurrent layer.
]]--

local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local torch = require 'torch'
local nn = require 'nn'
require 'rnn'

local parser = argparse() {
    description = 'Replace last fully connected layer with a recurrent layer.'
}
parser:argument('model', 'Torch model')
parser:argument('output_model', 'Output rnn model')

local args = parser:parse()

model = torch.load(args.model)

local fc_layers, containers = model:findModules('nn.Linear')
local last_fc = fc_layers[#fc_layers]
local output_size = last_fc.weight:size(1)
local container = containers[#containers]
for i = 1, #(container.modules) do
    if container.modules[i] == last_fc then
        container.modules[i] = nn.Recurrent(
            nn.Identity(),
            last_fc --[[input layer]],
            nn.Linear(output_size, output_size) --[[feedback]],
            nn.Identity() --[[transfer]])
    end
end
model = nn.Sequencer(model)
print(model)

torch.save(args.output_model, model)
