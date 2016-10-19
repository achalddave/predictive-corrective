--[[ Add LSTM to a VGG-16 network.
--
-- This would ideally be a part of make_recurrent.lua, but it's a little more
-- effort than necessary.
--]]

package.path = package.path .. ";layers/?.lua"

local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local torch = require 'torch'
local nn = require 'nn'
require 'cunn'
require 'rnn'

require 'CAvgTable'

local parser = argparse() {
    description = 'Replace last fully connected layer with a recurrent layer.'
}
parser:option('--model', 'Torch model'):count(1)
parser:option('--output', 'Output rnn model'):count(1)

local args = parser:parse()

local LSTM_OUTPUT_SIZE = 512
local model = torch.load(args.model)
if torch.isTypeOf(model, 'nn.DataParallelTable') then
    model = model:get(1)
end

local fc_layers, fc_containers = model:findModules('nn.Linear')

-- Remove the last layer.
model:remove()

-- Add an LSTM layer.
local lstm_input = fc_layers[#fc_layers - 1].weight:size(1)
model:add(nn.LSTM(lstm_input, LSTM_OUTPUT_SIZE))

-- Add a final linear layer.
local num_labels = fc_layers[#fc_layers].weight:size(1)
model:add(nn.Linear(LSTM_OUTPUT_SIZE, num_labels))

print(model)
torch.save(args.output, model)
