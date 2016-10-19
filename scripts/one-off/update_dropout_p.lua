package.path = package.path .. ";layers/?.lua"

local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local torch = require 'torch'
local nn = require 'nn'
require 'rnn'
require 'cunn'

require 'CAvgTable'

local parser = argparse() {
    description = 'Increase dropout probability in existing dropout layers.'
}
parser:option('--model', 'Torch model'):count(1)
parser:option('--output', 'Output model'):count(1)
parser:option('--new_p', 'New dropout probability'):count(1):convert(tonumber)

local args = parser:parse()
local model = torch.load(args.model)

local dropout_layers, _ = model:findModules('nn.Dropout')
for _, dropout in ipairs(dropout_layers) do
    print('Dropout p was', dropout.p, '; updating to ', args.new_p)
    dropout.p = args.new_p
end
print(model)
torch.save(args.output, model)
