local argparse = require 'argparse'
local cutorch = require 'cutorch'
local nn = require 'nn'
local torch = require 'torch'
require 'rnn'
require 'cunn'
require 'cudnn'

local parser = argparse()

parser:argument('model', 'Model file')
parser:argument('output_path', 'Output path')

local args = parser:parse()
nn.DataParallelTable.deserializeNGPUs = 1
cutorch.setDevice(1)

local model = torch.load(args.model)
print('Loaded model')
print(model)
print(model:findModules('nn.Recurrent'))
local hidden_layer = model:findModules('nn.Recurrent')[1].feedbackModule

print('Saving matrix of size', hidden_layer.weight:size())
torch.save(args.output_path, hidden_layer.weight)
