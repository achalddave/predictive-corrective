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

nn.DataParallelTable.deserializeNGPUs = cutorch.getDeviceCount()

local model = torch.load(args.model)
print('Loaded model')
local hidden_layer = model:findModules('nn.Recurrent')[1].feedbackModule
print(model:findModules('nn.Recurrent'))
print(model:findModules('nn.Recurrent')[1])
print(hidden_layer)


print('Saving matrix of size', hidden_layer.weight:size())
torch.save(args.output_path, hidden_layer.weight)
