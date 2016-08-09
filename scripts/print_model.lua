--[[ Print out a model's architecture. ]]--

local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local torch = require 'torch'
local nn = require 'nn'
require 'rnn'
require 'layers/CAvgTable'
require 'cunn'

local parser = argparse() {
    description = 'Print model architecture.'
}
parser:argument('model', 'Torch model')
local args = parser:parse()
print(torch.load(args.model))
