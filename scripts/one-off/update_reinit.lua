package.path = package.path .. ";layers/?.lua"

local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local torch = require 'torch'
local nn = require 'nn'
require 'rnn'
require 'cunn'

require 'layers/init'

local parser = argparse() {
    description = 'Update reinitialize rate in torch model'
}
parser:option('--model', 'Torch model'):count(1)
parser:option('--output', 'Output model'):count(1)
parser:option('--reinit_rate', 'New reinit rate'):count(1):convert(tonumber)

local args = parser:parse()
local model = torch.load(args.model)

local reinit_types = {
    'nn.CRollingDiffTable', 'nn.PeriodicResidualTable', 'nn.CCumSumTable'}
for _, reinit_type in ipairs(reinit_types) do
    local reinit_layers, _ = model:findModules(reinit_type)
    assert(#reinit_layers == 1)
    for _, reinit_layer in ipairs(reinit_layers) do
        reinit_layer:set_reinitialize_rate(args.reinit_rate)
    end
end

torch.save(args.output, model)
