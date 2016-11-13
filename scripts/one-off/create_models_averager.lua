local argparse = require 'argparse'
local torch = require 'torch'
local nn = require 'nn'
require 'rnn'
require 'cudnn'
require 'cunn'
require 'cutorch'

require 'layers/init'

local parser = argparse() {
    description = 'Create a hierarchical residual network..'
}
parser:option('--model_1', 'Torch model 1'):count(1)
parser:option('--model_2', 'Torch model 2'):count(1)
parser:option('--output_model', 'Output model'):count(1)

local args = parser:parse()

local model_1 = torch.load(args.model_1)
local model_2 = torch.load(args.model_2)

if torch.isTypeOf(model_1, 'nn.DataParallelTable') then
    model_1 = model_1:get(1)
end
if torch.isTypeOf(model_2, 'nn.DataParallelTable') then
    model_2 = model_2:get(1)
end

local concat = nn.ParallelTable()
concat:add(model_1)
concat:add(model_2)

local output_model = nn.Sequential()
output_model:add(nn.SplitTable(1))
output_model:add(concat)
output_model:add(nn.CAddTable())
output_model:add(nn.MulConstant(0.5))

torch.save(args.output_model, output_model)
