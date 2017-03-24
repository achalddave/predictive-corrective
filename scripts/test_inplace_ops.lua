--[[ Test that a model provides the same results if inplace ops are turned off.
--
-- This forces all in place operations (e.g. ReLU) to copy their inputs, and
-- compares it against the default in place setting for the model.
--
-- Of course, this is not fool-proof, but just a quick check with some random
-- inputs, which is good enough to catch obvious differences.
--]]

local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local torch = require 'torch'
local nn = require 'nn'
require 'rnn'
require 'layers/init'
require 'cunn'

local parser = argparse() {
    description = 'Test that in place operations are not affecting output'
}
parser:option('--model', 'Torch model'):count(1)
parser:flag('--decorate_sequencer')

local args = parser:parse()

INPUT_SIZE = {4, 1, 3, 224, 224}

local model = torch.load(args.model)
print('Loaded model')
if args.decorate_sequencer then
    model = nn.Sequencer(model)
end
model = model:cuda()
-- model:evaluate()
print('evaluate')

-- Yes, we have to clone the model: forwarding inputs, then setting in place to
-- false doesn't do the right thing, since the output of the in place operation
-- is already set to the storage of the previous layer's output.
local model_no_inplace = model:clone()
model_no_inplace:apply(function(x)
    if torch.isTypeOf(x, 'nn.Threshold') or
            torch.isTypeOf(x, 'cudnn._Pointwise') then
        x.inplace = false
    end
end)

for _, dropout in ipairs(model:findModules('nn.Dropout')) do
    dropout.p = 0
end
for _, dropout in ipairs(model_no_inplace:findModules('nn.Dropout')) do
    dropout.p = 0
end

-- Default input
local gradients
local input = torch.ones(unpack(INPUT_SIZE)):cuda()
print('Forwarding')

local output_default = model:forward(input):clone()

gradients = torch.rand(output_default:size()):cuda()
local grad_out_default = model:backward(input, gradients):clone()

local output_no_inplace = model_no_inplace:forward(input):clone()
print(output_no_inplace:mean())
local grad_out_no_inplace = model_no_inplace:backward(input, gradients):clone()
assert(torch.all(torch.le(torch.abs(output_default - output_no_inplace), 1e-7)))
assert(torch.all(torch.le(torch.abs(grad_out_default - grad_out_no_inplace), 1e-7)))
print('Success!')
