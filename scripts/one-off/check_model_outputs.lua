--[[ Ensure that two models perform the same forward/backward computation.
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
    description = 'Ensure that two models perform the same computation.'
}
parser:option('--left', 'Torch model'):count(1)
parser:option('--right', 'Output model'):count(1)

torch.manualSeed(0)
cutorch.manualSeedAll(0)
math.randomseed(0)
local args = parser:parse()
left = torch.load(args.left)
left:cuda():evaluate()

right = torch.load(args.right)
right:cuda():evaluate()

local criterion = nn.SequencerCriterion(nn.MultiLabelSoftMarginCriterion():cuda())
local criterion_gradients

left:clearState()
right:clearState()

i = torch.ones(4, 1, 3, 224, 224):cuda()
print('input', i:norm())
left_outputs = left:forward(i):clone()
right_outputs = right:forward(i):clone()

for j = 1,2 do
    print(j)
    left:zeroGradParameters()
    right:zeroGradParameters()
    i = torch.rand(4, 1, 3, 224, 224):cuda()
    print('input', i:norm())
    left_outputs = left:forward(i):clone()
    right_outputs = right:forward(i):clone()

    print(left_outputs:norm())
    print(right_outputs:norm())
    assert(torch.all(torch.le(torch.abs(left_outputs - right_outputs), 1e-4)))

    left_labels = left_outputs:clone()
    left_labels:zero()
    right_labels = left_labels:clone()

    criterion:forward(left_outputs, left_labels)
    criterion_gradients = criterion:backward(left_outputs, left_labels)
    print('crit grad', criterion_gradients:norm())

    left:backward(i, criterion_gradients)
    left:updateParameters(0.01)

    right:backward(i, criterion_gradients)
    right:updateParameters(0.01)

    left:clearState()
    right:clearState()

    i = torch.rand(4, 1, 3, 224, 224):cuda()
    left_outputs = left:forward(i):clone()
    right_outputs = right:forward(i):clone()
    print('left after', left_outputs:norm())
    print('right after', right_outputs:norm())

    assert(torch.all(torch.le(torch.abs(left_outputs - right_outputs), 1e-4)))
end
