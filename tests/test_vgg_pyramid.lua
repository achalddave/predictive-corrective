-- Test a pyramid network that starts averaging at conv4-3. Assumes a network
-- with inputs of length 4, where input (x_1, x_2) and (x_3, x_4) are averaged
-- at conv4-3, and later (conv5-3_1, conv5-3_2) are averaged.
local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local torch = require 'torch'
local nn = require 'nn'
require 'rnn'
require 'cunn'
require 'layers/CAvgTable'

local parser = argparse()
parser:option('--single_model', 'Single frame model'):count(1)
parser:option('--pyramid_model', 'Pyramid model'):count(1)
local args = parser:parse()

local single_model = torch.load(args.single_model):cuda()
local pyramid_model = torch.load(args.pyramid_model):cuda()
single_model:evaluate()
pyramid_model:evaluate()

-- Make all ReLUs not be in place in single mode.
do
    local relus = single_model:findModules('cudnn.ReLU')
    for i = 1, #relus do relus[i].inplace = false end

    relus = pyramid_model:findModules('cudnn.ReLU')
    for i = 1, #relus do relus[i].inplace = false end
end

do -- Check that if replicated input is passed, output is the same.
    local input = torch.rand(4, 1, 3, 224, 224):cuda()
    for i = 1, 4 do input[i] = input[1]:clone() end
    assert(torch.all(torch.eq(
        pyramid_model:forward(input),
        single_model:forward(input[{{1}}]))))
    print('Output matches for replicated input')
end

local input = torch.rand(4, 1, 3, 224, 224):cuda()

-- Check that conv4_3 matches.
local single_c43 = 10
local single_c43_outputs = {}
local pyramid_c43 = {single_c43,
                     single_c43 + 10,
                     single_c43 + 20,
                     single_c43 + 30}
pyramid_model:forward(input)
for i = 1, 4 do
    single_model:forward(input[{{i}}])
    single_c43_outputs[i] = single_model:findModules(
            'cudnn.SpatialConvolution')[single_c43].output:clone()
    local pyramid_c43_output = pyramid_model:findModules(
            'cudnn.SpatialConvolution')[pyramid_c43[i]].output
    assert(torch.all(torch.eq(
        single_c43_outputs[i],
        pyramid_c43_output
    )))
    print(string.format('Conv4-3 matches for input %d', i))
end

-- Check that averaged conv4_3 matches.
for i = 1, 2 do
    local pyramid_averaged = pyramid_model:findModules('nn.CAvgTable')[i].output
    local expected_average = 0.5 * (single_c43_outputs[2*i-1]
                                    + single_c43_outputs[2*i])
    assert(torch.all(torch.eq(pyramid_averaged, expected_average)))
end
print('Conv4-3 averages match.')

-- Check that the Conv5_3 average is correct given the input. (We don't
-- explicitly test that the conv5-3 input matches what it should be, since the
-- conv5-3 input is computed from averages of conv4-3 and can't easily be
-- validated against the single frame model.)
pyramid_c53_1 = pyramid_c43[#pyramid_c43] + 3
pyramid_c53_2 = pyramid_c53_1 + 3
local pyramid_c53_1_output = pyramid_model:findModules(
    'cudnn.SpatialConvolution')[pyramid_c53_1].output
local pyramid_c53_2_output = pyramid_model:findModules(
    'cudnn.SpatialConvolution')[pyramid_c53_2].output
local expected_average = 0.5 * (pyramid_c53_1_output + pyramid_c53_2_output)
local pyramid_average = pyramid_model:findModules('nn.CAvgTable')[3].output
assert(torch.all(torch.eq(expected_average, pyramid_average)))
print('Conv5-3 average is correct given conv5-3 input')
