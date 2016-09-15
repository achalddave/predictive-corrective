--[[
Create a pyramid like network, where intermediate layers are averaged over time.

This network takes as input tensors of shape
(sequence_length, batch_size, num_channels, height, width), and outputs
(1, batch_size, num_outputs), where
the singleton dimension is for compatibility with DataParallelTable, which
expects the batch size dimension index to remain constant.
]]--

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
    description = 'Replace last fully connected layer with a recurrent layer.'
}
parser:option('--model', 'Torch model'):count(1)
parser:option('--output', 'Output model'):count(1)
parser:flag('--weighted_avg',
            'Whether to add learnable weights for averaging layers. ' ..
            'Initialized to 0.5.'):default(false)
parser:flag('--untie_weights',
            "If specified, don't tie weights of parallel stubs."):default(false)

local args = parser:parse()

local SEQUENCE_LENGTH = 4

local model = torch.load(args.model)
if torch.isTypeOf(model, 'nn.DataParallelTable') then
    model = model:get(1)
end
if torch.isTypeOf(model, 'nn.Sequencer') then
    -- Remove Sequencer and Recursor decorators.
    model = model:get(1):get(1)
end

local CONV4_3_INDEX = 22
local CONV5_3_INDEX = 29

function extract_stub(model, start_index, end_index)
    local output_model = nn.Sequential()
    for i = start_index, end_index do
        output_model:add(model:get(i):clone(
            'weight', 'bias', 'gradWeight', 'gradBias'))
    end
    return output_model
end

function create_averager(sequence_length, weighted)
    --[[
    -- Create a model that takes in a table of inputs and averages every
    -- consecutive pair output (without overlap).  The number of inputs to the
    -- model is `sequence_length` and the number of outputs from the model is
    -- `sequence_length / 2`.
    --]]
    assert(sequence_length % 2 == 0, 'Sequence length must be even.')
    weighted = weighted == nil and false or weighted

    -- Takes average of table of inputs
    local averaging_layer
    do
        if not weighted then
            averaging_layer = nn.CAvgTable()
        else
            averaging_layer = nn.Sequential()
            local multiplications = nn.ParallelTable()
            local multiplier = nn.Mul()
            multiplier.weight[1] = 0.5  -- Initialize to 0.5
            multiplications:add(multiplier:clone())
            multiplications:add(multiplier:clone())
            averaging_layer:add(multiplications)
            averaging_layer:add(nn.CAddTable())
        end
    end

    -- Average every 2 model_stub outputs.
    local pair_averagers = nn.ConcatTable()
    local num_pairs = sequence_length / 2
    for pair_index = 1, num_pairs do
        local left_index = pair_index * 2 - 1 -- Index of 'left' frame in pair.
        local averager = nn.Sequential()
        averager:add(nn.NarrowTable(left_index, 2)) -- Select the current pair.
        averager:add(averaging_layer:clone()) -- Average the current pair.
        pair_averagers:add(averager)
    end
    local output_model = nn.Sequential()
    output_model:add(pair_averagers)
    return output_model
end

local stub_to_conv4_3 = extract_stub(model, 1, CONV4_3_INDEX)
local parallel_stubs_to_conv4_3 = nn.ParallelTable()
for _ = 1, SEQUENCE_LENGTH do
    if args.untie_weights then
        parallel_stubs_to_conv4_3:add(stub_to_conv4_3:clone())
    else
        parallel_stubs_to_conv4_3:add(stub_to_conv4_3:sharedClone())
    end
end
local conv4_3_averager = create_averager(SEQUENCE_LENGTH, args.weighted_avg)

-- Conv4_3 -> Conv5_3
local stub_conv4_3_to_conv5_3 = extract_stub(
    model, CONV4_3_INDEX + 1, CONV5_3_INDEX)
local parallel_stubs_conv4_3_to_conv5_3 = nn.ParallelTable()
for _ = 1, (SEQUENCE_LENGTH/2) do
    if args.untie_weights then
        parallel_stubs_conv4_3_to_conv5_3:add(stub_conv4_3_to_conv5_3:clone())
    else
        parallel_stubs_conv4_3_to_conv5_3:add(
            stub_conv4_3_to_conv5_3:sharedClone())
    end
end
local conv5_3_averager = create_averager(SEQUENCE_LENGTH / 2, args.weighted_avg)

-- Conv5_3 -> Output
local stub_conv5_3_to_output = extract_stub(
    model, CONV5_3_INDEX + 1, model:size())

local output_model = nn.Sequential()
output_model:add(nn.SplitTable(1))
output_model:add(parallel_stubs_to_conv4_3)
output_model:add(conv4_3_averager)
output_model:add(parallel_stubs_conv4_3_to_conv5_3)
output_model:add(conv5_3_averager)
-- At this point we should have a table with exactly one element, which is the
-- tensor of averaged conv5 activations. Use JoinTable to convert it to a
-- tensor. (We could also use SelectTable, but JoinTable will cause errors if
-- the table somehow has more than one elements, which is nice.)
output_model:add(nn.JoinTable(1))
output_model:add(stub_conv5_3_to_output)
-- Output should be (1, batch_size, num_labels) to be compatible with
-- DataParallelTable.
output_model:add(nn.Unsqueeze(1))
print(output_model)
torch.save(args.output, output_model)
