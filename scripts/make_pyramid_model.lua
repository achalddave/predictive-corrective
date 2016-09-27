--[[
Create a pyramid like network, where intermediate layers are merged over time.

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

local nninit = require 'nninit'

require 'CAvgTable'

local parser = argparse() {
    description = 'Replace last fully connected layer with a recurrent layer.'
}
parser:option('--model', 'Torch model'):count(1)
parser:option('--output', 'Output model'):count(1)
parser:option('--merge_type',
              'Type of merging. Options: \n' ..
              'sum: Sum the two inputs. \n' ..
              'avg: Average the two inputs. \n'):count(1)
parser:option('--weight_type',
              'Specify relationship between and initialization of pyramid ' ..
              'weights. Options: \n' ..
              'tied: Tie weights between pyramid paths. \n' ..
              'untied: Untie the weights between pyramid paths. \n' ..
              'residual_tied: Initialize and tie weights on all but last ' ..
                             'frame. \n')
      :default('tied')
parser:flag('--weighted_merge',
            'Whether to add learnable weights before merging. ' ..
            'Initialized to 0.5.'):default(false)
parser:flag('--untie_weights',
            "If specified, don't tie weights of parallel stubs."):default(false)

local args = parser:parse()

local MERGE_OPTIONS = {
    sum = 1,
    avg = 2
}
local merge_type = MERGE_OPTIONS[args.merge_type]
assert(merge_type ~= nil, 'Invalid merge option.')

local WEIGHT_TYPE_OPTIONS = { tied = 1, untied = 2, residual_tied = 3 }
local weight_type = WEIGHT_TYPE_OPTIONS[args.weight_type]
assert(weight_type ~= nil, 'Invalid weight type.')

local SEQUENCE_LENGTH = 4

local model = torch.load(args.model)
if torch.isTypeOf(model, 'nn.DataParallelTable') then
    model = model:get(1)
end
if torch.isTypeOf(model, 'nn.Sequencer') then
    -- Remove Sequencer and Recursor decorators.
    model = model:get(1):get(1)
end

local CONV4_3_INDEX = 22 -- HACK!
local CONV5_3_INDEX = 29 -- HACK!

local function extract_stub(model, start_index, end_index)
    local output_model = nn.Sequential()
    for i = start_index, end_index do
        output_model:add(model:get(i):sharedClone())
    end
    return output_model
end

local function create_merger(sequence_length, merge_type, weighted)
    --[[
    -- Create a model that takes in a table of inputs and merge every
    -- consecutive pair output (without overlap).  The number of inputs to the
    -- model is `sequence_length` and the number of outputs from the model is
    -- `sequence_length / 2`.
    --]]
    assert(sequence_length % 2 == 0, 'Sequence length must be even.')
    weighted = weighted == nil and false or weighted

    -- Merge table of inputs
    local merging_layer = nn.Sequential()
    do
        if weighted then
            local multiplications = nn.ParallelTable()
            local multiplier = nn.Mul()
            multiplier.weight[1] = 1  -- Initialize to 1.
            multiplications:add(multiplier)
            multiplications:add(multiplier:clone())
            merging_layer:add(multiplications)
            merging_layer:add(nn.CAddTable())
        end

        if merge_type == MERGE_OPTIONS['sum'] then
            merging_layer = nn.CAddTable()
        elseif merge_type == MERGE_OPTIONS['avg'] then
            merging_layer = nn.CAvgTable()
        else
            error('Invalid merge type:', merge_type)
        end
    end

    -- Merge every 2 model_stub outputs.
    local pair_mergers = nn.ConcatTable()
    local num_pairs = sequence_length / 2
    for pair_index = 1, num_pairs do
        local left_index = pair_index * 2 - 1 -- Index of 'left' frame in pair.
        local merger = nn.Sequential()
        merger:add(nn.NarrowTable(left_index, 2)) -- Select the current pair.
        merger:add(merging_layer:clone()) -- Merge the current pair.
        pair_mergers:add(merger)
    end
    local output_model = nn.Sequential()
    output_model:add(pair_mergers)
    return output_model
end

local function reset_weights(model)
    local reset_conv_layers = model:findModules(
        'cudnn.SpatialConvolution')
    for _, layer in ipairs(reset_conv_layers) do
        layer:init('weight', nninit.xavier)
        layer:init('bias', nninit.constant, 0)
    end
    local reset_linear_layers = model:findModules('nn.Linear')
    for _, layer in ipairs(reset_linear_layers) do
        layer:init('weight', nninit.xavier)
        layer:init('bias', nninit.constant, 0)
    end
end

local stub_to_conv4_3 = extract_stub(model, 1, CONV4_3_INDEX)
local parallel_stubs_to_conv4_3 = nn.ParallelTable()
-- For residual weights.
local reset_stub_to_conv4_3 = stub_to_conv4_3:clone()
reset_weights(reset_stub_to_conv4_3)
for i = 1, SEQUENCE_LENGTH do
    local stub
    if weight_type == WEIGHT_TYPE_OPTIONS.untied then
        stub = stub_to_conv4_3:clone()
    elseif weight_type == WEIGHT_TYPE_OPTIONS.tied then
        stub = stub_to_conv4_3:sharedClone()
    elseif weight_type == WEIGHT_TYPE_OPTIONS.residual_tied then
        if i ~= SEQUENCE_LENGTH then
            -- Share weights on all but the last frame.
            stub = reset_stub_to_conv4_3:sharedClone()
        else
            stub = stub_to_conv4_3:clone()
        end
    end
    parallel_stubs_to_conv4_3:add(stub)
end
local conv4_3_merger = create_merger(
    SEQUENCE_LENGTH, merge_type, args.weighted_merge)

-- Conv4_3 -> Conv5_3
local stub_conv4_3_to_conv5_3 = extract_stub(
    model, CONV4_3_INDEX + 1, CONV5_3_INDEX)
local parallel_stubs_conv4_3_to_conv5_3 = nn.ParallelTable()
-- For residual weights.
local reset_stub_conv4_3_to_conv5_3 = stub_conv4_3_to_conv5_3:clone()
reset_weights(reset_stub_conv4_3_to_conv5_3)
for i = 1, (SEQUENCE_LENGTH/2) do
    local stub
    if weight_type == WEIGHT_TYPE_OPTIONS.untied then
        stub = stub_conv4_3_to_conv5_3:clone()
    elseif weight_type == WEIGHT_TYPE_OPTIONS.tied then
        stub = stub_conv4_3_to_conv5_3:sharedClone()
    elseif weight_type == WEIGHT_TYPE_OPTIONS.residual_tied then
        if i ~= (SEQUENCE_LENGTH / 2) then
            -- Share weights on all but the last frame.
            stub = reset_stub_conv4_3_to_conv5_3:sharedClone()
        else
            stub = stub_conv4_3_to_conv5_3:clone()
        end
    end
    parallel_stubs_conv4_3_to_conv5_3:add(stub)
end
local conv5_3_merger = create_merger(SEQUENCE_LENGTH / 2, merge_type, args.weighted_merge)

-- Conv5_3 -> Output
local stub_conv5_3_to_output = extract_stub(
    model, CONV5_3_INDEX + 1, model:size())

local output_model = nn.Sequential()
output_model:add(nn.SplitTable(1))
output_model:add(parallel_stubs_to_conv4_3)
output_model:add(conv4_3_merger)
output_model:add(parallel_stubs_conv4_3_to_conv5_3)
output_model:add(conv5_3_merger)
-- At this point we should have a table with exactly one element, which is the
-- tensor of merged conv5 activations. Use JoinTable to convert it to a
-- tensor. (We could also use SelectTable, but JoinTable will cause errors if
-- the table somehow has more than one elements, which is nice.)
output_model:add(nn.JoinTable(1))
output_model:add(stub_conv5_3_to_output)
-- Output should be (1, batch_size, num_labels) to be compatible with
-- DataParallelTable.
output_model:add(nn.Unsqueeze(1))
print(output_model)
torch.save(args.output, output_model)
