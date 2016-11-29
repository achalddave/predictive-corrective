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
    description = 'Create a pyramid shaped network.'
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
              'untie_last: Tie weight except for on the last frame. \n' ..
              'residual_untie_last: Initialize and tie weights on all but ' ..
                             'last frame. \n')
      :default('tied')
parser:flag('--weighted_merge',
            'Whether to add learnable weights before merging. ' ..
            'Initialized to 0.5.'):default(false)

local args = parser:parse()

local MERGE_LAYER_INDICES = {22 --[[conv43]], 29 --[[conv53]]}
-- Number of input frames.
local SEQUENCE_LENGTH = 4
-- How many consecutive inputs to merge at each merging step.
local MERGE_INPUT_LENGTH = 2

local MERGE_OPTIONS = { sum = 1, avg = 2 }
local WEIGHT_TYPE_OPTIONS = {
    tied = 1,
    untied = 2,
    untie_last = 3,
    residual_untie_last = 4
}

local merge_type = MERGE_OPTIONS[args.merge_type]
assert(merge_type ~= nil, 'Invalid merge option.')

local weight_type = WEIGHT_TYPE_OPTIONS[args.weight_type]
assert(weight_type ~= nil, 'Invalid weight type.')

local model = torch.load(args.model)
if torch.isTypeOf(model, 'nn.DataParallelTable') then
    model = model:get(1)
end
if torch.isTypeOf(model, 'nn.Sequencer') then
    -- Remove Sequencer and Recursor decorators.
    model = model:get(1):get(1)
end

local function extract_stub(model, start_index, end_index)
    local output_model = nn.Sequential()
    for i = start_index, end_index do
        output_model:add(model:get(i):sharedClone())
    end
    return output_model
end

local function create_merger(
    sequence_length, merge_input_length, merge_type, weighted)
    --[[
    -- Create a model that takes in a table of inputs and merge every
    -- consecutive pair output (without overlap).  The number of inputs to the
    -- model is `sequence_length` and the number of outputs from the model is
    -- `sequence_length / merge_input_length`.
    --]]
    assert(sequence_length % merge_input_length == 0,
           'Sequence length must be even.')
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
            merging_layer = nn.Sequential()
            merging_layer:add(nn.CAddTable())
            merging_layer:add(nn.MulConstant(1.0 / merge_input_length))
        end
    end

    -- Merge every `merge_input_length` model_stub outputs.
    local mergers = nn.ConcatTable()
    local num_mergers = sequence_length / merge_input_length
    for merge_index = 1, num_mergers do
        -- Index of first frame in merge inputs.
        local left_index = (merge_index - 1) * merge_input_length + 1
        local merger = nn.Sequential()
        -- Select merge inputs
        merger:add(nn.NarrowTable(left_index, merge_input_length))
        merger:add(merging_layer:clone()) -- Merge the current inputs.
        mergers:add(merger)
    end
    local output_model = nn.Sequential()
    output_model:add(mergers)
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

local previous_layer_index = 0
local model_stubs = {}
for i, layer_index in ipairs(MERGE_LAYER_INDICES) do
    local stub = extract_stub(model, previous_layer_index + 1, layer_index)
    local parallel_stubs = nn.ParallelTable()
    local reset_stub = stub:clone()
    reset_weights(reset_stub)

    local input_length = SEQUENCE_LENGTH / (MERGE_INPUT_LENGTH^(i - 1))
    for step = 1, input_length do
        local new_stub
        if weight_type == WEIGHT_TYPE_OPTIONS.untied then
            new_stub = stub:clone()
        elseif weight_type == WEIGHT_TYPE_OPTIONS.tied then
            new_stub = stub:sharedClone()
        elseif weight_type == WEIGHT_TYPE_OPTIONS.untie_last then
            -- Share weights on all but the last frame.
            if step ~= input_length then
                new_stub = stub:sharedClone()
            else
                new_stub = stub:clone()
            end
        elseif weight_type == WEIGHT_TYPE_OPTIONS.residual_untie_last then
            -- Share weights on all but the last frame.
            if step ~= input_length then
                new_stub = reset_stub:sharedClone()
            else
                new_stub = stub:clone()
            end
        end
        parallel_stubs:add(new_stub)
    end
    local merger = create_merger(
        input_length, MERGE_INPUT_LENGTH, merge_type, args.weighted_merge)
    table.insert(model_stubs, parallel_stubs)
    table.insert(model_stubs, merger)
    previous_layer_index = layer_index
end

-- Last layer -> output
local stub_to_output = extract_stub(
    model, MERGE_LAYER_INDICES[#MERGE_LAYER_INDICES] + 1, model:size())

local output_model = nn.Sequential()
output_model:add(nn.SplitTable(1))
for _, model_stub in ipairs(model_stubs) do
    output_model:add(model_stub)
end
-- At this point we should have a table with exactly one element, which is the
-- tensor of merged conv5 activations. Use JoinTable to convert it to a
-- tensor. (We could also use SelectTable, but JoinTable will cause errors if
-- the table somehow has more than one elements, which is nice.)
output_model:add(nn.JoinTable(1))
output_model:add(stub_to_output)
-- Output should be (1, batch_size, num_labels) to be compatible with
-- DataParallelTable.
output_model:add(nn.Unsqueeze(1))
torch.save(args.output, output_model)
print('Model saved to', args.output)
