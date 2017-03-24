--[[
Create a Predictive-Corrective network.

Below is an example of a `residual block` that computes z1 from z0, with a
reinitialization rate of 2.

z10 ----------------f^1()------------> z11
z20 --> (z20-z10) --r^1()--> (+z11) -> z21

z30 ----------------f^1()------------> z31
z40 --> (z40-z30) --r^1()--> (+z31) -> z41

where f^1 is an 'initial' network and r^1 is a 'residual' network. The above
block can be recursively applied. f^1() for example can be the layers of
VGG-16 up to conv4-3, and f^2() can be the layers between conv4-3 and conv5-3.

By default, this network takes as input tensors of shape
(sequence_length, batch_size, num_channels, height, width), and outputs
(sequence_length, batch_size, num_outputs).

If '--recurrent' is specified, then the input should be a single step in the
sequence, with shape (batch_size, num_channels, height, width), and the output
will be (batch_size, num_outputs).
]]--

package.path = package.path .. ";layers/?.lua"

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
parser:option('--model', 'Torch model'):count(1)
parser:option('--output', 'Output model'):count(1)
parser:flag('--recurrent',
            'If true, create a recurrent version that maintains hidden ' ..
            'state across calls.')

local args = parser:parse()
print('Args:')
print(args)

-- conv12: 3
-- conv22: 8
-- conv33: 15
-- conv43: 22
-- conv53: 29
-- fc7: 36
-- fc8: 39
local MERGE_LAYER_INDICES = {15, 36}
-- How many consecutive inputs to merge at each merging step.
local REINITIALIZE_RATES = {1, 16} -- , 2}
assert(#REINITIALIZE_RATES == #MERGE_LAYER_INDICES)
print('Merge layer indices:', MERGE_LAYER_INDICES)
print('Reinitialize rates:', REINITIALIZE_RATES)
local model = torch.load(args.model)
model:clearState()
print('Loaded model.')
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

local previous_layer_index = 0
local model_stubs = {}
for i, layer_index in ipairs(MERGE_LAYER_INDICES) do
    local reinitialize_rate = REINITIALIZE_RATES[i]

    local stub = extract_stub(model, previous_layer_index + 1, layer_index)
    if reinitialize_rate == 1 then
        -- Adding the whole PredictiveCorrective block is expensive and
        -- unnecessary if the re-initialization rate is set to 1, anyway, so
        -- just add the initial network in this case.
        if args.recurrent then
            table.insert(model_stubs, stub)
        else
            local mapper = nn.MapTable()
            mapper:add(stub)
            table.insert(model_stubs, mapper)
        end
    else
        if args.recurrent then
            table.insert(model_stubs,
                         nn.PredictiveCorrectiveRecurrent(
                             stub --[[init]],
                             stub:clone() --[[residual]],
                             reinitialize_rate))
        else
            print(os.date('%X'), 'Creating differencer')
            local differencer = nn.CRollingDiffTable(reinitialize_rate)

            print(os.date('%X'), 'Creating residual')
            local periodic_stubs = nn.PeriodicResidualTable(
                reinitialize_rate,
                stub --[[init]],
                stub:clone() --[[residual]])

            print(os.date('%X'), 'Creating cumulative sum')
            local cumulative_sum = nn.CCumSumTable(reinitialize_rate)

            print(os.date('%X'), 'Combining pieces')
            local processing_block = nn.Sequential()
            processing_block:add(differencer)
            processing_block:add(periodic_stubs)
            processing_block:add(cumulative_sum)
            table.insert(model_stubs, processing_block)
        end
    end

    previous_layer_index = layer_index
end

-- Last layer -> output
local stub_to_output = extract_stub(
    model, MERGE_LAYER_INDICES[#MERGE_LAYER_INDICES] + 1, model:size())

local output_model = nn.Sequential()
if not args.recurrent then
    output_model:add(nn.SplitTable(1))
end
for _, model_stub in ipairs(model_stubs) do
    output_model:add(model_stub)
end
if args.recurrent then
    -- XXX HACK XXX: The recurrent model's adder layer re-uses the output at
    -- time t as input to time t + 1; this means we can't use an in-place ReLU.
    if stub_to_output:get(1).inplace == true then
        stub_to_output:get(1).inplace = false
    end
    output_model:add(stub_to_output)
else
    -- Add a singleton dimension that we can join over.
    stub_to_output:add(nn.Unsqueeze(1))
    output_model:add(nn.MapTable():add(stub_to_output))
    output_model:add(nn.JoinTable(1))
end
output_model:clearState()
collectgarbage()
collectgarbage()
torch.save(args.output, output_model)
print('Model saved to', args.output)
