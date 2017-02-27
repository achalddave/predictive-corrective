--[[ Copy model, keeping only selected outputs from the model.
--
-- Example usage:
--   th scripts/select_model_outputs.lua \
--      --model /path/to/model_with_65_outputs.t7 \
--      --output /path/to/output_model_with_3_outputs.t7 \
--      --select 1 21 50
--
-- Note that the output model's outputs will be in the order as specified by
-- select. In the above example, the output model will produce 3 predictions,
-- with the first prediction for label 1, the second for 21, and the third for
-- 50.
--]]

package.path = package.path .. ";layers/?.lua"

local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local torch = require 'torch'
local nn = require 'nn'
require 'rnn'
require 'cunn'

local log = require 'util/log'

local parser = argparse() {
    description = 'Create a pyramid shaped network.'
}
parser:option('--model', 'Torch model'):count(1)
parser:option('--output', 'Output model'):count(1)
parser:option('--select',
    'List of 1-indexed indices of outputs to select.')
    :args('*')
    :convert(tonumber)

local args = parser:parse()

local model = torch.load(args.model):cuda()
if torch.isTypeOf(model, 'nn.DataParallelTable') then
    model = model:get(1)
end

local layers, containers = model:findModules('nn.Linear')
local old_layer = layers[#layers]
old_layer:clearState()
log.info('Updating last linear layer:')
log.info(old_layer)

local input_size = old_layer.weight:size(2)
local new_layer = nn.Linear(input_size,
                            #args.select --[[output_size]]):cuda()
for new_index, old_index in ipairs(args.select) do
    new_layer.weight[new_index] = old_layer.weight[old_index]
    if old_layer.bias ~= nil then
        new_layer.bias[new_index] = old_layer.bias[old_index]
    end
end

-- Ensure that the outputs match between the two layers.
local input = torch.rand(input_size):cuda()
local new_output = new_layer:forward(input)
local old_output = old_layer:forward(input)
print(new_output)
for new_index, old_index in ipairs(args.select) do
    assert(new_output[new_index] == old_output[old_index])
    log.info(string.format('Test passed: New output %s matches old output %s.',
        new_index, old_index))
end

for i, layer in ipairs(containers[#layers].modules) do
    if layer == old_layer then
        containers[#layers].modules[i] = new_layer
    end
end

model:clearState()
torch.save(args.output, model)
