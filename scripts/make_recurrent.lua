--[[
Turn a network into a recurrent network by connecting a layer over time.
]]--

package.path = package.path .. ";layers/?.lua"

local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local torch = require 'torch'
local nn = require 'nn'
require 'cunn'
require 'rnn'

require 'CAvgTable'

local parser = argparse() {
    description = 'Replace last fully connected layer with a recurrent layer.'
}
parser:option('--model', 'Torch model'):count(1)
parser:option('--layer_type',
                'Layer type to output activations from. ' ..
                'E.g. cudnn.SpatialConvolution'):count(1)
parser:option('--layer_type_index',
                'Which of the layer types to extract.'):count(1)
      :convert(tonumber)
      :count(1)
parser:option('--hidden',
    'Type of hidden connection. Options: \n' ..
    'linear:  Fully connected layer between hidden weights. \n' ..
    'avg: Average current activations with previous hidden activation. \n' ..
    'avg_ends: Average first and last time step activations. \n' ..
    'poolavg: Average current activations with pooled previous activation. \n')
    :default('linear')
parser:option('--output', 'Output rnn model'):count(1)

local args = parser:parse()

local model = torch.load(args.model)
if torch.isTypeOf(model, 'nn.DataParallelTable') then
    model = model:get(1)
end

local layers, containers = model:findModules(args.layer_type)
-- Use old_layer as input to a recurrent layer, then replace the old_layer in
-- the model with the recurrent layer.
local old_layer = layers[args.layer_type_index]
local old_layer_container = containers[args.layer_type_index]

for i = 1, #(old_layer_container.modules) do
    if old_layer_container.modules[i] == old_layer then
        local replacement_layer
        if args.hidden == 'linear' then
            local output_size = old_layer.weight:size(1)
            local recurrent_layer = nn.Recurrent(
                nn.Identity() --[[start]],
                nn.Identity() --[[input layer]],
                nn.Linear(output_size, output_size) --[[feedback]],
                nn.Identity() --[[transfer]],
                nn.CAddTable() --[[merge]])
            old_layer_container:insert(recurrent_layer, i + 1)
        elseif args.hidden == 'avg' or args.hidden == 'avg_ends' then
            -- Convert input tensor to singleton table.
            local tensor_to_table = nn.ConcatTable()
            tensor_to_table:add(nn.Identity())

            local copy_layer = nn.Copy(nil, nil, true --[[forceCopy]])
            local collect_recurrent_to_table = nn.Recurrent(
                tensor_to_table, --[[start: Convert first output to table]]
                copy_layer --[[input]],
                nn.Identity() --[[feedback]],
                nn.FlattenTable() --[[transfer]],
                1 --[[rho; overridden by Sequencer]],
                nn.Identity() --[[merge]])

            old_layer_container:insert(collect_recurrent_to_table, i + 1)
            if args.hidden == 'avg_ends' then
                local first_last_selector = nn.ConcatTable()
                first_last_selector:add(nn.SelectTable(1))
                first_last_selector:add(nn.SelectTable(-1))
                old_layer_container:insert(first_last_selector, i + 2)
                old_layer_container:insert(nn.CAvgTable(), i + 3)
            else
                old_layer_container:insert(nn.CAvgTable(), i + 2)
            end
        elseif args.hidden == 'poolavg' then
            -- Pooling layer averages after each new input; we would like to sum
            -- after each new input, then divide by #inputs at the end. I don't
            -- know how to easily do this.
            error('Currently broken.')
            local pool_layer = nn.SpatialMaxPooling(
                3 --[[ kernel_width ]],
                3 --[[ kernel_height ]],
                1 --[[ stride_width ]],
                1 --[[ stride_height ]],
                1 --[[ pad_width ]],
                1 --[[ pad_height ]]
            )
            local averaging_layer = nn.Sequential()
            averaging_layer:add(nn.CAvgTable())

            replacement_layer = nn.Recurrent(
                nn.Identity() --[[start]],
                old_layer --[[input layer]],
                pool_layer --[[feedback]],
                nn.Identity() --[[transfer]],
                1 --[[rho]],
                averaging_layer --[[merge]])
        else
            error('Unknown --hidden option:' .. args.hidden .. '.' ..
                  'See --help for valid options')
        end
    end
end
print(model)

torch.save(args.output, model)
