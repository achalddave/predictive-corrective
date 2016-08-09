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
            replacement_layer = nn.Recurrent(
                nn.Identity() --[[start]],
                old_layer --[[input layer]],
                nn.Linear(output_size, output_size) --[[feedback]],
                nn.Identity() --[[transfer]],
                nn.CAddTable() --[[merge]])
        elseif args.hidden == 'avg' then
            local averaging_layer = nn.Sequential()
            averaging_layer:add(nn.CAvgTable())

            replacement_layer = nn.Recurrent(
                nn.Identity() --[[start]],
                old_layer --[[input layer]],
                nn.Identity() --[[feedback]],
                nn.Identity() --[[transfer]],
                1 --[[rho]],
                averaging_layer --[[merge]])
        elseif args.hidden == 'poolavg' then
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
        old_layer_container.modules[i] = replacement_layer
    end
end
print(model)

torch.save(args.output, model)
