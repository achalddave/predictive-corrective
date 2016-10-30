--[[ Extract paths from pyramid model that merges at conv43, conv53. ]]--

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
    description = 'Extract paths from a pyramid model.'
}
parser:option('--model', 'Torch model'):count(1)
parser:option('--output', 'Output model'):count(1)
parser:option('--path', 'Which path to extract'):count(1):convert(tonumber)

local args = parser:parse()

-- Ensure that tonumber(path) is in {1, 2, 3, 4}
assert(args.path >= 1 and args.path <= 4)

nn.DataParallelTable.deserializeNGPUs = 1

print('Loading model.')
local model = torch.load(args.model)
print('Loaded model.')

if torch.isTypeOf(model, 'nn.DataParallelTable') then
    model = model:get(1)
end

local function replace_layer(container, old_layer, new_layer)
    local replaced = false
    for i = 1, #(container.modules) do
        if container.modules[i] == old_layer then
            -- Make sure we're not in some strange scenario where the same layer
            -- exists in the model twice.
            assert(not replaced)
            container.modules[i] = new_layer
            replaced = true
        end
    end
end

local function remove_layer(container, layer_remove)
    for i = 1, #(container.modules) do
        if container.modules[i] == layer_remove then
            container:remove(i)
            break;
        end
    end
end

model:remove(1)  -- Remove nn.SplitTable
model:remove()  -- Remove nn.Unsqueeze
-- parallelSelectors[3] says that for path 3, select the 3rd element of the
-- first parallelTable and the 2nd element of the second parallelTable.
local parallel_table_selectors_map = {{1, 1}, {2, 1}, {3, 2}, {4, 2}}

local parallel_table_selector = parallel_table_selectors_map[args.path]
local parallel_table, parallel_table_containers = model:findModules(
    'nn.ParallelTable')
replace_layer(parallel_table_containers[1],
              parallel_table[1],
              parallel_table[1]:get(parallel_table_selector[1]))
replace_layer(parallel_table_containers[2],
              parallel_table[2],
              parallel_table[2]:get(parallel_table_selector[2]))

local concat_tables, concat_table_containers = model:findModules(
    'nn.ConcatTable')
remove_layer(concat_table_containers[1], concat_tables[1])
remove_layer(concat_table_containers[2], concat_tables[2])

local join_tables, join_table_containers = model:findModules(
    'nn.JoinTable')
remove_layer(join_table_containers[1], join_tables[1])

-- Remove empty sequentials.
local sequential, sequential_containers = model:findModules('nn.Sequential')
for i, layer in ipairs(sequential) do
    if layer:size() == 0 then
        remove_layer(sequential_containers[i], layer)
    end
end

torch.save(args.output, model)
