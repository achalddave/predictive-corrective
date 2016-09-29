--[[ Flip a pyramid model temporally.
--
-- Let f be the original network, and g the flipped network. Then,
--
-- g(x_1, x_2, x_3, x_4) = f(x_4, x_3, x_2, x_1)
--
-- where the x_i are images.
--]]

local argparse = require 'argparse'
local torch = require 'torch'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'nn'
require 'rnn'

package.path = package.path .. ";layers/?.lua"

local parser = argparse() {
    description = 'Flip a pyramid model temporally.'
}
parser:option('--model', 'Input model'):count(1)
parser:option('--output', 'Output model'):count(1)

local args = parser:parse()

local model = torch.load(args.model)
local original_model = model:clone()

local parallel_tables = model:findModules('nn.ParallelTable')
for _, parallel_table in ipairs(parallel_tables) do
    local num_modules = #parallel_table.modules
    local flipped_modules = {}
    for m = 1, num_modules do
        flipped_modules[num_modules - m + 1] = parallel_table.modules[m]
    end
    parallel_table.modules = flipped_modules
end

torch.save(args.output, model)
