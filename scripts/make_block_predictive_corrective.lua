--[[
-- Convert a predictive-corrective network to use PredictiveCorrectiveBlock.
--
-- Predictive-corrective networks created by make_predictive_corrective.lua
-- contain nn.Sequential blocks with the following layers:
--    {CRollingDiffTable, PeriodicResidualTable, CCumSumTable}
--
-- This script replaces that Sequential block with an PredictiveCorrectiveBlock
-- module, which can handle dynamic re-initializion and discarding of frames.
--
-- The PredictiveCorrectiveBlock is fairly hacky, and creates and destroys
-- modules on the fly, so we do not use it by default. However, for the dynamic
-- initialization and frame discarding experiments, it's necessary to use this
-- block.
--]]

package.path = package.path .. ";layers/?.lua"

local argparse = require 'argparse'
local torch = require 'torch'
local nn = require 'nn'
require 'rnn'
require 'cudnn'
require 'cunn'
require 'cutorch'

local log = require 'util/log'
require 'layers/init'

local parser = argparse() {
    description = 'Convert network to use PredictiveCorrectiveBlock module.'
}
parser:option('--model', 'Predictive-corrective model'):count(1)
parser:option('--output', 'Output model using PredictiveCorrectiveBlock')
      :count(1)

local args = parser:parse()
log.info('Args:')
log.info(args)

local model = torch.load(args.model)
local sequential_layers = model:findModules('nn.Sequential')

for _, layer in ipairs(sequential_layers) do
    if #layer.modules == 3 and
            torch.isTypeOf(layer.modules[1], nn.CRollingDiffTable) and
            torch.isTypeOf(layer.modules[2], nn.PeriodicResidualTable) and
            torch.isTypeOf(layer.modules[3], nn.CCumSumTable) then
        local differ, periodic_residual, summer = unpack(layer.modules)
        local reinitialize_rate = differ.reinitialize_rate
        assert(reinitialize_rate == periodic_residual.reinitialize_rate)
        assert(reinitialize_rate == summer.reinitialize_rate)

        layer:remove()
        layer:remove()
        layer:remove()
        layer:add(nn.PredictiveCorrectiveBlock(periodic_residual.init,
                                               periodic_residual.residual,
                                               math.huge --[[init_threshold]],
                                               reinitialize_rate,
                                               -1 --[[ignore_threshold]]))
    end
end

torch.save(args.output, model)
