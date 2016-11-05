--[[
--
-- Input: X = {x1, ..., xn}, reinitialize_rate: 3
-- Output: {f_1(X), f_2(X), f_2(X), f_1(X), f_2(X), f_2(X), ...}
--                                  ^-- reinitialized.
--]]

local nn = require 'nn'
local torch = require 'torch'
require 'layers/ConcatTableFunctional'
require 'dpnn'  -- for sharedClone()

local PeriodicResidualTable, parent = torch.class('nn.PeriodicResidualTable',
                                                  'nn.ConcatTableFunctional')

function PeriodicResidualTable:__init(reinitialize_rate, init, residual)
    parent.__init(self)
    self.reinitialize_rate = reinitialize_rate
    self.init = init
    self.residual = residual
    self:_update(self.reinitialize_rate)
end

function PeriodicResidualTable:_add_module(i)
    local module = nn.Sequential()
    module:add(nn.SelectTable(i))
    if (i - 1) % self.reinitialize_rate == 0 then
        module:add(self.init:sharedClone())
    else
        module:add(self.residual:sharedClone())
    end
    table.insert(self.modules, module)
end

function PeriodicResidualTable:__tostring__()
    local str = torch.type(self)
    str = str .. ' { reinitialize_rate ' .. self.reinitialize_rate .. ' }'
    return str
end
