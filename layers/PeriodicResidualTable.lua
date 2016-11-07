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
    self.reinitialize_rate = reinitialize_rate
    self.init = init
    self.residual = residual
    parent.__init(self)

    self:_update(self.reinitialize_rate)
end

function PeriodicResidualTable:_add_module(i)
    local module = nn.Sequential()
    module:add(nn.SelectTable(i))
    -- self.init and self.residual have to be in self.modules so that any call
    -- to self.modules will also be made to self.init and self.residual (e.g.
    -- model:training()).
    if i == 1 then
        module:add(self.init)
    elseif i == 2 then
        module:add(self.residual)
    elseif (i - 1) % self.reinitialize_rate == 0 then
        module:add(self.init:sharedClone())
    else
        module:add(self.residual:sharedClone())
    end
    table.insert(self.modules, module)
end

function PeriodicResidualTable:read(file, versionNumber)
    parent.read(self, file, versionNumber)
    if #self.modules == 0 or self.modules[1]:get(2) ~= self.init then
        -- Model was created with old code that didn't set the first and second
        -- modules to be equivalently self.init. Reset the self.modules array.
        print('PeriodicResidualTable: Repopulating self.modules for old ' ..
              'model; this is for backwards compatibility and can be ignored.')
        self.modules = {}
        self:_update(self.reinitialize_rate)
    end
end

function PeriodicResidualTable:__tostring__()
    local str = torch.type(self)
    str = str .. ' { reinitialize_rate ' .. self.reinitialize_rate .. ' }'
    return str
end

function PeriodicResidualTable:clearState()
   for i = 1, #self.modules do
      self.modules[i] = nil
   end
   self.init:clearState()
   self.residual:clearState()
   parent.clearState(self)
end
