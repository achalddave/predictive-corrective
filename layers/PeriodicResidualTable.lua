--[[
--
-- Input: X = {x1, ..., xn}, reinitialize_rate: 3
-- Output: {f_1(X), f_2(X), f_2(X), f_1(X), f_2(X), f_2(X), ...}
--                                  ^-- reinitialized.
--]]

local nn = require 'nn'
local torch = require 'torch'
require 'layers/ConcatTableFunctionalReinit'
require 'dpnn'  -- for sharedClone()

local PeriodicResidualTable, parent = torch.class(
    'nn.PeriodicResidualTable', 'nn.ConcatTableFunctionalReinit')

function PeriodicResidualTable:__init(reinitialize_rate, init, residual)
    self.init = init
    self.residual = residual
    parent.__init(self, reinitialize_rate)
end

function PeriodicResidualTable:_add_reinit(i)
    local module = nn.Sequential()
    module:add(nn.SelectTable(i))
    -- self.init and self.residual have to be in self.modules so that any call
    -- to self.modules will also be made to self.init and self.residual (e.g.
    -- model:training()).
    if i == 1 then
        module:add(self.init)
    else
        module:add(self.init:sharedClone())
    end
    self.modules[i] = module
end

function PeriodicResidualTable:_add_update(i)
    local module = nn.Sequential()
    module:add(nn.SelectTable(i))
    if i == 2 then
        module:add(self.residual)
    else
        module:add(self.residual:sharedClone())
    end
    self.modules[i] = module
end

function PeriodicResidualTable:read(file, versionNumber)
    parent.read(self, file, versionNumber)
    if #self.modules == 0 or self.modules[1]:get(2) ~= self.init then
        -- Model was created with old code that didn't set the first and second
        -- modules to be self.init and self.residual. Reset the self.modules
        -- array.
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
   -- Keep all but one init and residual clone in self.modules
   for i = 1, #self.modules do
      self.modules[i]:clearState()
      if i > 3 then self.modules[i] = nil end
   end
   parent.clearState(self)
end
