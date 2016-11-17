--[[ Computes rolling difference of input table.

input:  {x_1, x_2, x_3, x_4, x_5, x_6}, reinitialize_rate: 3
output: {x_1, x_2 - x_1, x_3 - x_2, x_4, x_5 - x_4, x_6 - x_5}
                                    ^-- reinitialized.
]]

local nn = require 'nn'
local torch = require 'torch'
require 'layers/ConcatTableFunctionalReinit'

local CRollingDiffTable, parent = torch.class('nn.CRollingDiffTable',
                                              'nn.ConcatTableFunctionalReinit')

function CRollingDiffTable:_add_reinit(i)
    self.modules[i] = nn.SelectTable(i)
end

function CRollingDiffTable:_add_update(i)
    local differencer = nn.Sequential()
    -- Select x_{t-1}, x_t.
    differencer:add(nn.NarrowTable(i - 1, 2))
    -- Compute x_{t-1} - x_t, then multiply by -1.
    differencer:add(nn.CSubTable())
    differencer:add(nn.MulConstant(-1))
    self.modules[i] = differencer
end

function CRollingDiffTable:updateOutput(input)
    return parent.updateOutput(self, input)
end

function CRollingDiffTable:__tostring__()
    local str = torch.type(self)
    str = str .. ' { reinitialize_rate ' .. self.reinitialize_rate .. ' }'
    return str
end

function CRollingDiffTable:clearState()
   for i = 1, #self.modules do
      self.modules[i] = nil
   end
   parent.clearState(self)
end
