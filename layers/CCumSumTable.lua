--[[ Computes cumulative sum of input table.

input:  {x_1, x_2, x_3, x_4, x_5, x_6}, reinitialize_rate: 3
output: {y_1, x_1 + x_2, x_1 + x_2 + x_3, x_4, x_5 - x_4, x_6 - x_5}
                                          ^-- reinitialized.
]]

local nn = require 'nn'
local torch = require 'torch'
require 'layers/ConcatTableFunctionalReinit'

local CCumSumTable, parent = torch.class('nn.CCumSumTable',
                                         'nn.ConcatTableFunctionalReinit')

function CCumSumTable:_add_reinit(i)
    self.modules[i] = nn.SelectTable(i)
end

function CCumSumTable:_add_update(i)
    local sum = nn.Sequential()
    local last_reinit = self:_last_reinit(i)
    -- Select x_{last reinit} to x_t.
    sum:add(nn.NarrowTable(last_reinit, i - last_reinit + 1))
    sum:add(nn.CAddTable())
    self.modules[i] = sum
end

function CCumSumTable:updateOutput(input)
    return parent.updateOutput(self, input)
end

function CCumSumTable:__tostring__()
    local str = torch.type(self)
    str = str .. ' { reinitialize_rate ' .. self.reinitialize_rate .. ' }'
    return str
end

function CCumSumTable:clearState()
   for i = 1, #self.modules do
      self.modules[i] = nil
   end
   self:_update(self.reinitialize_rate)
   parent.clearState(self)
end
