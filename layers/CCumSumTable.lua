--[[ Computes cumulative sum of input table.

input:  {x_1, x_2, x_3, x_4, x_5, x_6}, reinitialize_rate: 3
output: {y_1, x_1 + x_2, x_1 + x_2 + x_3, x_4, x_5 - x_4, x_6 - x_5}
                                          ^-- reinitialized.
]]

local nn = require 'nn'
local torch = require 'torch'
require 'layers/ConcatTableFunctional'

local CCumSumTable, parent = torch.class('nn.CCumSumTable',
                                         'nn.ConcatTableFunctional')

function CCumSumTable:__init(reinitialize_rate)
    self.reinitialize_rate = reinitialize_rate
    parent.__init(self)

    self:_update(self.reinitialize_rate)
end

function CCumSumTable:_add_module(i)
    if self.modules[i] ~= nil then
        return
    end
    if (i - 1) % self.reinitialize_rate == 0 then
        self.modules[i] = nn.SelectTable(i)
    else
        local sum = nn.Sequential()
        local last_reinit = (
            math.floor((i - 1) / self.reinitialize_rate)
                * self.reinitialize_rate + 1)
        -- Select x_{last reinit} to x_t.
        sum:add(nn.NarrowTable(last_reinit, i - last_reinit + 1))
        sum:add(nn.CAddTable())
        self.modules[i] = sum
    end
end

function CCumSumTable:updateOutput(input)
    assert(#input % self.reinitialize_rate == 0,
           string.format('Input length (%d) should be a multiple of '..
                         'reinitialize rate (%d).',
                         #input, self.reinitialize_rate))
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
   parent.clearState(self)
end
