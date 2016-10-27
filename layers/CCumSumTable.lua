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
    parent.__init(self)
    self.reinitialize_rate = reinitialize_rate
end

function CCumSumTable:_create_sum(start)
    local modules = {nn.SelectTable(start)}
    for i = start + 1, start + self.reinitialize_rate - 1 do
        local sum = nn.Sequential()
        -- Select x_{t-1}, x_t.
        sum:add(nn.NarrowTable(start, i))
        -- Compute x_{t-1} - x_t, then multiply by -1.
        sum:add(nn.CAddTable())
        table.insert(modules, sum)
    end
    return modules
end

function CCumSumTable:_add_module(i)
    if (i - 1) % self.reinitialize_rate and not self.modules[i] then
        local modules = self:_create_sum(i)
        for j, module in ipairs(modules) do
            self.modules[i + j - 1] = modules[j]
        end
    else
        assert(self.modules[i] ~= nil)
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
