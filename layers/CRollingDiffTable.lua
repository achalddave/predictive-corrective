--[[ Computes rolling difference of input table.

input:  {x_1, x_2, x_3, x_4, x_5, x_6}, reinitialize_rate: 3
output: {x_1, x_2 - x_1, x_3 - x_2, x_4, x_5 - x_4, x_6 - x_5}
                                    ^-- reinitialized.
]]

local nn = require 'nn'
local torch = require 'torch'
require 'layers/ConcatTableFunctional'

local CRollingDiffTable, parent = torch.class('nn.CRollingDiffTable',
                                              'nn.ConcatTableFunctional')

function CRollingDiffTable:__init(reinitialize_rate)
    parent.__init(self)
    self.reinitialize_rate = reinitialize_rate
end

function CRollingDiffTable:_create_differencer(start)
    local modules = {nn.SelectTable(start)}
    for i = start + 1, start + self.reinitialize_rate - 1 do
        local differencer = nn.Sequential()
        -- Select x_{t-1}, x_t.
        differencer:add(nn.NarrowTable(i - 1, 2))
        -- Compute x_{t-1} - x_t, then multiply by -1.
        differencer:add(nn.CSubTable())
        differencer:add(nn.MulConstant(-1))
        table.insert(modules, differencer)
    end
    return modules
end

function CRollingDiffTable:_add_module(i)
    if (i - 1) % self.reinitialize_rate and not self.modules[i] then
        local modules = self:_create_differencer(i)
        for j, module in ipairs(modules) do
            self.modules[i + j - 1] = modules[j]
        end
    else
        assert(self.modules[i] ~= nil)
    end
end

function CRollingDiffTable:updateOutput(input)
    assert(#input % self.reinitialize_rate == 0,
           string.format('Input length (%d) should be a multiple of '..
                         'reinitialize rate (%d).',
                         #input, self.reinitialize_rate))
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
