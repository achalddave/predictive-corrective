--[[ Computes rolling difference of input table.

input:  {x_1, x_2, x_3, x_4, x_5, x_6}, reinitialize_rate: 3
output: {x_1, x_2 - x_1, x_3 - x_2, x_4, x_5 - x_4, x_6 - x_5}
                                    ^-- reinitialized.
]]

local nn = require 'nn'
local torch = require 'torch'

local CRollingDiffTable, parent = torch.class('nn.CRollingDiffTable',
                                              'nn.ConcatTable')

function CRollingDiffTable:__init(reinitialize_rate)
    parent.__init(self)
    self.reinitialize_rate = reinitialize_rate
    self.module = self:_update(self.reinitialize_rate)
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

function CRollingDiffTable:_add_differencer()
    local modules = self:_create_differencer(#self.modules + 1)
    for _, module in ipairs(modules) do
        table.insert(self.modules, module)
    end
end

function CRollingDiffTable:_update(num_input)
    if num_input < #self.modules then
        for i = num_input + 1, #self.modules do
            self.modules[i] = nil
        end
    elseif num_input > #self.modules then
        for i = 1, num_input, self.reinitialize_rate do
            if not self.modules[i] then
                self:_add_differencer()
            end
        end
    end
end

function CRollingDiffTable:updateOutput(input)
    assert(#input % self.reinitialize_rate == 0,
           string.format('Input length (%d) should be a multiple of '..
                         'reinitialize rate (%d).',
                         #input, self.reinitialize_rate))
    self.output = {}
    self:_update(#input)
    return parent.updateOutput(self, input)
end

function CRollingDiffTable:updateGradInput(input, gradOutput)
    self:_update(#input)
    return parent.updateGradInput(self, input, gradOutput)
end

function CRollingDiffTable:accGradParameters(input, gradOutput, scale)
    self:_update(#input)
    return parent.accGradParameters(self, input, gradOutput, scale)
end

function CRollingDiffTable:accUpdateGradParameters(input, gradOutput, lr)
    self:_update(#input)
    return parent.accUpdateGradParameters(self, input, gradOutput, lr)
end

function CRollingDiffTable:zeroGradParameters()
    parent.zeroGradParameters(self)
end

function CRollingDiffTable:updateParameters(learningRate)
    parent.updateParameters(self, learningRate)
end

function CRollingDiffTable:clearState()
    parent.clearState(self)
end

function CRollingDiffTable:__tostring__()
    local str = torch.type(self)
    str = str .. ' { reinitialize_rate ' .. self.reinitialize_rate .. ' }'
    return str
end
