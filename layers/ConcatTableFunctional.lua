local nn = require 'nn'
local torch = require 'torch'

local ConcatTableFunctional, parent = torch.class('nn.ConcatTableFunctional',
                                                  'nn.ConcatTable')

function ConcatTableFunctional:__init()
    parent.__init(self)
end

function ConcatTableFunctional:_add_module(i)
    assert(false, 'Should be implemented by child class.')
end

function ConcatTableFunctional:_update(num_input)
    if num_input < #self.modules then
        for i = num_input + 1, #self.modules do
            self.modules[i] = nil
        end
    elseif num_input > #self.modules then
        for i = 1, num_input do
            if not self.modules[i] then
                self:_add_module(i)
            end
        end
        self:type(self._type)
        if self._training then self:training() else self:evaluate() end
    end
end

function ConcatTableFunctional:updateOutput(input)
    self.output = {}
    self:_update(#input)
    return parent.updateOutput(self, input)
end

function ConcatTableFunctional:updateGradInput(input, gradOutput)
    self:_update(#input)
    return parent.updateGradInput(self, input, gradOutput)
end

function ConcatTableFunctional:accGradParameters(input, gradOutput, scale)
    self:_update(#input)
    return parent.accGradParameters(self, input, gradOutput, scale)
end

function ConcatTableFunctional:accUpdateGradParameters(input, gradOutput, lr)
    self:_update(#input)
    return parent.accUpdateGradParameters(self, input, gradOutput, lr)
end

function ConcatTableFunctional:type(type, tensorCache)
    self._type = type
    return parent.type(self, type, tensorCache)
end

function ConcatTableFunctional:training()
    parent.training(self)
    self._training = true
end

function ConcatTableFunctional:evaluate()
    parent.evaluate(self)
    self._training = false
end
