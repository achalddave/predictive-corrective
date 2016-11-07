local nn = require 'nn'
local torch = require 'torch'
require 'layers/ConcatTableFunctional'

local ConcatTableFunctionalReinit, parent = torch.class(
    'nn.ConcatTableFunctionalReinit',
    'nn.ConcatTableFunctional')

function ConcatTableFunctionalReinit:__init(reinitialize_rate)
    self.reinitialize_rate = reinitialize_rate

    parent.__init(self)
    self:_update(self.reinitialize_rate)
end

function ConcatTableFunctionalReinit:_add_module(i)
    if self:_is_reinit(i) then
        self:_add_reinit(i)
    else
        self:_add_update(i)
    end
end

function ConcatTableFunctionalReinit:_is_reinit(i)
    return (i - 1) % self.reinitialize_rate == 0
end

function ConcatTableFunctionalReinit:_add_reinit(_)
    error('_add_reinit should be implemented by child class.')
end

function ConcatTableFunctionalReinit:_add_update(_)
    error('_add_update should be implemented by child class.')
end

function ConcatTableFunctionalReinit:_last_reinit(i)
    -- When did we last reinitialize before the ith input?
    return (math.floor((i - 1) / self.reinitialize_rate)
                * self.reinitialize_rate + 1)
end
