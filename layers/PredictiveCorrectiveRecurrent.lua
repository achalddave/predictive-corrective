local nn = require 'nn'
local rnn = require 'rnn'
local torch = require 'torch'

local InitUpdateRecurrent = require 'layers/InitUpdateRecurrent'

local InputCouplerRecurrent, parent = torch.class(
    'nn.InputCouplerRecurrent', 'nn.Recurrence')

function InputCouplerRecurrent:__init(rho)
    --[[
    -- Output rolling couples of inputs:
    --     output(t) = {input(t), input(t-1)}
    --
    -- Therefore, the hidden state at time t is {input(t-1), input(t-2)}.
    --
    -- The main reason this is a separate class is so we can override
    -- getHiddenState so that the output size does not need to be specified, as
    -- with nn.Recurrence (we know the output size as a function of the input
    -- size a priori - it's simply output size = {input size, input size})
    --]]
    nn.AbstractRecurrent.__init(self, rho or 9999)

    -- Given input {a, {b, c}}, outputs {a, b}.
    -- In our case, a, b, c are the inputs at time t, t-1, and t-2 respectively.
    local pair_selector = nn.ConcatTable()
    pair_selector:add(nn.SelectTable(1)) -- Select a
    pair_selector:add(nn.Sequential()
        :add(nn.SelectTable(2)) -- Select {b, c}
        :add(nn.SelectTable(1))) -- Select b
    self.recurrentModule = pair_selector

    self.module = self.recurrentModule
    self.modules[1] = self.recurrentModule
    self.sharedClones[1] = self.recurrentModule

    self.typeTensor = torch.Tensor()
end

function InputCouplerRecurrent:getHiddenState(step, input)
    local prevOutput
    if step == 0 then
        if input then
            self.zeroTensor = self:recursiveResizeZero(
                self.zeroTensor, {input:size(), input:size()})
        end
        prevOutput = self.userPrevOutput or self.outputs[step] or self.zeroTensor
    else
        -- previous output of this module
        prevOutput = self.outputs[step]
    end
    -- call getHiddenState on recurrentModule as they may contain
    -- AbstractRecurrent instances...
    return {prevOutput, nn.Container.getHiddenState(self, step)}
end

local CRollingDiffRecurrent, parent = torch.class(
    'nn.CRollingDiffRecurrent', 'nn.Sequential')
function CRollingDiffRecurrent:__init(rho)
    --[[
    -- Computes difference between consecutive inputs:
    --     output(t) = input(t) - input(t-1)
    --]]
    parent.__init(self)
    self.modules = {nn.InputCouplerRecurrent(rho), nn.CSubTable()}
end

local CCumSumRecurrent, parent = torch.class(
    'nn.CCumSumRecurrent', 'nn.Recurrence')
function CCumSumRecurrent:__init(rho)
    --[[
    -- Output componentwise cumulative sum of inputs:
    --     output(t) = input(t) + output(t-1)
    --
    -- The main reason this is a separate class is so we can override
    -- getHiddenState so that the output size does not need to be specified, as
    -- with nn.Recurrence (we know the output size is exactly the input size).
    --]]
    nn.AbstractRecurrent.__init(self, rho or 9999)

    self.recurrentModule = nn.CAddTable()

    self.module = self.recurrentModule
    self.modules[1] = self.recurrentModule
    self.sharedClones[1] = self.recurrentModule

    self.typeTensor = torch.Tensor()
end

function CCumSumRecurrent:getHiddenState(step, input)
    local prevOutput
    if step == 0 then
        if input then
            self.zeroTensor = self:recursiveResizeZero(
                self.zeroTensor, input:size())
        end
        prevOutput = self.userPrevOutput or self.outputs[step] or self.zeroTensor
    else
        -- previous output of this module
        prevOutput = self.outputs[step]
    end
    -- call getHiddenState on recurrentModule as they may contain
    -- AbstractRecurrent instances...
    return {prevOutput, nn.Container.getHiddenState(self, step)}
end

local PredictiveCorrectiveRecurrent, parent = torch.class(
    'nn.PredictiveCorrectiveRecurrent', 'nn.Sequential')

function PredictiveCorrectiveRecurrent:__init(init, update, rho)
    parent.__init(self)

    self.modules = {
        nn.CRollingDiffRecurrent(rho),
        nn.InitUpdateRecurrent(init, update, rho),
        nn.CCumSumRecurrent(rho)
    }
end
