--[[
-- Module for a recurrent Predictive-Corrective block.
--
-- The main export of this module is nn.PredictiveCorrectiveRecurrent
--
-- Defines the following helper modules:
--     - nn.InputCouplerRecurrent: Used for nn.CRollingDiffRecurrent
--     - nn.CRollingDiffRecurrent
--     - nn.CCumSumRecurrent
--]]
local nn = require 'nn'
local torch = require 'torch'
local __ = require 'moses'
require 'rnn'

local InitUpdateRecurrent = require 'layers/InitUpdateRecurrent'

local InputCouplerRecurrent, parent = torch.class(
    'nn.InputCouplerRecurrent', 'nn.Recurrence')

function InputCouplerRecurrent:__init(reinitialize_rate, rho)
    --[[
    -- Output rolling couples of inputs:
    --     output(t) = {input(t), input(t-1)}
    -- For the first time step, and on every reinitialization (which happens at
    -- the reinitialize_rate), we output
    --     output(t) = {input(t), zeroTensor}
    --
    -- This is used to implement the differencer.
    --
    -- The hidden state at time t is {input(t-1), input(t-2)}.
    --
    -- The main reason this is a separate class is so we can override
    -- getHiddenState so that the output size does not need to be specified, as
    -- with nn.Recurrence (we know the output size as a function of the input
    -- size a priori - it's simply output size = {input size, input size})
    --]]
    nn.AbstractRecurrent.__init(self, rho or 9999)

    self.reinitialize_rate = reinitialize_rate or math.huge
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

function InputCouplerRecurrent:_is_reinit(step)
    -- Apparently (0 % math.huge) is not == 0, so we explicitly check for the
    -- first step.
    return step == 1 or (step - 1) % self.reinitialize_rate == 0
end

function InputCouplerRecurrent:getHiddenState(step, input)
    local prevOutput
    if self:_is_reinit(step+1) then
        if input then
            self.zeroTensor = self:recursiveResizeZero(
                self.zeroTensor, {input:size(), input:size()})
        end
        prevOutput = self.userPrevOutput or self.zeroTensor
    else
        -- previous output of this module
        prevOutput = self.outputs[step]
    end
    return {prevOutput, nn.Container.getHiddenState(self, step)}
end

function InputCouplerRecurrent:_updateGradInput(input, gradOutput)
    --[[
    -- Args:
    --     input
    --     gradOutput: Table containing two elements of same size as input.
    --]]
    assert(self.updateGradInputStep >= self.step - self.rho,
           string.format('Called backward more than rho+1=%d times',
                         self.rho+1))
    assert(self.step > 1, "expecting at least one updateOutput")
    local step = self.updateGradInputStep - 1
    assert(step >= 1)

    -- set the output/gradOutput states of current Module
    local recurrentModule = self:getStepModule(step)

    -- The output of this step was fed as input to the next step; get the
    -- gradient from the next step and add it to gradOutput (which is the
    -- gradient from the current step).
    local _gradOutput = self:getGradHiddenState(step)[1]
    self._gradOutputs[step] = nn.rnn.recursiveCopy(self._gradOutputs[step], _gradOutput)
    nn.rnn.recursiveAdd(self._gradOutputs[step], gradOutput)
    gradOutput = self._gradOutputs[step]

    local gradInputTable = recurrentModule:updateGradInput(
        {input, self:getHiddenState(step-1)[1]}, gradOutput)

    -- Set the gradient for the previous step (whose output was provided as
    -- input to the current step). If our current step is a re-initialization
    -- step, then zero out the gradient for the previous step (since it was not
    -- used as input to the current step).
    local previous_step_grad = __.slice(gradInputTable, 2, #gradInputTable)
    if self:_is_reinit(step) then
        if not self.zeroGradTensor then
            self.zeroGradTensor = nn.rnn.recursiveResizeAs(
                self.zeroGradTensor, previous_step_grad)
            nn.rnn.recursiveFill(self.zeroGradTensor, 0)
        end
        self:setGradHiddenState(step - 1, self.zeroGradTensor)
    else
        self:setGradHiddenState(step - 1, previous_step_grad)
    end

    return gradInputTable[1]
end

local CRollingDiffRecurrent, parent = torch.class(
    'nn.CRollingDiffRecurrent', 'nn.Sequential')
function CRollingDiffRecurrent:__init(reinitialize_rate, rho)
    --[[
    -- Computes difference between consecutive inputs:
    --     output(t) = input(t) - input(t-1)
    --
    -- Use InputCouplerRecurrent, and produce successive differences between
    -- pairs.  InputCouplerRecurrent outputs {input(t), input(t-1)}, which is
    -- then passed through a CSubTable() in this module.  Note that the
    -- re-initialization is naturally handled by InputCouplerRecurrent, which
    -- will ensure that input(t-1) is a zero tensor upon re-initialization.
    --]]
    parent.__init(self)
    self.modules = {
        nn.InputCouplerRecurrent(reinitialize_rate, rho),
        nn.CSubTable()
    }
end

local CCumSumRecurrent, parent = torch.class(
    'nn.CCumSumRecurrent', 'nn.Recurrence')
function CCumSumRecurrent:__init(reinitialize_rate, rho)
    --[[
    -- Output componentwise cumulative sum of inputs:
    --     output(t) = input(t) + output(t-1)
    --
    -- The main reason this is a separate class is so we can override
    -- getHiddenState so that the output size does not need to be specified, as
    -- with nn.Recurrence (we know the output size is exactly the input size).
    --]]
    nn.AbstractRecurrent.__init(self, rho or 9999)

    self.reinitialize_rate = reinitialize_rate or math.huge

    self.recurrentModule = nn.CAddTable()

    self.module = self.recurrentModule
    self.modules[1] = self.recurrentModule
    self.sharedClones[1] = self.recurrentModule

    self.typeTensor = torch.Tensor()
end

function CCumSumRecurrent:_is_reinit(step)
    -- Apparently (0 % math.huge) is not == 0, so we explicitly check for the
    -- first step.
    return step == 1 or (step - 1) % self.reinitialize_rate == 0
end

function CCumSumRecurrent:getHiddenState(step, input)
    local prevOutput
    if self:_is_reinit(step+1) then
        if input then
            self.zeroTensor = self:recursiveResizeZero(
                self.zeroTensor, input:size())
        end
        prevOutput = self.userPrevOutput or self.zeroTensor
    else
        -- previous output of this module
        prevOutput = self.outputs[step]
    end
    return {prevOutput, nn.Container.getHiddenState(self, step)}
end

function CCumSumRecurrent:_updateGradInput(input, gradOutput)
    return InputCouplerRecurrent._updateGradInput(self, input, gradOutput)
end

local PredictiveCorrectiveRecurrent, parent = torch.class(
    'nn.PredictiveCorrectiveRecurrent', 'nn.Sequential')

function PredictiveCorrectiveRecurrent:__init(
        init, update, reinitialize_rate, rho)
    parent.__init(self)

    self.modules = {
        nn.CRollingDiffRecurrent(reinitialize_rate, rho),
        nn.InitUpdateRecurrent(init, update, reinitialize_rate, rho),
        nn.CCumSumRecurrent(reinitialize_rate, rho)
    }
end
