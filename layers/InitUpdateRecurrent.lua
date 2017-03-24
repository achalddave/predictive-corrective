local nn = require 'nn'
local rnn = require 'rnn'
local torch = require 'torch'
local __ = require 'moses'
require 'dpnn'

local InitUpdateRecurrent, parent = torch.class(
    'nn.InitUpdateRecurrent', 'nn.Recurrent')

function InitUpdateRecurrent:__init(init, update, reinitialize_rate, rho)
    --[[
    -- Similar to nn.Recursor, but with a separate 'init' module.
    --
    -- if init
    --     output = init(input)
    -- else
    --     output = update(input)
    --
    -- This class is based off of nn.Recursor from the rnn package from
    -- element-research.
    --
    -- Args:
    --    init (nn.Module): Initialization module
    --    update (nn.Module): Update module
    --    reinitialize_rate (int): If specified, re-initialize the module at
    --        this rate (i.e. run the init module on every reinitialize_rate'th
    --        input).
    --    rho (int): See nn.Recurrence doc.
    --]]

    nn.AbstractRecurrent.__init(self, rho)

    self.reinitialize_rate = reinitialize_rate ~= nil and reinitialize_rate
                                                      or math.huge

    self.initialModule = init
    self.recurrentModule = update
    self.sharedClones = {self.initialModule, self.recurrentModule}

    self.modules = {self.initialModule, self.recurrentModule}
end

function InitUpdateRecurrent:_is_reinit(step)
    -- Apparently (0 % math.huge) is not == 0, so we explicitly check for the
    -- first step.
    return step == 1 or (step - 1) % self.reinitialize_rate == 0
end

function InitUpdateRecurrent:_last_reinit(i)
    -- When did we last reinitialize before the ith input?
    return (math.floor((i - 1) / self.reinitialize_rate)
                * self.reinitialize_rate + 1)
end

function InitUpdateRecurrent:maskZero()
    error("InitUpdateRecurrent doesn't support maskZero as it uses a " ..
          "different module for the first time-step. Use nn.Recurrence " ..
          "instead.")
end

function InitUpdateRecurrent:trimZero()
    error("InitUpdateRecurrent doesn't support trimZero as it uses a " ..
          "different module for the first time-step. Use nn.Recurrence " ..
          "instead.")
end

function InitUpdateRecurrent:getStepModule(step)
    assert(step, "expecting step at arg 1")
    local module = self.sharedClones[step]
    if not module then
        module = self:_is_reinit(step) and self.initialModule:stepClone()
                                       or self.recurrentModule:stepClone()
        self.sharedClones[step] = module
        self.nSharedClone = __.size(self.sharedClones)
    end
    return module
end

function InitUpdateRecurrent:_updateGradInput(input, gradOutput)
    assert(self.step - self.updateGradInputStep + 1 <= self.rho,
           string.format('Called backward more than rho=%d times',
                         self.rho))
    nn.Recurrent._updateGradInput(self, input, gradOutput)
end

-- We need to override :forget() so that we compact sharedClones and gradOutputs
-- in a way that keeps the initialization modules in reinit steps, and the
-- recurrent modules in update steps.
function InitUpdateRecurrent:forget()
    -- The {initial,recurrent}Modules may contain an AbstractRecurrent instance
    -- (issue 107).
    nn.Container.forget(self)

    -- Bring all states back to the start of the sequence buffers.
    if self.train ~= false then
        self.outputs = {}
        self.gradInputs = {}
        local next_init = 1
        local next_update = 2
        local sharedClones = {}
        local gradOutputs = {}
        -- Compact sharedClones and gradOutput, ensuring that we don't put a
        -- recurrentModule in an init step, or an initModule in an update step.
        for _, step in ipairs(__.sort(__.keys(self.sharedClones))) do
            local new_step
            if self:_is_reinit(step) then
                new_step = next_init
                next_init = next_init + self.reinitialize_rate
            else
                new_step = next_update
                next_update = next_update + 1
                if self:_is_reinit(next_update) then
                    next_update = next_update + 1
                end
            end
            sharedClones[new_step] = self.sharedClones[step]
            gradOutputs[new_step] = self._gradOutputs[step]
        end
        self.sharedClones = sharedClones
        self._gradOutputs = gradOutputs
        self.gradOutputs = {}
        if self.cells then
            self.cells = {}
            self.gradCells = {}
        end
    end

    -- Forget the past inputs; restart from first step.
    self.step = 1

    -- AbstractRecurrent:forget() explicitly checks if self.recurrentModule and
    -- self.initialModule are in self.sharedClones, and adds them if they are
    -- not. I don't know why this is necessary, so I'm not implementing it in
    -- case I introduce a bug. This assert will ensure the modules are in
    -- sharedClones, and if it fails, I can can debug to figure out why.
    assert(__.any(self.sharedClones, function(clone)
        return torch.pointer(clone) == torch.pointer(self.recurrentModule)
    end), 'self.recurrentModule not in self.sharedClones')

    assert(__.any(self.sharedClones, function(clone)
        return torch.pointer(clone) == torch.pointer(self.initialModule)
    end), 'self.initialModule not in self.sharedClones')

    return self
end

function InitUpdateRecurrent:recycle()
    self.nSharedClone = self.nSharedClone or __.size(self.sharedClones)

    local rho = self.rho + 1
    if self.sharedClones[self.step] == nil then
        local previous_step
        if self:_is_reinit(self.step) == self:_is_reinit(self.step - rho) then
            previous_step = self.step - rho
        elseif self:_is_reinit(self.step) then
            previous_step = self:_last_reinit(self.step - rho)
        end
        if previous_step ~= nil then
            self.sharedClones[self.step] = self.sharedClones[previous_step]
            self.sharedClones[previous_step] = nil
            self._gradOutputs[self.step] = self._gradOutputs[previous_step]
            self._gradOutputs[previous_step] = nil
        end
    end

    self.outputs[self.step-rho-1] = nil
    self.gradInputs[self.step-rho-1] = nil

    return self
end


function InitUpdateRecurrent:updateOutput(input)
    -- output(t) = transfer(feedback(output_(t-1)) + input(input_(t)))
    local module
    if self.train ~= false then
        -- set/save the output states
        self:recycle()
        module = self:getStepModule(self.step)
    else
        -- self.output is the previous output of this module
        module = self:_is_reinit(self.step) and self.initialModule
                                            or self.recurrentModule
    end
    local output = module:updateOutput(input)

    self.outputs[self.step] = output
    self.output = output
    self.step = self.step + 1
    self.gradPrevOutput = nil
    self.updateGradInputStep = nil
    self.accGradParametersStep = nil
    return self.output
end

function InitUpdateRecurrent:_updateGradInput(input, gradOutput)
    assert(self.step > 1, "expecting at least one updateOutput")
    local step = self.updateGradInputStep - 1
    local module = self:getStepModule(step)
    return module:updateGradInput(input, gradOutput)
end

function InitUpdateRecurrent:_accGradParameters(input, gradOutput, scale)
    local step = self.accGradParametersStep - 1

    local gradOutput = (step == self.step-1) and gradOutput
                                             or self._gradOutputs[step]

    local module = self:getStepModule(step)
    module:accGradParameters(input, gradOutput, scale)
end

function InitUpdateRecurrent:includingSharedClones(f)
    nn.AbstractRecurrent.includingSharedClones(self, f)
end

function InitUpdateRecurrent:reinforce(reward)
    -- I don't know what this method is supposed to do, so I'm not going to try
    -- to implemenet it.
    error('Reinforce not implemented for InitUpdateRecurrent.')
end

function InitUpdateRecurrent:__tostring__()
    -- Bits and pieces from nn.Recurrent, nn.Sequential, and nn.ConcatTable.
    local tab = '  '
    local line = '\n'
    local next_bar = ' |`-> '
    local next = '  `-> '
    local ext = '  |    '
    local str = torch.type(self)
    local last = '  ... -> '
    str = str .. ' {' .. line .. tab .. 'input(t)'

    str = str .. line .. tab .. next_bar ..
          '(t==1): ' ..
          tostring(self.initialModule):gsub(line, line .. tab .. ext)

    str = str .. line .. tab .. next ..
          '(t>1): ' ..
          tostring(self.recurrentModule):gsub(line, line .. tab .. ext)

    str = str .. line .. tab .. last .. 'output'
    str = str .. line .. '}'
    return str
end
