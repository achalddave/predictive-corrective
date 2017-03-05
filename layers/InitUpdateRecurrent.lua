local nn = require 'nn'
local rnn = require 'rnn'
local torch = require 'torch'
local __ = require 'moses'
require 'dpnn'

require 'util/strict'

local InitUpdateRecurrent, parent = torch.class(
    'nn.InitUpdateRecurrent', 'nn.Recurrent')

function InitUpdateRecurrent:__init(init, update, rho)
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
    --]]

    nn.AbstractRecurrent.__init(self, rho)

    self.initialModule = init
    self.update = update

    -- Recurrent module will be passed {input_t, output_{t-1}} by the parent
    -- class; we select the first input (input_t) and pass it through
    -- self.update.
    self.recurrentModule = nn.Sequential()
        :add(nn.SelectTable(1))
        :add(self.update)
    self.sharedClones[2] = self.recurrentModule

    self.modules = {self.initialModule, self.recurrentModule}
end

function InitUpdateRecurrent:getStepModule(step)
    assert(step, "expecting step at arg 1")
    assert(step > 1, 'step must be >1 since first step is for initialModule.')
    local recurrentModule = self.sharedClones[step]
    if not recurrentModule then
        recurrentModule = self.recurrentModule:stepClone()
        self.sharedClones[step] = recurrentModule
        self.nSharedClone = __.size(self.sharedClones)
    end
    return recurrentModule
end

function InitUpdateRecurrent:_updateGradInput(input, gradOutput)
    assert(self.updateGradInputStep >= self.step - self.rho,
           string.format('Called backward more than rho+1=%d times',
                         self.rho+1))
    nn.Recurrent._updateGradInput(self, input, gradOutput)
end

-- Updated since our sharedClones start at index 2.
function InitUpdateRecurrent:recycle()
   self.nSharedClone = self.nSharedClone or __.size(self.sharedClones)

   local rho = math.max(self.rho + 1, self.nSharedClone)
   if self.sharedClones[self.step] == nil then
      self.sharedClones[self.step] = self.sharedClones[self.step-rho]
      self.sharedClones[self.step-rho] = nil
      self._gradOutputs[self.step] = self._gradOutputs[self.step-rho]
      self._gradOutputs[self.step-rho] = nil
   end

   self.outputs[self.step-rho-1] = nil
   self.gradInputs[self.step-rho-1] = nil

   return self
end

-- Updated since our sharedClones start at index 2.
function InitUpdateRecurrent:forget()
    local offset = 1

    -- the recurrentModule may contain an AbstractRecurrent instance (issue 107)
    nn.Container.forget(self)

    -- bring all states back to the start of the sequence buffers
    if self.train ~= false then
        self.outputs = {}
        self.gradInputs = {}
        local compactSharedClones = __.compact(self.sharedClones)
        self.sharedClones = {}
        for i, v in ipairs(compactSharedClones) do
            self.sharedClones[i + offset] = v
        end

        local compactGradOutputs = __.compact(self._gradOutputs)
        self._gradOutputs = {}
        for i, v in ipairs(compactGradOutputs) do
            self._gradOutputs[i + offset] = v
        end
        self.gradOutputs = {}
        if self.cells then
            self.cells = {}
            self.gradCells = {}
        end
    end

    -- forget the past inputs; restart from first step
    self.step = 1

    if not self.rmInSharedClones then
        -- Asserts that issue 129 is solved. In forget as it is often called.
        -- Asserts that self.recurrentModule is part of the sharedClones.
        -- Since its used for evaluation, it should be used for training.
        local nClone, maxIdx = 0, offset
        for k,v in pairs(self.sharedClones) do -- to prevent odd bugs
            if torch.pointer(v) == torch.pointer(self.recurrentModule) then
                self.rmInSharedClones = true
                maxIdx = math.max(k, maxIdx)
            end
            nClone = nClone + 1
        end
        if nClone > 1 then
            if not self.rmInSharedClones then
                print"WARNING : recurrentModule should be added to sharedClones in constructor."
                print"Adding it for you."
                assert(torch.type(self.sharedClones[maxIdx]) == torch.type(self.recurrentModule))
                self.recurrentModule = self.sharedClones[maxIdx]
                self.rmInSharedClones = true
            end
        end
    end
    return self
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
          tostring(self.update):gsub(line, line .. tab .. ext)

    str = str .. line .. tab .. last .. 'output'
    str = str .. line .. '}'
    return str
end
