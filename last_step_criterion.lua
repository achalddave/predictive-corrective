------------------------------------------------------------------------
--[[ LastStepCriterion ]]--
-- Applies a criterion only to the last input and target in the correpsonding
-- inputs and targets tables.
--
-- Useful for nn.Repeater and nn.Sequencer.
-- WARNING : assumes that the decorated criterion is stateless, i.e.
-- the backward doesn't need to be preceded by a commensurate forward.
------------------------------------------------------------------------
require 'nn'

local LastStepCriterion, parent = torch.class('nn.LastStepCriterion', 'nn.Criterion')

function LastStepCriterion:__init(criterion)
   parent.__init(self)
   self.criterion = criterion
   if torch.isTypeOf(criterion, 'nn.ModuleCriterion') then
      error("LastStepCriterion shouldn't decorate a ModuleCriterion. "..
         "Instead, try the other way around : "..
         "ModuleCriterion decorates a LastStepCriterion. "..
         "Its modules can also be similarly decorated with a Sequencer.")
   end
   self.gradInput = {}
end

function LastStepCriterion:updateOutput(input, target)
   self.output = 0
   local nStep
   if torch.isTensor(input) then
      assert(torch.isTensor(target),
             "Expecting target to be a Tensor since input is a Tensor.")
      assert(target:size(1) == input:size(1),
             "Target should have as many elements as input.")
      nStep = input:size(1)
   else
      assert(torch.type(target) == 'table', "Expecting target to be a table.")
      assert(#target == #input, "Target should have as many elements as input.")
      nStep = #input
   end

   self.output = self.criterion:forward(input[nStep], target[nStep])
   return self.output
end

function LastStepCriterion:updateGradInput(input, target)
   self.gradInput = {}
   if torch.isTensor(input) then
      assert(torch.isTensor(target),
             "Expecting target to be a Tensor since input is a Tensor.")
      assert(target:size(1) == input:size(1),
             "Target should have as many elements as input.")
      nStep = input:size(1)
   else
      assert(torch.type(target) == 'table', "Expecting target to be a table.")
      assert(#target == #input, "Target should have as many elements as input.")
      nStep = #input
   end

   local tableGradInput = {}
   for i=1,nStep-1 do
       tableGradInput[i] = torch.zeros(target[i]:size()):typeAs(input)
   end
   tableGradInput[nStep] = self.criterion:backward(input[nStep], target[nStep])

   if torch.isTensor(input) then
      self.gradInput = tableGradInput[1].new()
      self.gradInput:resize(nStep, unpack(tableGradInput[1]:size():totable()))
      for step=1,nStep do
         self.gradInput[step]:copy(tableGradInput[step])
      end
   else
      self.gradInput = tableGradInput
   end

   return self.gradInput
end
