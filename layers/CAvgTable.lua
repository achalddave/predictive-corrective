require 'nn'

local CAvgTable, parent = torch.class('nn.CAvgTable', 'nn.Module')

function CAvgTable:__init(ip)
   parent.__init(self)
   self.inplace = ip
   self.gradInput = {}
end

function CAvgTable:updateOutput(input)
   if self.inplace then
      self.output:set(input[1])
   else
      self.output:resizeAs(input[1]):copy(input[1])
   end
   for i=2,#input do
      self.output:add(input[i])
   end
   self.output:div(#input)
   return self.output
end

function CAvgTable:updateGradInput(input, gradOutput)
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[1].new()
      if self.inplace then
         self.gradInput[i]:set(gradOutput):div(#input)
      else
         self.gradInput[i]:resizeAs(input[i]):copy(gradOutput):div(#input)
      end
   end

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end
