local log = require 'util/log'
require 'nn'
require 'cunn'
require 'cudnn'
local cutorch = require 'cutorch'

gpus = {1}
model = nn.Sequential()
          :add(cudnn.SpatialConvolution(3, 512, 3,3, 1,1, 1,1))
          :add(cudnn.SpatialConvolution(3, 512, 3,3, 1,1, 1,1))
          :add(cudnn.SpatialConvolution(3, 512, 3,3, 1,1, 1,1))
          :add(cudnn.SpatialConvolution(3, 512, 3,3, 1,1, 1,1))
          :add(nn.View(-1):setNumInputDims(3))
          :add(nn.Linear(512*224*224, 1)):cuda()
          :add(nn.View(-1)):cuda()
          -- :add(nn.Mean(1,1)):cuda()

model = nn.DataParallelTable(1)
          :add(model, gpus)

-- for i = 16, 100 do
--     local success, err = pcall(function()
--         model = nn.Sequential()
--                 :add(cudnn.SpatialConvolution(3, 512, 3,3, 1,1, 1,1))
--                 :add(nn.View(-1):setNumInputDims(3))
--                 :add(nn.Mean(1,1)):cuda()
-- 
--         model = nn.DataParallelTable(1)
--                 :add(model, {1})
-- 
--         input = torch.randn(i,3,224,224):cuda()
--         grad =  torch.randn(i):cuda()
--         model:forward(input)
--         model:backward(input, grad)
--         model = nil
--         collectgarbage()
--         collectgarbage()
--     end)
--     if not success then
--         log.info('Failed at batch size', i)
--         break
--     end
-- end

batch_size = 10
log.info('Batch size', batch_size)
input = torch.randn(batch_size, 3, 224, 224):cuda()
grad = torch.randn(batch_size):cuda()
collectgarbage()
collectgarbage()


function comma_value(amount)
  local formatted = amount
  while true do  
    formatted, k = string.gsub(formatted, "^(-?%d+)(%d%d%d)", '%1,%2')
    if (k==0) then
      break
    end
  end
  return formatted
end

for i = 1,4 do
  log.info('iteration ' .. tostring(i))

  if i >= 3 then
    model:clearState()
    collectgarbage()
    collectgarbage()
  end
  model:forward(input)
  model:backward(input, grad)
  log.info('Memory usage at end of iter', i)
  for _, gpu in ipairs(gpus) do
      free_mem, total_mem = cutorch.getMemoryUsage(gpu)
      used_mem = total_mem - free_mem
      log.info('gpu:', gpu, 'mem:', comma_value(tostring(used_mem)))
  end
  collectgarbage()
  collectgarbage()
end
