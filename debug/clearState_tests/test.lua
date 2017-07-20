local log = require 'util/log'
require 'rnn'
require 'nn'
require 'cunn'
require 'cudnn'
local cutorch = require 'cutorch'

gpus = {1}
model = nn.Sequential()
    :add(cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.ReLU(true))
    :add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.ReLU(true))
    :add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    :add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.ReLU(true))
    :add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.ReLU(true))
    :add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    :add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.ReLU(true))
    :add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.ReLU(true))
    :add(cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.ReLU(true))
    :add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    :add(cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.ReLU(true))
    :add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.ReLU(true))
    :add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.ReLU(true))
    :add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    :add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.ReLU(true))
    :add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.ReLU(true))
    :add(cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1))
    :add(cudnn.ReLU(true))
    :add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
    :add(nn.View(-1):setNumInputDims(3))
    :add(nn.Linear(25088, 4096))
    :add(cudnn.ReLU(true))
    :add(nn.Dropout(0.500000))
    :add(nn.Linear(4096, 4096))
    :add(cudnn.ReLU(true))
    :add(nn.Dropout(0.500000))
    :add(nn.Linear(4096, 1000))
    :add(cudnn.SoftMax()):cuda()

-- model = nn.DataParallelTable(1)
--           :add(model, gpus)

batch_size = 30
log.info('Batch size', batch_size)
input = torch.randn(batch_size, 3, 224, 224):cuda()
grad = torch.randn(batch_size, 1000):cuda()
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
  collectgarbage()
  collectgarbage()
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
