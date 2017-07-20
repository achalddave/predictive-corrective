--[[ Displays memory 'leak' with nn.MapTable after clearState() ]]--

local nn = require 'nn'
local cunn = require 'cunn'

model = nn.MapTable():add(nn.SpatialConvolution(3, 256, 3, 3, 1, 1, 1, 1, 1))
                     :cuda()
i = {torch.rand(30, 3, 224, 224):cuda(), torch.rand(30, 3, 224, 224):cuda()}
mp, mgp = model:getParameters()

function check_mem() os.execute('nvidia-smi | grep luajit') end
-- Train two iterations without clear state:
print('Before training 1'); check_mem() -- 277 MiB
o = model:forward(i)
print('After forward 1'); check_mem() -- 3254 MiB

-- Train another iteration:
print('Before training 2'); check_mem() -- 3254 MiB
o = model:forward(i)
print('After forward 2'); check_mem() -- 3254 MiB

-- Clear state:
model:resize(1)
collectgarbage()
collectgarbage()

-- Train a final iteration before clearState. This final forward call causes an
-- increase in memory usage for the rest of the program!
print('Before training 3 (after clearState)'); check_mem() -- 3254 MiB
o = model:forward(i)
print('After forward 3 (after clearState)'); check_mem() -- 4724 MiB!

-- Garbage collection doesn't fix it.
collectgarbage()
collectgarbage()
print('After collectgarbage()'); check_mem() -- 4724 MiB!
