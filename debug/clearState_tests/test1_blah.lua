require 'scripts/imports.lua'
local optim = require 'optim'
local log = require 'util/log'

-- log.info('Loading model.')
-- old = torch.load('/data/achald/MultiTHUMOS/models/vgg_single_frame/train_all/sampling-with-aug/permuted_with_no_replace/02-12-17-19-12-05/model_30_c33_1-debug.t7'):cuda()
-- log.info('Loaded model')
-- old:clearState()
-- collectgarbage()
-- collectgarbage()

-- TODO(achald): Find out which keys are the problematic ones.
-- local x = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1):cuda()

--[[
-- local old_x = old:get(2):get(1):get(15)
-- to_replace = { 'padW', 'nInputPlane', 'output', 'name', 'gradInput', 'iSize', '_type', 'gradBias', 'groups', 'dH', 'dW', 'output_offset', 'kW', 'kH', 'weight_offset', 'input_offset', 'weight', 'train', 'gradWeight', 'bias', 'padH', 'nOutputPlane' }
to_replace = { 'weight', 'gradWeight', 'bias', 'gradBias' }
for _, k in ipairs(to_replace) do
    -- print(k)
    -- print(old_x[k]:storage():size())
    -- print(x[k]:storage():size())
    -- x[k] = old_x[k]
    -- old_x[k] = old_x[k]:clone()
    -- old_x[k] = x[k]
end
collectgarbage()
collectgarbage()
]]--

a = nn.Sequential()
    :add(nn.SplitTable(1))
    :add(nn.MapTable()
        :add(nn.Sequential()
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
            -- :add(x)
            -- :add(old_x)
            )):cuda()
log.info('Created net')

log.info('Getting parameters')
mp, mgp = a:getParameters()
log.info('Got parameters')

log.info('After creation')
os.execute('nvidia-smi | grep luajit')
a:clearState()
collectgarbage()
collectgarbage()

--[[
-- old.modules[2].modules[1].modules[15] = nil
-- old = nil
--]]


-- Works with 17
batch_size = 1
log.info('Batch size', batch_size)
i = torch.rand(4, batch_size, 3, 224, 224):cuda()

local function clearState(should_log)
    should_log = should_log == nil and true or should_log
    if should_log then
        log.info('Before clearState', j)
        os.execute('nvidia-smi | grep luajit')
    end
    a:clearState()
    if should_log then
        log.info('After clearState', j)
        os.execute('nvidia-smi | grep luajit')
    end
end

local state = {}
local config = {}
-- a = old
for j = 1, 3 do
    log.info('Before training', j)
    os.execute('nvidia-smi | grep luajit')

    if j >= 2 then
        -- mp, mgp = a:getParameters()
        clearState(false)
        -- mp, mgp = a:getParameters()
    end

    o = a:forward(i)
    log.info('After forward', j)
    os.execute('nvidia-smi | grep luajit')
end
