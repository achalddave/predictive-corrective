require 'scripts/imports.lua'
local optim = require 'optim'
local log = require 'util/log'

-- a = torch.load('/data/achald/MultiTHUMOS/models/vgg_single_frame/train_all/background_factor_20/with_weight_decay_10x_fc/11-02-16-12-39-12/model_30.t7'):cuda()
-- b = a

a = torch.load('/data/achald/MultiTHUMOS/models/vgg_single_frame/train_all/sampling-with-aug/permuted/02-09-17-19-01-00/model_30_conv33_1_fc7_4.t7'):cuda()
a:clearState()
a:training()
b = nn.DataParallelTable(2)
b:add(a, {1, 2, 3, 4})
b = b:cuda()

log.info('Getting parameters')
mp, mgp = b:getParameters()
log.info('Got parameters')

-- Works with 32
batch_size = 34
i = torch.rand(4, batch_size, 3, 224, 224):cuda()
l = torch.rand(4, batch_size, 65):cuda()

c = nn.SequencerCriterion(nn.MultiLabelSoftMarginCriterion():cuda()):cuda()

o = b:forward(i)
loss = c:forward(o, l)

local function backward()
    g = c:backward(o, l)
    b:backward(i, g)
    return loss, mgp
end

local state = {}
local config = {}
for j = 1, 50 do
    b:clearState()
    collectgarbage()
    collectgarbage()

    log.info('Training', j)
    b:training()
    b:clearState()
    collectgarbage()
    collectgarbage()
    b:zeroGradParameters()
    o = b:forward(i)
    loss = c:forward(o, l)
    backward()

    log.info('Evaluating', j)
    b:evaluate()
    b:clearState()
    collectgarbage()
    collectgarbage()
    b:forward(i)
end

-- loss = c:forward(o, l)

-- g = c:backward(o, l)
-- b:backward(i, g)
