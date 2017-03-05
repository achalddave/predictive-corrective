local nn = require 'nn'
local torch = require 'torch'

local test_util = require 'tests/test_util'
require 'layers/PredictiveCorrectiveRecurrent'

local function test_no_reinit()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local model = nn.PredictiveCorrectiveRecurrent(
        init, update, torch.LongStorage({5, 5}), torch.LongStorage({5, 5}), 2, math.huge)
    model = nn.Sequencer(model)

    local inputs = {}
    for i = 1, 4 do inputs[i] = torch.rand(5, 5) end

    local outputs = model:forward(inputs)

    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], 2*(inputs[2] - inputs[1]) + outputs[1]))
    assert(test_util.equals(outputs[3], 2*(inputs[3] - inputs[2]) + outputs[2]))
    assert(test_util.equals(outputs[4], 2*(inputs[4] - inputs[3]) + outputs[3]))
end

local function test_reinit_every_time()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local model = nn.PredictiveCorrectiveRecurrent(
        init, update, torch.LongStorage({5, 5}), torch.LongStorage({5, 5}), 2, math.huge)

    local inputs = {}
    for i = 1, 4 do inputs[i] = torch.rand(5, 5) end

    local outputs = {}
    for i = 1, 4 do
        outputs[i] = model:forward(inputs[i]):clone()
        model:forget()
    end

    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], inputs[2]))
    assert(test_util.equals(outputs[3], inputs[3]))
    assert(test_util.equals(outputs[4], inputs[4]))
end

local function test_clearState()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local model = nn.PredictiveCorrectiveRecurrent(
        init, update, torch.LongStorage({5, 5}), torch.LongStorage({5, 5}), 2, math.huge)
    model = nn.Sequencer(model)

    local inputs = {}
    for i = 1, 4 do inputs[i] = torch.rand(5, 5) end

    local outputs = model:forward(inputs)

    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], 2*(inputs[2] - inputs[1]) + outputs[1]))
    assert(test_util.equals(outputs[3], 2*(inputs[3] - inputs[2]) + outputs[2]))
    assert(test_util.equals(outputs[4], 2*(inputs[4] - inputs[3]) + outputs[3]))

    model:clearState()
    outputs = model:forward(inputs)
    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], 2*(inputs[2] - inputs[1]) + outputs[1]))
    assert(test_util.equals(outputs[3], 2*(inputs[3] - inputs[2]) + outputs[2]))
    assert(test_util.equals(outputs[4], 2*(inputs[4] - inputs[3]) + outputs[3]))
end

test_util.run_test(test_no_reinit, 'No reinit test')
test_util.run_test(test_reinit_every_time, 'Reinit every time')
test_util.run_test(test_clearState, 'clearState')
