local nn = require 'nn'
local torch = require 'torch'
local test_util = require 'tests/test_util'
require 'layers/PredictiveCorrectiveBlock'

local function test_no_reinit()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local pcb = nn.PredictiveCorrectiveBlock(init, update, math.huge)

    local inputs = {}
    for i = 1, 4 do inputs[i] = torch.rand(5, 5) end

    local outputs = pcb:forward(inputs)

    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], 2*(inputs[2] - inputs[1]) + outputs[1]))
    assert(test_util.equals(outputs[3], 2*(inputs[3] - inputs[2]) + outputs[2]))
    assert(test_util.equals(outputs[4], 2*(inputs[4] - inputs[3]) + outputs[3]))
end

local function test_reinit_every_time()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local pcb = nn.PredictiveCorrectiveBlock(init, update, 0)

    local inputs = {}
    for i = 1, 4 do inputs[i] = torch.rand(5, 5) end

    local outputs = pcb:forward(inputs)

    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], inputs[2]))
    assert(test_util.equals(outputs[3], inputs[3]))
    assert(test_util.equals(outputs[4], inputs[4]))
end

local function test_reinit_threshold()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local pcb = nn.PredictiveCorrectiveBlock(init, update, 0.1)

    local inputs = {}
    inputs[1] = torch.ones(5, 5)
    inputs[2] = torch.ones(5, 5)
    inputs[3] = torch.zeros(5, 5)  -- Should trigger a reinit.
    inputs[4] = torch.zeros(5, 5)

    assert((inputs[3] - inputs[2]):norm() == 5)

    local outputs = pcb:forward(inputs)
    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], 2*(inputs[2] - inputs[1]) + outputs[1]))
    assert(test_util.equals(outputs[3], inputs[3]))
    assert(test_util.equals(outputs[4], 2*(inputs[4] - inputs[3]) + outputs[3]))
end

local function test_longer_input_after_shorter_input()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local pcb = nn.PredictiveCorrectiveBlock(init, update, 0.1)

    local inputs = {}
    inputs[1] = torch.ones(5, 5)
    inputs[2] = torch.ones(5, 5)
    inputs[3] = torch.zeros(5, 5)  -- Should trigger a reinit.
    inputs[4] = torch.zeros(5, 5)

    assert((inputs[3] - inputs[2]):norm() == 5)

    local outputs = pcb:forward(inputs)
    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], 2*(inputs[2] - inputs[1]) + outputs[1]))
    assert(test_util.equals(outputs[3], inputs[3]))
    assert(test_util.equals(outputs[4], 2*(inputs[4] - inputs[3]) + outputs[3]))

    inputs = {}
    for i = 1, 4 do inputs[i] = torch.ones(5, 5) end
    for i = 5, 8 do inputs[i] = torch.zeros(5, 5) end

    outputs = pcb:forward(inputs)
    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], 2*(inputs[2] - inputs[1]) + outputs[1]))
    assert(test_util.equals(outputs[3], 2*(inputs[3] - inputs[2]) + outputs[2]))
    assert(test_util.equals(outputs[4], 2*(inputs[4] - inputs[3]) + outputs[3]))
    assert(test_util.equals(outputs[5], inputs[5]))
    assert(test_util.equals(outputs[6], 2*(inputs[6] - inputs[5]) + outputs[5]))
    assert(test_util.equals(outputs[7], 2*(inputs[7] - inputs[6]) + outputs[6]))
    assert(test_util.equals(outputs[8], 2*(inputs[8] - inputs[7]) + outputs[7]))
end

local function test_shorter_input_after_longer_input()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local pcb = nn.PredictiveCorrectiveBlock(init, update, 0.1)

    local inputs = {}
    for i = 1, 4 do inputs[i] = torch.ones(5, 5) end
    for i = 5, 8 do inputs[i] = torch.zeros(5, 5) end

    local outputs = pcb:forward(inputs)
    assert(#outputs == #inputs)
    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], 2*(inputs[2] - inputs[1]) + outputs[1]))
    assert(test_util.equals(outputs[3], 2*(inputs[3] - inputs[2]) + outputs[2]))
    assert(test_util.equals(outputs[4], 2*(inputs[4] - inputs[3]) + outputs[3]))
    assert(test_util.equals(outputs[5], inputs[5]))
    assert(test_util.equals(outputs[6], 2*(inputs[6] - inputs[5]) + outputs[5]))
    assert(test_util.equals(outputs[7], 2*(inputs[7] - inputs[6]) + outputs[6]))
    assert(test_util.equals(outputs[8], 2*(inputs[8] - inputs[7]) + outputs[7]))

    inputs = {}
    inputs[1] = torch.ones(5, 5)
    inputs[2] = torch.ones(5, 5)
    inputs[3] = torch.zeros(5, 5)  -- Should trigger a reinit.
    inputs[4] = torch.zeros(5, 5)

    outputs = pcb:forward(inputs)
    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], 2*(inputs[2] - inputs[1]) + outputs[1]))
    assert(test_util.equals(outputs[3], inputs[3]))
    assert(test_util.equals(outputs[4], 2*(inputs[4] - inputs[3]) + outputs[3]))
end

local function test_ignoreInputs()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local pcb = nn.PredictiveCorrectiveBlock(
        init, update, 1 --[[init threshold]], math.huge --[[max update]],
        1 --[[ignore threshold]])

    local inputs = {}
    for i = 1, 4 do inputs[i] = torch.ones(5, 5) + (0.1 * i)  end
    for i = 5, 8 do inputs[i] = torch.zeros(5, 5) - (25*2*i) end

    local outputs = pcb:forward(inputs)
    assert(#outputs == #inputs)
    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], inputs[1]))
    assert(test_util.equals(outputs[3], inputs[1]))
    assert(test_util.equals(outputs[4], inputs[1]))
    assert(test_util.equals(outputs[5], inputs[5]))
    assert(test_util.equals(outputs[6], inputs[6]))
    assert(test_util.equals(outputs[7], inputs[7]))
    assert(test_util.equals(outputs[8], inputs[8]))
end

local function test_clearState()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local pcb = nn.PredictiveCorrectiveBlock(init, update, 0.1)

    local inputs = {}
    inputs[1] = torch.ones(5, 5)
    inputs[2] = torch.ones(5, 5)
    inputs[3] = torch.zeros(5, 5)  -- Should trigger a reinit.
    inputs[4] = torch.zeros(5, 5)

    assert((inputs[3] - inputs[2]):norm() == 5)

    local outputs = pcb:forward(inputs)
    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], 2*(inputs[2] - inputs[1]) + outputs[1]))
    assert(test_util.equals(outputs[3], inputs[3]))
    assert(test_util.equals(outputs[4], 2*(inputs[4] - inputs[3]) + outputs[3]))

    pcb:clearState()
    outputs = pcb:forward(inputs)
    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], 2*(inputs[2] - inputs[1]) + outputs[1]))
    assert(test_util.equals(outputs[3], inputs[3]))
    assert(test_util.equals(outputs[4], 2*(inputs[4] - inputs[3]) + outputs[3]))
end

test_util.run_test(test_no_reinit, 'No reinit test')
test_util.run_test(test_reinit_every_time, 'Reinit every time')
test_util.run_test(test_reinit_threshold, 'Reinit with threshold')
test_util.run_test(test_longer_input_after_shorter_input,
                   'Longer input after shorter input')
test_util.run_test(test_shorter_input_after_longer_input,
                   'Shorter input after longer input')
test_util.run_test(test_ignoreInputs,
                   'Ignore inputs.')
test_util.run_test(test_clearState, 'clearState')
