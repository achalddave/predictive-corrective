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

local function test_dataParallelTable()
    -- In general, we can't handle batched inputs (i.e. more than 1 sequence at
    -- a time), since this model maintains state for the sequence. However, with
    -- DataParallelTable, there is a copy of this model on each GPU, so we can
    -- use as many sequences as GPUs. This test ensures that this is, in fact,
    -- the case, and that the state of the model on one GPU doesn't affect the
    -- model on another GPU.
    require 'cunn'
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local single_model = nn.Sequential()
    single_model:add(nn.SplitTable(1))
    single_model:add(nn.PredictiveCorrectiveBlock(init, update, 0.1))
    -- Add a dimension to join the batch over, then join over it.
    single_model:add(nn.MapTable():add(nn.Unsqueeze(1)))
    single_model:add(nn.JoinTable(1))

    single_model:cuda()
    model = nn.DataParallelTable(2)
    model:add(single_model, {1, 2, 3, 4})

    local inputs = torch.ones(5 --[[sequence]], 4 --[[batch]], 5, 5):cuda()
    inputs[{{2,5}, {1}}] = 0  -- Trigger reinit at frame 2 for batch index 1
    inputs[{{3,5}, {2}}] = 0  -- Trigger reinit at frame 3 for batch index 2
    inputs[{{4,5}, {3}}] = 0  -- Trigger reinit at frame 4 for batch index 3
    inputs[{{5,5}, {4}}] = 0  -- Trigger reinit at frame 5 for batch index 4

    -- Size: (6, 4, 5, 5)
    local outputs = model:forward(inputs)

    do
        batch1_inputs = inputs[{{}, 1}]
        batch1_outputs = outputs[{{}, 1}]
        assert(test_util.equals(batch1_outputs[1], batch1_inputs[1]))
        assert(test_util.equals(batch1_outputs[2], batch1_inputs[2])) -- reinit
        assert(test_util.equals(
            batch1_outputs[3],
            2*(batch1_inputs[3] - batch1_inputs[2]) + batch1_outputs[2]))
        assert(test_util.equals(
            batch1_outputs[4],
            2*(batch1_inputs[4] - batch1_inputs[3]) + batch1_outputs[3]))
        assert(test_util.equals(
            batch1_outputs[5],
            2*(batch1_inputs[5] - batch1_inputs[4]) + batch1_outputs[4]))
    end

    do
        batch2_inputs = inputs[{{}, 2}]
        batch2_outputs = outputs[{{}, 2}]
        assert(test_util.equals(batch2_outputs[1], batch2_inputs[1]))
        assert(test_util.equals(
            batch2_outputs[2],
            2*(batch2_inputs[2] - batch2_inputs[1]) + batch2_outputs[1]))
        assert(test_util.equals(batch2_outputs[3], batch2_inputs[3])) -- reinit
        assert(test_util.equals(
            batch2_outputs[4],
            2*(batch2_inputs[4] - batch2_inputs[3]) + batch2_outputs[3]))
        assert(test_util.equals(
            batch2_outputs[5],
            2*(batch2_inputs[5] - batch2_inputs[4]) + batch2_outputs[4]))
    end

    do
        batch3_inputs = inputs[{{}, 3}]
        batch3_outputs = outputs[{{}, 3}]
        assert(test_util.equals(batch3_outputs[1], batch3_inputs[1]))
        assert(test_util.equals(
            batch3_outputs[2],
            2*(batch3_inputs[2] - batch3_inputs[1]) + batch3_outputs[1]))
        assert(test_util.equals(
            batch3_outputs[3],
            2*(batch3_inputs[3] - batch3_inputs[2]) + batch3_outputs[2]))
        assert(test_util.equals(batch2_outputs[4], batch2_inputs[4])) -- reinit
        assert(test_util.equals(
            batch3_outputs[5],
            2*(batch3_inputs[5] - batch3_inputs[4]) + batch3_outputs[4]))
    end

    do
        batch4_inputs = inputs[{{}, 4}]
        batch4_outputs = outputs[{{}, 4}]
        assert(test_util.equals(batch4_outputs[1], batch4_inputs[1]))
        assert(test_util.equals(
            batch4_outputs[2],
            2*(batch4_inputs[2] - batch4_inputs[1]) + batch4_outputs[1]))
        assert(test_util.equals(
            batch4_outputs[3],
            2*(batch4_inputs[3] - batch4_inputs[2]) + batch4_outputs[2]))
        assert(test_util.equals(
            batch4_outputs[4],
            2*(batch4_inputs[4] - batch4_inputs[3]) + batch4_outputs[3]))
        assert(test_util.equals(batch4_outputs[5], batch4_inputs[5])) -- reinit
    end
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
test_util.run_test(test_dataParallelTable, 'DataParallelTable test')
