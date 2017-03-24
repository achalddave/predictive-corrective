local nn = require 'nn'
local torch = require 'torch'
local __ = require 'moses'

local test_util = require 'tests/test_util'
require 'layers/init'

local function test_no_reinit()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local model = nn.PredictiveCorrectiveRecurrent(init, update)
    model = nn.Sequencer(model)

    local inputs = {}
    for i = 1, 4 do inputs[i] = torch.rand(5, 5) end

    local outputs = model:forward(inputs)

    assert(test_util.equals(outputs[1], inputs[1]))
    assert(test_util.equals(outputs[2], 2*(inputs[2] - inputs[1]) + outputs[1]))
    assert(test_util.equals(outputs[3], 2*(inputs[3] - inputs[2]) + outputs[2]))
    assert(test_util.equals(outputs[4], 2*(inputs[4] - inputs[3]) + outputs[3]))
end

local function test_no_reinit_backward()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local model = nn.PredictiveCorrectiveRecurrent(init, update, math.huge)
    model = nn.Sequencer(model)

    local inputs = {}
    for i = 1, 4 do inputs[i] = torch.rand(5, 5) end

    local outputs = model:forward(inputs)
    -- Just test that this does not error.
    model:backward(inputs, inputs)
end

local function test_backward_one_step()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local reinit = 4
    local rho = 4

    local model = nn.PredictiveCorrectiveRecurrent(init, update, reinit, rho)
    model = nn.Sequencer(model)

    local model_block = nn.Sequential()
        :add(nn.CRollingDiffTable(reinit))
        :add(nn.PeriodicResidualTable(reinit, init:clone(), update:clone()))
        :add(nn.CCumSumTable(reinit))

    local inputs = {torch.rand(5, 5)}

    local outputs_block = model_block:forward(inputs)
    local outputs = model:forward(inputs)

    assert(test_util.equals(outputs_block[1], outputs[1]))

    local gradients = outputs

    local computed_gradients = model:backward(inputs, gradients)
    local computed_gradients_block = model_block:backward(inputs, gradients)
    assert(test_util.equals(computed_gradients[1], computed_gradients_block[1]))
end

local function test_backward_multiple_steps()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local reinit = 4
    local rho = 4

    local function clone(table_of_tensors)
        return __.invoke(table_of_tensors, function(x) return x:clone() end)
    end

    -- We don't want to put the assertions in a loop so that a failed assertion
    -- points to an explicit step. To do this, we create two separate methods
    -- for 4 vs. 8 inputs.
    local function compare_modules_4(module1, module2)
        local inputs = {}
        for i = 1, 4 do inputs[i] = torch.rand(5, 5) end
        local outputs1 = clone(module1:forward(inputs))
        local outputs2 = clone(module2:forward(inputs))

        assert(test_util.equals(outputs1[1], outputs2[1]))
        assert(test_util.equals(outputs1[2], outputs2[2]))
        assert(test_util.equals(outputs1[3], outputs2[3]))
        assert(test_util.equals(outputs1[4], outputs2[4]))

        local gradients_out = clone(outputs1)
        local gradients1 = module1:backward(inputs, gradients_out)
        local gradients2 = module2:backward(inputs, gradients_out)

        assert(test_util.equals(gradients1[1], gradients2[1]))
        assert(test_util.equals(gradients1[2], gradients2[2]))
        assert(test_util.equals(gradients1[3], gradients2[3]))
        assert(test_util.equals(gradients1[4], gradients2[4]))
    end

    local function compare_modules_8(module1, module2)
        local inputs = {}
        for i = 1, 8 do inputs[i] = torch.rand(5, 5) end
        local outputs1 = clone(module1:forward(inputs))
        local outputs2 = clone(module2:forward(inputs))

        assert(test_util.equals(outputs1[1], outputs2[1]))
        assert(test_util.equals(outputs1[2], outputs2[2]))
        assert(test_util.equals(outputs1[3], outputs2[3]))
        assert(test_util.equals(outputs1[4], outputs2[4]))
        assert(test_util.equals(outputs1[5], outputs2[5]))
        assert(test_util.equals(outputs1[6], outputs2[6]))
        assert(test_util.equals(outputs1[7], outputs2[7]))
        assert(test_util.equals(outputs1[8], outputs2[8]))

        local gradients_out = clone(outputs1)
        local gradients1 = module1:backward(inputs, gradients_out)
        local gradients2 = module2:backward(inputs, gradients_out)

        assert(test_util.equals(gradients1[1], gradients2[1]))
        assert(test_util.equals(gradients1[2], gradients2[2]))
        assert(test_util.equals(gradients1[3], gradients2[3]))
        assert(test_util.equals(gradients1[4], gradients2[4]))
        assert(test_util.equals(gradients1[5], gradients2[5]))
        assert(test_util.equals(gradients1[6], gradients2[6]))
        assert(test_util.equals(gradients1[7], gradients2[7]))
        assert(test_util.equals(gradients1[8], gradients2[8]))
    end

    -- Test CRollingDiffRecurrent
    do
        local differ = nn.Sequencer(nn.CRollingDiffRecurrent(reinit))
        local differ_block = nn.CRollingDiffTable(reinit)
        compare_modules_4(differ, differ_block)
        compare_modules_8(differ, differ_block)
    end

    -- Test InitUpdateRecurrent
    do
        local init_update = nn.Sequencer(nn.InitUpdateRecurrent(
            init:clone(), update:clone(), reinit))
        local init_update_block = nn.PeriodicResidualTable(
            reinit, init:clone(), update:clone())
        compare_modules_4(init_update, init_update_block)
        compare_modules_8(init_update, init_update_block)
    end

    -- Test CCumSumRecurrent
    do
        local summer = nn.Sequencer(nn.CCumSumRecurrent(reinit))
        local summer_block = nn.CCumSumTable(reinit)
        compare_modules_4(summer, summer_block)
        compare_modules_8(summer, summer_block)
    end

    -- Test PredictiveCorrectiveRecurrent
    do
        local model = nn.Sequencer(
            nn.PredictiveCorrectiveRecurrent(init, update, reinit, rho))
        local model_block = nn.Sequential()
            :add(nn.CRollingDiffTable(reinit))
            :add(nn.PeriodicResidualTable(reinit, init:clone(), update:clone()))
            :add(nn.CCumSumTable(reinit))
        compare_modules_4(model, model_block)
        compare_modules_8(model, model_block)
    end
end

local function test_compare_gradients_different_rhos(module)
    -- Ensure that the setting of rho does not affect the last few
    -- gradients.
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local reinit = 8
    local rho1 = 4
    local rho2 = 8

    local model1 = nn.PredictiveCorrectiveRecurrent(
        init, update, reinit, rho1)
    local model2 = nn.PredictiveCorrectiveRecurrent(
        init:clone(), update:clone(), reinit, rho2)

    local inputs = {}
    for i = 1, 8 do inputs[i] = torch.rand(5, 5) end

    local outputs1 = torch.Tensor(8, 5, 5)
    local outputs2 = torch.Tensor(8, 5, 5)
    for i = 1, 8 do
        outputs1[i] = model1:forward(inputs[i]):clone()
        outputs2[i] = model2:forward(inputs[i]):clone()
    end

    assert(test_util.equals(outputs1, outputs2))

    -- Compute gradients for both models. model1 will only get 4 backprop steps,
    -- but they should be equivalent to the last (or first, depending on how you
    -- look at it) 4 backprop steps of model2.
    local gradients1 = torch.Tensor(8, 5, 5)
    local gradients2 = torch.Tensor(8, 5, 5)
    for i = 8, 5, -1 do
        gradients1[i] = model1:backward(outputs1[i], inputs[i]):clone()
    end
    for i = 8, 1, -1 do
        gradients2[i] = model2:backward(outputs2[i], inputs[i]):clone()
    end

    assert(test_util.equals(gradients1[{{5, 8}}], gradients2[{{5, 8}}]))
end

local function test_compare_gradients_different_rhos_sequencer(module)
    -- Ensure that the setting of rho does not affect the last few
    -- gradients, even with nn.Sequencer wrapper.
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local reinit = 8
    local rho1 = 4
    local rho2 = 8

    local model1 = nn.Sequencer(nn.PredictiveCorrectiveRecurrent(
        init, update, reinit, rho1))
    model1:remember('both')
    model1:training()
    local model2 = nn.Sequencer(nn.PredictiveCorrectiveRecurrent(
        init:clone(), update:clone(), reinit, rho2))
    model2:remember('both')
    model2:training()

    local inputs = torch.rand(8, 5, 5)

    local outputs1 = torch.Tensor(8, 5, 5)
    outputs1[{{1, 4}}] = model1:forward(inputs[{{1, 4}}]):clone()
    outputs1[{{5, 8}}] = model1:forward(inputs[{{5, 8}}]):clone()
    local outputs2 = model2:forward(inputs):clone()
    assert(test_util.equals(outputs1, outputs2))

    -- Compute gradients for both models. model1 will only get 4 backprop steps,
    -- but they should be equivalent to the last (or first, depending on how you
    -- look at it) 4 backprop steps of model2.
    local gradients1 = model1:backward(outputs1[{{5, 8}}], inputs[{{5, 8}}]):clone()
    local gradients2 = model2:backward(outputs2, inputs):clone()

    assert(test_util.equals(gradients1, gradients2[{{5, 8}}]))
end

local function test_sequencer_remember()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local model1 = nn.PredictiveCorrectiveRecurrent(
        init, update, 2 --[[reinit]], 2 --[[rho]])
    model1 = nn.Sequencer(model1)
    model1:remember('both')

    local model2 = nn.PredictiveCorrectiveRecurrent(
        init:clone(), update:clone(), 2 --[[reinit]], 2 --[[rho]])
    model2 = nn.Sequencer(model2)

    local inputs = torch.Tensor(4, 5, 5)
    for i = 1, 4 do inputs[i] = torch.rand(5, 5) end

    local outputs1 = torch.Tensor(4, 5, 5)
    outputs1[{{1, 2}}] = model1:forward(inputs[{{1, 2}}]):clone()
    outputs1[{{3, 4}}] = model1:forward(inputs[{{3, 4}}]):clone()

    local outputs2 = model2:forward(inputs)

    assert(test_util.equals(outputs1, outputs2))
end

local function test_sequencer_highjack_rho()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local reinit = 4
    local rho = 2 -- Should be overridden by nn.Sequencer

    local model = nn.PredictiveCorrectiveRecurrent(init, update, reinit, rho)
    model = nn.Sequencer(model)

    local model_block = nn.Sequential()
        :add(nn.CRollingDiffTable(reinit))
        :add(nn.PeriodicResidualTable(reinit, init:clone(), update:clone()))
        :add(nn.CCumSumTable(reinit))

    local inputs = {}
    for i = 1, 4 do inputs[i] = torch.rand(5, 5) end

    local outputs_block = model_block:forward(inputs)
    local outputs = model:forward(inputs)

    assert(test_util.equals(outputs_block[1], outputs[1]))

    local gradients = outputs

    local computed_gradients = model:backward(inputs, gradients)
    local computed_gradients_block = model_block:backward(inputs, gradients)
    assert(test_util.equals(computed_gradients[1], computed_gradients_block[1]))
    assert(test_util.equals(computed_gradients[2], computed_gradients_block[2]))
    assert(test_util.equals(computed_gradients[3], computed_gradients_block[3]))
    assert(test_util.equals(computed_gradients[4], computed_gradients_block[4]))
end


local function test_backward_small_rho()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local reinit = 4
    local rho = 1

    local model = nn.PredictiveCorrectiveRecurrent(init, update, reinit, rho)

    local model_block = nn.Sequential()
        :add(nn.CRollingDiffTable(reinit))
        :add(nn.PeriodicResidualTable(reinit, init:clone(), update:clone()))
        :add(nn.CCumSumTable(reinit))

    local inputs = {}
    for i = 1, 4 do inputs[i] = torch.rand(5, 5) end

    local outputs_block = model_block:forward(inputs)
    local outputs = {}
    for i = 1, 4 do outputs[i] = model:forward(inputs[i]):clone() end

    assert(test_util.equals(outputs_block[1], outputs[1]))
    assert(test_util.equals(outputs_block[2], outputs[2]))
    assert(test_util.equals(outputs_block[3], outputs[3]))
    assert(test_util.equals(outputs_block[4], outputs[4]))

    local gradients = outputs

    local computed_gradients = {}
    computed_gradients[4] = model:backward(inputs[4], gradients[4]):clone()

    local computed_gradients_block = model_block:backward(inputs, gradients)
    assert(test_util.equals(computed_gradients[4], computed_gradients_block[4]))
end

local function test_reinit_every_time()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local model = nn.PredictiveCorrectiveRecurrent(init, update, math.huge)

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
    local model = nn.PredictiveCorrectiveRecurrent(init, update, math.huge)
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

test_util.run_test(test_no_reinit, 'Forward without reinit')
test_util.run_test(test_no_reinit_backward, 'Backward without reinit')
test_util.run_test(test_reinit_every_time, 'Forward reinit every time')
test_util.run_test(test_backward_one_step, 'Test backward one step')
test_util.run_test(test_backward_multiple_steps, 'Backward multiple steps')
test_util.run_test(test_compare_gradients_different_rhos,
                   'Compare different rhos')
test_util.run_test(test_compare_gradients_different_rhos_sequencer,
                   'Compare different rhos with Sequencer')
test_util.run_test(test_backward_small_rho, 'Test backward with small rho')
test_util.run_test(test_sequencer_highjack_rho, 'Highjack rho')
test_util.run_test(test_sequencer_remember, 'Test sequencer remember')
test_util.run_test(test_clearState, 'clearState')
