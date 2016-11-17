local nn = require 'nn'
local torch = require 'torch'
local test_util = require 'tests/test_util'
require 'layers/PeriodicResidualTable'

local function test_single_block()
    local init = nn.MulConstant(1)
    local residual = nn.MulConstant(2)
    local periodic = nn.PeriodicResidualTable(4, init, residual)

    local a = {}
    for i = 1, 4 do a[i] = torch.rand(5, 5) end

    local b = periodic:forward(a)
    assert(#b == #a)
    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], 2*a[2]))
    assert(test_util.equals(b[3], 2*a[3]))
    assert(test_util.equals(b[4], 2*a[4]))
end

local function test_reinit()
    local init = nn.MulConstant(1)
    local residual = nn.MulConstant(2)
    local periodic = nn.PeriodicResidualTable(2, init, residual)

    local a = {}
    for i = 1, 4 do a[i] = torch.rand(5, 5) end

    local b = periodic:forward(a)
    assert(#b == #a)
    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], 2*a[2]))
    assert(test_util.equals(b[3], a[3]))
    assert(test_util.equals(b[4], 2*a[4]))
end

local function test_reinit_with_incomplete_block()
    local init = nn.MulConstant(1)
    local residual = nn.MulConstant(2)
    local periodic = nn.PeriodicResidualTable(3, init, residual)

    local a = {}
    for i = 1, 8 do a[i] = torch.rand(5, 5) end

    local b = periodic:forward(a)
    assert(#b == #a)
    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], 2*a[2]))
    assert(test_util.equals(b[3], 2*a[3]))
    assert(test_util.equals(b[4], a[4]))
    assert(test_util.equals(b[5], 2*a[5]))
    assert(test_util.equals(b[6], 2*a[6]))
    assert(test_util.equals(b[7], a[7]))
    assert(test_util.equals(b[8], 2*a[8]))
end

local function test_long_input_vs_short_input()
    local init = nn.MulConstant(1)
    local residual = nn.MulConstant(2)

    local periodic = nn.PeriodicResidualTable(4, init, residual)

    local a = {}
    for i = 1, 8 do a[i] = torch.rand(5, 5) end

    local a_output = periodic:forward(a)
    for i = 1, 8 do a_output[i] = a_output[i]:clone() end

    -- First, ensure that the output is correct for a.
    assert(test_util.equals(a_output[1],   a[1]))
    assert(test_util.equals(a_output[2], 2*a[2]))
    assert(test_util.equals(a_output[3], 2*a[3]))
    assert(test_util.equals(a_output[4], 2*a[4]))
    assert(test_util.equals(a_output[5],   a[5]))
    assert(test_util.equals(a_output[6], 2*a[6]))
    assert(test_util.equals(a_output[7], 2*a[7]))
    assert(test_util.equals(a_output[8], 2*a[8]))

    -- Next, ensure that if we split a into two inputs, we get the same output.
    local b = {a[1], a[2], a[3], a[4]}
    local c = {a[5], a[6], a[7], a[8]}
    local b_output = periodic:forward(b)
    for i = 1, 4 do b_output[i] = b_output[i]:clone() end
    local c_output = periodic:forward(c)
    for i = 1, 4 do c_output[i] = c_output[i]:clone() end

    assert(test_util.almost_equals(b_output[1], a_output[1]))
    assert(test_util.almost_equals(b_output[2], a_output[2]))
    assert(test_util.almost_equals(b_output[3], a_output[3]))
    assert(test_util.almost_equals(b_output[4], a_output[4]))

    assert(test_util.almost_equals(c_output[1], a_output[5]))
    assert(test_util.almost_equals(c_output[2], a_output[6]))
    assert(test_util.almost_equals(c_output[3], a_output[7]))
    assert(test_util.almost_equals(c_output[4], a_output[8]))
end

local function test_clearState()
    local init = nn.MulConstant(1)
    local residual = nn.MulConstant(2)

    local periodic = nn.PeriodicResidualTable(4, init, residual)
    local a = {}
    for i = 1, 8 do a[i] = torch.rand(5, 5) end

    local a_output = periodic:forward(a)
    for i = 1, 8 do a_output[i] = a_output[i]:clone() end

    periodic:clearState()

    local b = {}
    for i = 1, 4 do b[i] = a[i] end
    local b_output = periodic:forward(b)

    assert(test_util.almost_equals(a_output[1], b_output[1]))
    assert(test_util.almost_equals(a_output[2], b_output[2]))
    assert(test_util.almost_equals(a_output[3], b_output[3]))
    assert(test_util.almost_equals(a_output[4], b_output[4]))
end

local function test_load_save()
    local init = nn.MulConstant(1)
    local residual = nn.MulConstant(2)
    local periodic = nn.PeriodicResidualTable(4, init, residual)

    local a = {}
    for i = 1, 8 do a[i] = torch.rand(5, 5) end

    local a_output = periodic:forward(a)
    for i = 1, 8 do a_output[i] = a_output[i]:clone() end

    periodic:clearState()

    local f = torch.MemoryFile()
    f:writeObject(periodic)
    f:seek(1)

    periodic = f:readObject()
    local a_output_after_load = periodic:forward(a)
    for i = 1, 8 do a_output_after_load[i] = a_output_after_load[i]:clone() end

    assert(test_util.almost_equals(a_output[1], a_output_after_load[1]))
    assert(test_util.almost_equals(a_output[2], a_output_after_load[2]))
    assert(test_util.almost_equals(a_output[3], a_output_after_load[3]))
    assert(test_util.almost_equals(a_output[4], a_output_after_load[4]))
    assert(test_util.almost_equals(a_output[5], a_output_after_load[5]))
    assert(test_util.almost_equals(a_output[6], a_output_after_load[6]))
    assert(test_util.almost_equals(a_output[7], a_output_after_load[7]))
    assert(test_util.almost_equals(a_output[8], a_output_after_load[8]))
end

function test_reset_slower_reinit()
    local init = nn.MulConstant(1)
    local residual = nn.MulConstant(2)
    local periodic = nn.PeriodicResidualTable(2, init, residual)

    local a = {}
    for i = 1, 4 do a[i] = torch.rand(5, 5) end

    local b = periodic:forward(a)
    assert(#b == #a)
    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], 2*a[2]))
    assert(test_util.equals(b[3], a[3]))
    assert(test_util.equals(b[4], 2*a[4]))

    periodic:set_reinitialize_rate(4)
    a = {}
    for i = 1, 8 do a[i] = torch.rand(5, 5) end

    b = periodic:forward(a)
    assert(#b == #a)

    assert(test_util.equals(b[1],   a[1]))
    assert(test_util.equals(b[2], 2*a[2]))
    assert(test_util.equals(b[3], 2*a[3]))
    assert(test_util.equals(b[4], 2*a[4]))
    assert(test_util.equals(b[5],   a[5]))
    assert(test_util.equals(b[6], 2*a[6]))
    assert(test_util.equals(b[7], 2*a[7]))
    assert(test_util.equals(b[8], 2*a[8]))
end

function test_reset_faster_reinit()
    local init = nn.MulConstant(1)
    local residual = nn.MulConstant(2)
    local periodic = nn.PeriodicResidualTable(4, init, residual)

    a = {}
    for i = 1, 8 do a[i] = torch.rand(5, 5) end

    b = periodic:forward(a)
    assert(#b == #a)

    assert(test_util.equals(b[1],   a[1]))
    assert(test_util.equals(b[2], 2*a[2]))
    assert(test_util.equals(b[3], 2*a[3]))
    assert(test_util.equals(b[4], 2*a[4]))
    assert(test_util.equals(b[5],   a[5]))
    assert(test_util.equals(b[6], 2*a[6]))
    assert(test_util.equals(b[7], 2*a[7]))
    assert(test_util.equals(b[8], 2*a[8]))
    periodic:set_reinitialize_rate(2)

    periodic:set_reinitialize_rate(2)
    a = {}
    for i = 1, 4 do a[i] = torch.rand(5, 5) end

    b = periodic:forward(a)
    assert(#b == #a)
    assert(test_util.equals(b[1],   a[1]))
    assert(test_util.equals(b[2], 2*a[2]))
    assert(test_util.equals(b[3],   a[3]))
    assert(test_util.equals(b[4], 2*a[4]))
end

test_util.run_test(test_single_block, 'Single block')
test_util.run_test(test_reinit, 'Reinit')
test_util.run_test(test_reinit_with_incomplete_block,
                   'Reinit with incomplete block')
test_util.run_test(test_long_input_vs_short_input,
                   'Long sequences vs. multiple short sequences')
test_util.run_test(test_clearState, 'Clear state')
test_util.run_test(test_load_save, 'Test loading after saving')
test_util.run_test(test_reset_slower_reinit,
                   'Reset to slower reinitialize rate')
test_util.run_test(test_reset_faster_reinit, 'Reset to faster reinit rate')
