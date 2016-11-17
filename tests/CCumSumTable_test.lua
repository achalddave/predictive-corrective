local nn = require 'nn'
local torch = require 'torch'
local test_util = require 'tests/test_util'
require 'layers/CCumSumTable'

local function test_single_diff()
    local summer = nn.CCumSumTable(2)
    local a = {}
    for i = 1, 2 do a[i] = torch.rand(5, 5) end
    local b = summer:forward(a)
    assert(#b == #a)
    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], a[2] + a[1]))
end

local function test_reinit()
    local summer = nn.CCumSumTable(2)
    local a = {}
    for i = 1, 4 do a[i] = torch.rand(5, 5) end
    local b = summer:forward(a)
    assert(#b == #a)
    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], a[2] + a[1]))
    assert(test_util.equals(b[3], a[3]))
    assert(test_util.equals(b[4], a[4] + a[3]))
end

local function test_small_input_after_larger_input()
    local summer = nn.CCumSumTable(3)
    local a = {}
    for i = 1, 6 do a[i] = torch.rand(5, 5) end
    local b = summer:forward(a)
    assert(#b == #a)
    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], a[1] + a[2]))
    assert(test_util.equals(b[3], a[1] + a[2] + a[3]))
    assert(test_util.equals(b[4], a[4]))
    assert(test_util.equals(b[5], a[4] + a[5]))
    assert(test_util.equals(b[6], a[4] + a[5] + a[6]))

    a = {a[1], a[2]}
    b = summer:forward(a)
    assert(#b == #a)
    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], a[2] + a[1]))
end

local function test_long_input_vs_short_input()
    local summer = nn.CCumSumTable(4)

    local a = {}
    for i = 1, 8 do a[i] = torch.rand(5, 5) end

    local a_output = summer:forward(a)
    for i = 1, 8 do a_output[i] = a_output[i]:clone() end

    -- First, ensure that the output is correct for a.
    assert(test_util.equals(a_output[1], a[1]))
    assert(test_util.equals(a_output[2], a[1] + a[2]))
    assert(test_util.equals(a_output[3], a[1] + a[2] + a[3]))
    assert(test_util.equals(a_output[4], a[1] + a[2] + a[3] + a[4]))
    assert(test_util.equals(a_output[5], a[5]))
    assert(test_util.equals(a_output[6], a[5] + a[6]))
    assert(test_util.equals(a_output[7], a[5] + a[6] + a[7]))
    assert(test_util.equals(a_output[8], a[5] + a[6] + a[7] + a[8]))

    -- Next, ensure that if we split a into two inputs, we get the same output.
    local b = {a[1], a[2], a[3], a[4]}
    local c = {a[5], a[6], a[7], a[8]}
    local b_output = summer:forward(b)
    for i = 1, 4 do b_output[i] = b_output[i]:clone() end
    local c_output = summer:forward(c)
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
    local summer = nn.CCumSumTable(4)
    local a = {}
    for i = 1, 8 do a[i] = torch.rand(5, 5) end

    local a_output = summer:forward(a)
    for i = 1, 8 do a_output[i] = a_output[i]:clone() end

    summer:clearState()

    local b = {}
    for i = 1, 4 do b[i] = a[i] end
    local b_output = summer:forward(b)

    assert(test_util.almost_equals(a_output[1], b_output[1]))
    assert(test_util.almost_equals(a_output[2], b_output[2]))
    assert(test_util.almost_equals(a_output[3], b_output[3]))
    assert(test_util.almost_equals(a_output[4], b_output[4]))
end

local function test_reset_slower_reinit()
    local summer = nn.CCumSumTable(2)
    local a = {}
    for i = 1, 4 do a[i] = torch.rand(5, 5) end

    local b = summer:forward(a)
    assert(#b == #a)
    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], a[2] + a[1]))
    assert(test_util.equals(b[3], a[3]))
    assert(test_util.equals(b[4], a[4] + a[3]))

    summer:set_reinitialize_rate(4)
    a = {}
    for i = 1, 8 do a[i] = torch.rand(5, 5) end
    b = summer:forward(a)
    assert(#b == #a)
    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], a[1] + a[2]))
    assert(test_util.equals(b[3], a[1] + a[2] + a[3]))
    assert(test_util.equals(b[4], a[1] + a[2] + a[3] + a[4]))
    assert(test_util.equals(b[5], a[5]))
    assert(test_util.equals(b[6], a[5] + a[6]))
    assert(test_util.equals(b[7], a[5] + a[6] + a[7]))
    assert(test_util.equals(b[8], a[5] + a[6] + a[7] + a[8]))
end

local function test_reset_faster_reinit()
    local summer = nn.CCumSumTable(4)
    local a = {}
    for i = 1, 8 do a[i] = torch.rand(5, 5) end
    local b = summer:forward(a)
    assert(#b == #a)
    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], a[1] + a[2]))
    assert(test_util.equals(b[3], a[1] + a[2] + a[3]))
    assert(test_util.equals(b[4], a[1] + a[2] + a[3] + a[4]))
    assert(test_util.equals(b[5], a[5]))
    assert(test_util.equals(b[6], a[5] + a[6]))
    assert(test_util.equals(b[7], a[5] + a[6] + a[7]))
    assert(test_util.equals(b[8], a[5] + a[6] + a[7] + a[8]))

    summer:set_reinitialize_rate(2)
    a = {}
    for i = 1, 4 do a[i] = torch.rand(5, 5) end
    b = summer:forward(a)
    assert(#b == #a)
    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], a[2] + a[1]))
    assert(test_util.equals(b[3], a[3]))
    assert(test_util.equals(b[4], a[4] + a[3]))
end
test_util.run_test(test_single_diff, 'Single summerence')
test_util.run_test(test_reinit, 'Reinit')
test_util.run_test(test_small_input_after_larger_input,
                   'Small input after larger input')
test_util.run_test(test_long_input_vs_short_input,
                   'Long input vs multiple short inputs')
test_util.run_test(test_clearState, 'Clear state')
test_util.run_test(test_reset_slower_reinit,
                   'Reset to slower reinitialize rate')
test_util.run_test(test_reset_faster_reinit, 'Reset to faster reinit rate')
