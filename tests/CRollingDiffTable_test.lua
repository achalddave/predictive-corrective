local nn = require 'nn'
local torch = require 'torch'
require 'layers/CRollingDiffTable'

local function test_not_enough_inputs()
    local differ = nn.CRollingDiffTable(3)
    local a = {torch.rand(5, 5)}
    local status, err = pcall(function () differ:forward(a) end)
    assert(status == false)
end

local function test_single_diff()
    local differ = nn.CRollingDiffTable(2)
    local a = {}
    for i = 1, 2 do a[i] = torch.rand(5, 5) end
    local b = differ:forward(a)
    assert(#b == #a)
    assert(torch.all(torch.eq(b[1], a[1])))
    assert(torch.all(torch.eq(b[2], a[2] - a[1])))
end

local function test_reinit()
    local differ = nn.CRollingDiffTable(2)
    local a = {}
    for i = 1, 4 do a[i] = torch.rand(5, 5) end
    local b = differ:forward(a)
    assert(#b == #a)
    assert(torch.all(torch.eq(b[1], a[1])))
    assert(torch.all(torch.eq(b[2], a[2] - a[1])))
    assert(torch.all(torch.eq(b[3], a[3])))
    assert(torch.all(torch.eq(b[4], a[4] - a[3])))
end

local function test_small_input_after_larger_input()
    local differ = nn.CRollingDiffTable(2)
    local a = {}
    for i = 1, 6 do a[i] = torch.rand(5, 5) end
    local b = differ:forward(a)
    assert(#b == #a)
    assert(torch.all(torch.eq(b[1], a[1])))
    assert(torch.all(torch.eq(b[2], a[2] - a[1])))
    assert(torch.all(torch.eq(b[3], a[3])))
    assert(torch.all(torch.eq(b[4], a[4] - a[3])))
    assert(torch.all(torch.eq(b[5], a[5])))
    assert(torch.all(torch.eq(b[6], a[6] - a[5])))

    a = {a[1], a[2]}
    b = differ:forward(a)
    assert(#b == #a)
    assert(torch.all(torch.eq(b[1], a[1])))
    assert(torch.all(torch.eq(b[2], a[2] - a[1])))
end

test_not_enough_inputs()
test_single_diff()
test_reinit()
test_small_input_after_larger_input()
