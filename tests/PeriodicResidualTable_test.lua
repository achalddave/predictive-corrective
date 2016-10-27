local nn = require 'nn'
local torch = require 'torch'
require 'layers/PeriodicResidualTable'

local function test_single_block()
    local init = nn.MulConstant(1)
    local residual = nn.MulConstant(2)
    local periodic = nn.PeriodicResidualTable(4, init, residual)

    local a = {}
    for i = 1, 4 do a[i] = torch.rand(5, 5) end

    local b = periodic:forward(a)
    assert(#b == #a)
    assert(torch.all(torch.eq(b[1], a[1])))
    assert(torch.all(torch.eq(b[2], 2*a[2])))
    assert(torch.all(torch.eq(b[3], 2*a[3])))
    assert(torch.all(torch.eq(b[4], 2*a[4])))
end

local function test_reinit()
    local init = nn.MulConstant(1)
    local residual = nn.MulConstant(2)
    local periodic = nn.PeriodicResidualTable(2, init, residual)

    local a = {}
    for i = 1, 4 do a[i] = torch.rand(5, 5) end

    local b = periodic:forward(a)
    assert(#b == #a)
    assert(torch.all(torch.eq(b[1], a[1])))
    assert(torch.all(torch.eq(b[2], 2*a[2])))
    assert(torch.all(torch.eq(b[3], a[3])))
    assert(torch.all(torch.eq(b[4], 2*a[4])))
end

local function test_reinit_with_incomplete_block()
    local init = nn.MulConstant(1)
    local residual = nn.MulConstant(2)
    local periodic = nn.PeriodicResidualTable(3, init, residual)

    local a = {}
    for i = 1, 8 do a[i] = torch.rand(5, 5) end

    local b = periodic:forward(a)
    assert(#b == #a)
    assert(torch.all(torch.eq(b[1], a[1])))
    assert(torch.all(torch.eq(b[2], 2*a[2])))
    assert(torch.all(torch.eq(b[3], 2*a[3])))
    assert(torch.all(torch.eq(b[4], a[4])))
    assert(torch.all(torch.eq(b[5], 2*a[5])))
    assert(torch.all(torch.eq(b[6], 2*a[6])))
    assert(torch.all(torch.eq(b[7], a[7])))
    assert(torch.all(torch.eq(b[8], 2*a[8])))
end

test_single_block()
test_reinit()
test_reinit_with_incomplete_block()
