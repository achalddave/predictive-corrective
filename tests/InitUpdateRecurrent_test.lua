local nn = require 'nn'
local rnn = require 'rnn'
local torch = require 'torch'
local test_util = require 'tests/test_util'
local __ = require 'moses'
require 'layers/InitUpdateRecurrent'

local function test_init_only()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local model = nn.InitUpdateRecurrent(init, update, 1)

    local a = {}
    for i = 1, 4 do a[i] = torch.rand(5, 5) end

    local b = {}
    for i = 1, 4 do
        b[i] = model:forward(a[i]):clone()
        model:forget()
    end
    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], a[2]))
    assert(test_util.equals(b[3], a[3]))
    assert(test_util.equals(b[4], a[4]))
end

local function test_updates()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local model = nn.InitUpdateRecurrent(init, update, 1)

    local a = {}
    for i = 1, 6 do a[i] = torch.rand(5, 5) end

    local b = {}
    b[1] = model:forward(a[1]):clone()
    b[2] = model:forward(a[2]):clone()
    b[3] = model:forward(a[3]):clone()
    model:forget()
    b[4] = model:forward(a[4]):clone()
    b[5] = model:forward(a[5]):clone()
    b[6] = model:forward(a[6]):clone()

    assert(test_util.equals(b[1], a[1]))
    assert(test_util.equals(b[2], 2*a[2]))
    assert(test_util.equals(b[3], 2*a[3]))
    assert(test_util.equals(b[4], a[4]))
    assert(test_util.equals(b[5], 2*a[5]))
    assert(test_util.equals(b[6], 2*a[6]))
end

local function test_check_sharedClones_table_sanity()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local model = nn.InitUpdateRecurrent(init, update, 2)
    assert(model.sharedClones[1] == nil)

    model:forward(torch.rand(5, 5))
    model:forward(torch.rand(5, 5))
    model:forward(torch.rand(5, 5))
    model:forward(torch.rand(5, 5))
    assert(__.size(model.sharedClones) == 3)
    assert(model.sharedClones[1] == nil)
    assert(model.sharedClones[2] ~= nil)
    assert(model.sharedClones[3] ~= nil)
    assert(model.sharedClones[4] ~= nil)

    model:forward(torch.rand(5, 5))
    model:forward(torch.rand(5, 5))
    model:forward(torch.rand(5, 5))
    assert(model.sharedClones[1] == nil)
    assert(model.sharedClones[2] == nil)
    assert(model.sharedClones[3] == nil)
    assert(model.sharedClones[4] == nil)
    assert(model.sharedClones[5] ~= nil)
    assert(model.sharedClones[6] ~= nil)
    assert(model.sharedClones[7] ~= nil)
    assert(__.size(model.sharedClones) == 3)

    model:forget()
    assert(model.sharedClones[1] == nil)
    assert(model.sharedClones[2] ~= nil)
    assert(model.sharedClones[3] ~= nil)
    assert(model.sharedClones[4] ~= nil)
    assert(model.sharedClones[5] == nil)
    assert(__.size(model.sharedClones) == 3)
end

local function test_load_save()
    local init = nn.MulConstant(1)
    local update = nn.MulConstant(2)
    local model = nn.InitUpdateRecurrent(init, update, 1)

    local a = {}
    for i = 1, 8 do a[i] = torch.rand(5, 5) end

    local a_output = {}
    for i = 1, 8 do a_output[i] = model:forward(a[i]):clone() end

    model:clearState()

    local f = torch.MemoryFile()
    f:writeObject(model)
    f:seek(1)

    model = f:readObject()
    local a_output_after_load = {}
    for i = 1, 8 do a_output_after_load[i] = model:forward(a[i]):clone() end

    assert(test_util.almost_equals(a_output[1], a_output_after_load[1]))
    assert(test_util.almost_equals(a_output[2], a_output_after_load[2]))
    assert(test_util.almost_equals(a_output[3], a_output_after_load[3]))
    assert(test_util.almost_equals(a_output[4], a_output_after_load[4]))
    assert(test_util.almost_equals(a_output[5], a_output_after_load[5]))
    assert(test_util.almost_equals(a_output[6], a_output_after_load[6]))
    assert(test_util.almost_equals(a_output[7], a_output_after_load[7]))
    assert(test_util.almost_equals(a_output[8], a_output_after_load[8]))
end

test_util.run_test(test_init_only, 'Init only')
test_util.run_test(test_updates, 'Updates')
test_util.run_test(test_check_sharedClones_table_sanity, 'sharedClones sanity')
test_util.run_test(test_load_save, 'Test loading after saving')
