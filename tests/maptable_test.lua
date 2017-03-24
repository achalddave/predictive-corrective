local nn = require 'nn'
local torch = require 'torch'
local test_util = require 'tests/test_util'

local function test_load_save()
    local model = nn.MapTable(nn.Sequential():add(nn.Dropout()))

    local f = torch.MemoryFile()
    f:writeObject(model)
    f:seek(1)

    model = f:readObject()
    assert(model.modules[1] == model.module)
end

local function test_load_save_copy()
    local model = nn.MapTable(nn.Sequential():add(nn.Dropout()))

    local f = torch.MemoryFile()
    local new_model = model:clone()
    new_model:clearState()
    f:writeObject(new_model)
    f:seek(1)

    model = f:readObject()
    assert(model.modules[1] == model.module)
end

test_util.run_test(test_load_save, 'Load save')
test_util.run_test(test_load_save_copy, 'Load save copy')
