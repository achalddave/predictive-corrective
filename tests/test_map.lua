package.path = package.path .. ";../?.lua"

local torch = require 'torch'

local evaluate = require 'evaluate'

function test_map1()
    local scores = torch.range(10, 1, -1):resize(10, 1)
    local labels = torch.ByteTensor(
        {1, 0, 1, 1, 1, 1, 0, 0, 0, 1}):resizeAs(scores:byte())
    ap = evaluate.compute_mean_average_precision(scores, labels)
    ap_true = (1 + 2/3 + 3/4 + 4/5 + 5/6 + 6/10) / 6
    assert(ap == ap_true,
           string.format('Expected %.5f, received %.5f', ap_true, ap))
end

function test_map2()
    local scores = torch.range(10, 1, -1):resize(10, 1)
    local labels = torch.ByteTensor(
        {1, 0, 0, 0, 1, 1, 0, 0, 0, 1}):resizeAs(scores:byte())
    ap = evaluate.compute_mean_average_precision(scores, labels)
    ap_true = (1 + 2/5 + 3/6 + 4/10) / 4
    assert(ap == ap_true,
           string.format('Expected %.5f, received %.5f', ap_true, ap))
end

function test_map3()
    local scores = torch.cat(torch.range(10, 1, -1), torch.range(10, 1, -1), 2)
    local labels = torch.ByteTensor(
        {{1, 0, 0, 0, 1, 1, 0, 0, 0, 1},
         {1, 0, 0, 0, 1, 1, 0, 0 ,0, 1}}):t()
    ap = evaluate.compute_mean_average_precision(scores, labels)
    ap_true = (1 + 2/5 + 3/6 + 4/10) / 4
    assert(ap == ap_true,
           string.format('Expected %.5f, received %.5f', ap_true, ap))
end

test_map1()
test_map2()
test_map3()
