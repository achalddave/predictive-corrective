local function equals(a, b)
    return torch.all(torch.eq(a, b))
end

local function almost_equals(a, b, threshold)
    if threshold == nil then threshold = 1e-10 end
    return torch.all(torch.lt(torch.abs(a - b), threshold))
end

function run_test(test_fn, test_name)
    print(string.format('Running test: %s', test_name))
    test_fn()
    print(string.format('Success!'))
end

return {
    equals = equals,
    almost_equals = almost_equals,
    run_test = run_test
}
