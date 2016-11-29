local argparse = require 'argparse'

local parser = argparse() {description='Sum tensors and save them'}
parser:option('--input'):count('*')
parser:option('--output')
local args = parser:parse()

local a = nil
for _, t7_file in ipairs(args.input) do
    if a == nil then
        a = torch.load(t7_file)
    else
        a = a + torch.load(t7_file)
    end
end
torch.save(args.output, a)
