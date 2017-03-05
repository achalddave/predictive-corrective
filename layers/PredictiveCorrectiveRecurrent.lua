local nn = require 'nn'
local rnn = require 'rnn'
local torch = require 'torch'

local InitUpdateRecurrent = require 'layers/InitUpdateRecurrent'

local PredictiveCorrectiveRecurrent, parent = torch.class(
    'nn.PredictiveCorrectiveRecurrent', 'nn.Sequential')

-- TODO(achald): Test this!
function PredictiveCorrectiveRecurrent:__init(
        init, update, input_size, output_size, num_input_dim, rho)
    parent.__init(self)

    -- Create a recurrence module where
    --     output(t) = {input(t), input(t-1)}
    -- Therefore, the hidden state at time t is {input(t-1), input(t-2)}.

    -- Given input {a, {b, c}}, outputs {a, b}.
    -- In our case, a, b, c are the inputs at time t, t-1, and t-2 respectively.
    local pair_selector = nn.ConcatTable()
    pair_selector:add(nn.SelectTable(1)) -- Select a
    pair_selector:add(nn.Sequential()
        :add(nn.SelectTable(2)) -- Select {b, c}
        :add(nn.SelectTable(1))) -- Select b

    local input_pairs_recurrence = nn.Recurrence(
        pair_selector,
        {input_size, input_size} --[[outputSize]],
        num_input_dim --[[nInputDim]],
        rho)
    local differencer = nn.Sequential()
    differencer:add(input_pairs_recurrence):add(nn.CSubTable())

    self.modules = {
        differencer,
        nn.InitUpdateRecurrent(init, update, rho),
        nn.Recurrence(nn.CAddTable(),
                      output_size --[[outputSize]],
                      num_input_dim --[[nInputDim]],
                      rho)
    }
end
