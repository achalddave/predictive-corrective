local torch = require 'torch'
local log = require 'util/log'

local function compute_average_precision(predictions, groundtruth)
    --[[
    Compute mean average prediction.
    From
    https://github.com/achalddave/average-precision/blob/e9edd7ef64f9d5f236cf2cf411627c234369eb72/lua/ap_torch.lua

    TODO(achald): Add average-precision as a submodule so it stays updated.

    Args:
        predictions ((num_samples) Tensor)
        groundtruth ((num_samples) Tensor): Contains 0/1 values.

    Returns:
        average_precision (num)
    ]]--
    predictions = predictions:float()
    groundtruth = groundtruth:byte()

    --[[
    Let P(k) be the precision at cut-off for item k. Then, we compute average
    precision for each label as

    \frac{ \sum_{k=1}^n (P(k) * is_positive(k)) }{ # of relevant documents }

    where is_positive(k) is 1 if the groundtruth labeled item k as positive.
    ]]--
    if not torch.any(groundtruth) then
        return 0
    end
    local _, sorted_indices = torch.sort(predictions, 1, true --[[descending]])
    local true_positives = 0
    local average_precision = 0

    local sorted_groundtruth = groundtruth:index(1, sorted_indices):float()

    local true_positives = torch.cumsum(sorted_groundtruth)
    local false_positives = torch.cumsum(1 - sorted_groundtruth)
    local num_positives = true_positives[-1]

    local precisions = torch.cdiv(
        true_positives,
        torch.cmax(true_positives + false_positives, 1e-16))
    local recalls = true_positives / num_positives

    -- Set precisions[i] = max(precisions[j] for j >= i)
    -- This is because (for j > i), recall[j] >= recall[i], so we can
    -- always use a lower threshold to get the higher recall and higher
    -- precision at j.
    for i = precisions:nElement()-1, 1, -1 do
        precisions[i] = math.max(precisions[i], precisions[i+1])
    end

    -- Append end points of the precision recall curve.
    local zero = torch.zeros(1):float()
    local one = torch.ones(1):float()
    precisions = torch.cat({zero, precisions, zero}, 1)
    recalls = torch.cat({zero, recalls, one})

    -- Find points where recall changes.
    local changes = torch.ne(recalls[{{2, -1}}], recalls[{{1, -2}}])
    local changes_plus_1 = torch.cat({torch.zeros(1):byte(), changes})
    changes = torch.cat({changes, torch.zeros(1):byte()})

    return torch.cmul((recalls[changes_plus_1] - recalls[changes]),
                      precisions[changes_plus_1]):sum()
end

-- TODO(achald): Move this to a separate util file.
function compute_mean_average_precision(predictions, groundtruth)
    --[[
    Compute mean average prediction.

    Args:
        predictions ((num_samples, num_classes) Tensor)
        groundtruth ((num_samples, num_classes) Tensor): Contains 0/1 values.

    Returns:
        mean_average_precision (num)
    ]]--
    -- TODO: Make these be the 'default' tensor, not float necessarily.
    predictions = predictions:float()
    local num_labels = predictions:size(2)
    local average_precisions = torch.Tensor(num_labels):zero()
    local label_has_sample = torch.ByteTensor(num_labels):zero()
    --[[
    Let P(k) be the precision at cut-off for item k. Then, we compute average
    precision for each label as

    \frac{ \sum_{k=1}^n (P(k) * is_positive(k)) }{ # of relevant documents }

    where is_positive(k) is 1 if the groundtruth labeled item k as positive.
    ]]--
    for label = 1, num_labels do
        if torch.any(groundtruth[{{}, label}]) then
            label_has_sample[label] = 1
            average_precisions[label] = compute_average_precision(
                predictions[{{}, label}], groundtruth[{{}, label}])
        end
    end
    -- Return mean of average precisions for labels which had at least 1 sample
    -- in the provided data.
    average_precisions = average_precisions[torch.eq(label_has_sample, 1)]
    if average_precisions:nElement() == 0 then
        log.warn('No positive labels! Returning -1.')
        return -1
    end
    return torch.mean(average_precisions)
end

return {
    compute_mean_average_precision = compute_mean_average_precision
}
