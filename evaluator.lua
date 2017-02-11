local classic = require 'classic'
local cudnn = require 'cudnn'
local cunn = require 'cunn'
local cutorch = require 'cutorch'
local torch = require 'torch'

local image_util = require 'util/image_util'
local log = require 'util/log'
local END_OF_SEQUENCE = require('data_loader').END_OF_SEQUENCE

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
        local label_groundtruth = groundtruth[{{}, label}]
        if torch.any(label_groundtruth) then
            label_has_sample[label] = 1
            local label_predictions = predictions[{{}, label}]
            local _, sorted_indices = torch.sort(
                label_predictions, 1, true --[[descending]])
            local true_positives = 0
            local average_precision = 0

            local sorted_groundtruth = label_groundtruth:index(
                1, sorted_indices):float()
            local true_positives = torch.cumsum(sorted_groundtruth)
            local num_guesses = torch.range(1, label_predictions:nElement())
            local precisions = torch.cdiv(true_positives, num_guesses)
            precisions = precisions[torch.eq(sorted_groundtruth, 1)]
            average_precisions[label] = precisions:mean()
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
