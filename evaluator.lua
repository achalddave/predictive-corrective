local class = require 'class'
local cudnn = require 'cudnn'
local cunn = require 'cunn'
local cutorch = require 'cutorch'
local image = require 'image'

local Evaluator = class('Evaluator')

function Evaluator:__init(args)
    --[[
    Used to evaluate the model on validation data.

    Args:
        model
        criterion
        data_loader
        pixel_mean
        batch_size
        crop_size
        num_labels
    ]]--
    self.model = args.model
    self.criterion = args.criterion
    self.data_loader = args.data_loader
    self.pixel_mean = torch.Tensor(args.pixel_mean)
    self.batch_size = args.batch_size
    self.crop_size = args.crop_size
    self.num_labels = args.num_labels

    self.gpu_inputs = torch.CudaTensor()
    self.gpu_labels = torch.CudaTensor()
end

function Evaluator:evaluate_batch()
    --[[
    Returns:
        loss: Output of criterion:forward on this batch.
        outputs: Output of model:forward on this batch.
        labels: True labels.
    ]]--
    local images_table, labels_table = self.data_loader:load_batch(
        self.batch_size)
    local images = torch.Tensor(#images_table, images_table[1]:size(1),
                                self.crop_size, self.crop_size)
    local labels = torch.ByteTensor(#labels_table, self.num_labels)
    for i, img in ipairs(images_table) do
        images[i] = self:_process(img:typeAs(images))
        labels[i] = self.data_loader:labels_to_tensor(
            labels_table[i], self.num_labels)
    end

    self.gpu_inputs:resize(images:size()):copy(images)
    self.gpu_labels:resize(labels:size()):copy(labels)
    local outputs = self.model:forward(self.gpu_inputs)
    local loss = self.criterion:forward(outputs, self.gpu_labels)

    return loss, outputs, labels
end

function Evaluator:evaluate_epoch(epoch, num_batches)
    self.model:evaluate()
    local epoch_timer = torch.Timer()
    local batch_timer = torch.Timer()

    local predictions = torch.Tensor(
        num_batches * self.batch_size, self.num_labels)
    local groundtruth = torch.ByteTensor(
        num_batches * self.batch_size, self.num_labels)

    local loss_epoch = 0
    for batch_index = 1, num_batches do
        batch_timer: reset()
        collectgarbage()
        cutorch.synchronize()
        local loss, curr_predictions, curr_groundtruth = self:evaluate_batch()
        cutorch.synchronize()

        -- Collect current predictions and groundtruth.
        local epoch_index_start = (batch_index - 1) * self.batch_size + 1
        predictions[{{epoch_index_start,
                      epoch_index_start + self.batch_size - 1},
                      {}}] = curr_predictions:type(predictions:type())
        groundtruth[{{epoch_index_start,
                      epoch_index_start + self.batch_size - 1},
                      {}}] = curr_groundtruth

        loss_epoch = loss_epoch + loss
    end
    local mean_average_precision = compute_mean_average_precision(
        predictions, groundtruth)
    print(string.format(
        '%s: Epoch: [%d][VALIDATION SUMMARY] Total Time(s): %.2f\t' ..
        'average loss (per batch): %.5f \t mAP: %.5f',
        os.date('%X'), epoch, epoch_timer:time().real, loss_epoch / num_batches,
        mean_average_precision))
    collectgarbage()
end

function Evaluator:_process(img)
    -- Avoid wrap around for ByteTensors, which are unsigned.
    assert(img:type() ~= torch.ByteTensor():type())

    -- Take center crop.
    img = image.crop(img, "c" --[[center crop]], self.crop_size, self.crop_size)
    assert(img:size(3) == self.crop_size)
    assert(img:size(2) == self.crop_size)

    for channel = 1, 3 do
        -- Subtract mean
        if self.pixel_mean then
            img[{{channel}, {}, {}}]:add(-self.pixel_mean[channel])
        end
        -- Divide by std.
        -- TODO(achald): Figure out if this is necessary; if so, implement it.
    end

    return img
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
            for num_guesses = 1, label_predictions:nElement() do
                local sample_index = sorted_indices[num_guesses]
                if label_groundtruth[sample_index] == 1 then
                    true_positives = true_positives + 1
                    local precision = true_positives / num_guesses
                    average_precision = average_precision + precision
                end
            end
            average_precisions[label] =
                average_precision / label_groundtruth:sum()
        end
    end
    -- Return mean of average precisions for labels which had at least 1 sample
    -- in the provided data.
    average_precisions = average_precisions[torch.eq(label_has_sample, 1)]
    if average_precisions:nElement() == 0 then
        print('No positive labels! Returning -1.')
        return -1
    end
    return torch.mean(average_precisions)
end

return {
    Evaluator = Evaluator,
    compute_mean_average_precision = compute_mean_average_precision
}
