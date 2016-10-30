local classic = require 'classic'
local cudnn = require 'cudnn'
local cunn = require 'cunn'
local cutorch = require 'cutorch'
local torch = require 'torch'

local image_util = require 'util/image_util'
local END_OF_SEQUENCE = require('data_loader').END_OF_SEQUENCE

local Evaluator = classic.class('Evaluator')

function Evaluator:_init(args)
    --[[
    Used to evaluate the model on validation data. See trainer.lua for
    documentation of arguments.

    Args:
        model
        criterion
        data_loader
        input_dimension_permutation
        pixel_mean
        batch_size
        crop_size
        num_labels
    ]]--
    self.model = args.model
    self.criterion = args.criterion
    self.data_loader = args.data_loader
    -- Only use input permutation if it is not the identity.
    self.input_dimension_permutation = nil
    for i = 1, 5 do
        if args.input_dimension_permutation ~= nil
                and args.input_dimension_permutation[i] ~= i then
            self.input_dimension_permutation = args.input_dimension_permutation
            break
        end
    end
    self.pixel_mean = torch.Tensor(args.pixel_mean)
    self.batch_size = args.batch_size
    self.crop_size = args.crop_size
    self.num_labels = args.num_labels

    self.gpu_inputs = torch.CudaTensor()
    self.gpu_labels = torch.CudaTensor()

    -- Prefetch the next batch.
    self.data_loader:fetch_batch_async(self.batch_size)
end

function Evaluator:evaluate_batch()
    --[[
    Returns:
        loss: Output of criterion:forward on this batch.
        outputs (Tensor): Output of model:forward on this batch. The tensor
            size is (sequence_length, batch_size, num_labels)
        labels (Tensor): True labels. Same size as the outputs.
    ]]--
    local images_table, labels = self.data_loader:load_batch(self.batch_size)
    -- Prefetch the next batch.
    self.data_loader:fetch_batch_async(self.batch_size)

    local num_steps = #images_table
    local num_channels = images_table[1][1]:size(1)
    local images = torch.Tensor(num_steps, self.batch_size, num_channels,
                                self.crop_size, self.crop_size)
    for step, step_images in ipairs(images_table) do
        for sequence, img in ipairs(step_images) do
            -- Process image after converting to the default Tensor type.
            -- (Originally, it is a ByteTensor).
            images[{step, sequence}] = image_util.augment_image_eval(
                img:typeAs(images), self.crop_size, self.crop_size,
                self.pixel_mean)
        end
    end
    if self.input_dimension_permutation then
        images = images:permute(unpack(self.input_dimension_permutation))
    end

    self.gpu_inputs:resize(images:size()):copy(images)
    self.gpu_labels:resize(labels:size()):copy(labels)
    local outputs = self.model:forward(self.gpu_inputs)
    -- If the output of the network is a single prediction for the sequence,
    -- compare it to the label of the last frame.
    if (outputs:size(1) == 1 or outputs:dim() == 2) and
            self.gpu_labels:size(1) ~= 1 then
        self.gpu_labels = self.gpu_labels[self.gpu_labels:size(1)]
    end
    local loss = self.criterion:forward(outputs, self.gpu_labels)

    self.gpu_inputs:resize(0)
    self.gpu_labels:resize(0)
    return loss, outputs, labels
end

function Evaluator:evaluate_epoch(epoch, num_batches)
    local epoch_timer = torch.Timer()
    local batch_timer = torch.Timer()
    self.model:evaluate()

    local predictions = torch.Tensor(
        num_batches * self.batch_size, self.num_labels)
    local groundtruth = torch.ByteTensor(
        num_batches * self.batch_size, self.num_labels)

    local loss_epoch = 0
    for batch_index = 1, num_batches do
        batch_timer: reset()
        collectgarbage()
        local loss, curr_predictions, curr_groundtruth = self:evaluate_batch()
        loss_epoch = loss_epoch + loss

        -- We only care about the predictions and groundtruth in the last step
        -- of the sequence.
        if curr_predictions:dim() == 3 and curr_predictions:size(1) > 1 then
            curr_predictions = curr_predictions[curr_predictions:size(1)]
        end
        if curr_groundtruth:dim() == 3 and curr_groundtruth:size(1) > 1 then
            curr_groundtruth = curr_groundtruth[curr_groundtruth:size(1)]
        end

        -- Collect current predictions and groundtruth.
        local epoch_index_start = (batch_index - 1) * self.batch_size + 1
        predictions[{{epoch_index_start,
                      epoch_index_start + self.batch_size - 1},
                      {}}] = curr_predictions:type(predictions:type())
        groundtruth[{{epoch_index_start,
                      epoch_index_start + self.batch_size - 1},
                      {}}] = curr_groundtruth
    end

    local mean_average_precision = compute_mean_average_precision(
        predictions, groundtruth)
    predictions = nil
    groundtruth = nil
    collectgarbage()
    collectgarbage()

    print(string.format(
        '%s: Epoch: [%d][VALIDATION SUMMARY] Total Time(s): %.2f\t' ..
        'average loss (per batch): %.5f \t mAP: %.5f',
        os.date('%X'), epoch, epoch_timer:time().real, loss_epoch / num_batches,
        mean_average_precision))
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

local SequentialEvaluator, SequentialEvaluatorSuper = classic.class(
    'SequentialEvaluator', Evaluator)
function SequentialEvaluator:_init(args)
    if args.input_dimension_permutation ~= nil then
        for i = 1, #args do
            if args.input_dimension_permutation[i] ~= i then
                error('SequentialEvaluator does not support ' ..
                      'input_dimension_permutation')
            end
        end
    end
    SequentialEvaluatorSuper._init(self, args)
    assert(self.batch_size == 1,
          'Currently, SequentialTrainer only supports batch size = 1. ' ..
          'See the "recurrent_batched_training" branch for some WIP on ' ..
          'allowing the batch size to be greater than 1.')
    assert(self.model:findModules('nn.Sequencer') ~= nil,
           'SequentialEvaluator requires that the input model be decorated ' ..
           'with nn.Sequencer.')
    assert(torch.isTypeOf(self.criterion, 'nn.SequencerCriterion'),
           'SequentialEvaluator expects SequencerCriterion.')
    self.model:remember('both')
end

function SequentialEvaluator:evaluate_batch()
    --[[
    Evaluate on a batch of data.

    Returns:
        loss: Output of criterion:forward on this batch.
        outputs (Tensor): Output of model:forward on this batch. The tensor
            size should be either (sequence_length, 1, num_labels). The
            sequence_length may be shorter at the end of the sequence (if the
            sequence ends before we get enough frames).
        labels (Tensor): True labels. Same size as the outputs.
        sequence_ended (bool): If true, specifies that this batch ends the
            sequence.
    ]]--

    local images_table, labels = self.data_loader:load_batch(1 --[[batch size]])
    if images_table[1][1] == END_OF_SEQUENCE then
        -- The sequence ended at the end of the last batch; reset the model and
        -- start loading the next sequence in the next batch.
        for step = 1, #images_table do
            -- The rest of the batch should be filled with END_OF_SEQUENCe.
            assert(images_table[step][1] == END_OF_SEQUENCE)
        end
        self.model:forget()
        return nil, nil, nil, true --[[sequence_ended]]
    end
    -- Prefetch the next batch.
    self.data_loader:fetch_batch_async(1 --[[batch size]])

    local num_steps = #images_table
    local num_channels = images_table[1][1]:size(1)
    local images = torch.Tensor(num_steps, 1 --[[batch size]], num_channels,
                                self.crop_size, self.crop_size)

    local num_valid_steps = num_steps
    for step, step_images in ipairs(images_table) do
        local img = step_images[1]
        if img == END_OF_SEQUENCE then
            -- We're out of frames for this sequence.
            num_valid_steps = step - 1
            break
        else
            -- Process image after converting to the default Tensor type.
            -- (Originally, it is a ByteTensor).
            images[step] = image_util.augment_image_eval(
                img:typeAs(images), self.crop_size, self.crop_size,
                self.pixel_mean)
        end
    end
    local sequence_ended = num_valid_steps ~= num_steps
    if sequence_ended then
        labels = labels[{{1, num_valid_steps}}]
        images = images[{{1, num_valid_steps}}]
        for step = num_valid_steps + 1, #images_table do
            -- The rest of the batch should be filled with END_OF_SEQUENCe.
            assert(images_table[step][1] == END_OF_SEQUENCE)
        end
    end

    self.gpu_inputs:resize(images:size()):copy(images)
    self.gpu_labels:resize(labels:size()):copy(labels)

    -- Should be of shape (sequence_length, batch_size, num_classes)
    local outputs = self.model:forward(self.gpu_inputs)
    local loss = self.criterion:forward(outputs, self.gpu_labels)
    if sequence_ended then
        self.model:forget()
    end
    return loss, outputs, labels, sequence_ended
end

function SequentialEvaluator:evaluate_epoch(epoch, num_sequences)
    self.model:evaluate()
    local epoch_timer = torch.Timer()
    local batch_timer = torch.Timer()

    local predictions, groundtruth

    local epoch_loss = 0
    for _ = 1, num_sequences do
        batch_timer:reset()
        collectgarbage()
        local sequence_ended = false
        local sequence_predictions, sequence_groundtruth
        local sequence_loss = 0
        local num_steps_in_sequence = 0
        while not sequence_ended do
            local loss, batch_predictions, batch_groundtruth, sequence_ended_ =
                self:evaluate_batch()
            -- HACK: Assign to definition outside of while loop.
            sequence_ended = sequence_ended_
            if loss == nil then
                assert(sequence_ended)
                break
            end
            sequence_loss = sequence_loss + loss

            assert(torch.isTensor(batch_predictions))
            -- Remove sequence dimension.
            num_steps_in_sequence = num_steps_in_sequence +
                batch_predictions:size(1)
            batch_predictions = batch_predictions[{{}, 1}]
            batch_groundtruth = batch_groundtruth[{{}, 1}]
            if sequence_predictions == nil then
                sequence_predictions = batch_predictions
                sequence_groundtruth = batch_groundtruth
            else
                sequence_predictions = torch.cat(
                    sequence_predictions, batch_predictions, 1)
                sequence_groundtruth = torch.cat(
                    sequence_groundtruth, batch_groundtruth, 1)
            end
        end
        epoch_loss = epoch_loss + sequence_loss
        if predictions == nil then
            predictions = sequence_predictions
            groundtruth = sequence_groundtruth
        else
            predictions = torch.cat(predictions, sequence_predictions, 1)
            groundtruth = torch.cat(groundtruth, sequence_groundtruth, 1)
        end
    end

    local mean_average_precision = compute_mean_average_precision(
        predictions, groundtruth)
    print(string.format(
        '%s: Epoch: [%d]VALIDATION SUMMARY] Total Time(s): %.2f\t' ..
        'average loss (per batch): %.5f \t mAP: %.5f',
        os.date('%X'), epoch, epoch_timer:time().real,
        epoch_loss / num_sequences, mean_average_precision))
end


return {
    Evaluator = Evaluator,
    SequentialEvaluator = SequentialEvaluator,
    compute_mean_average_precision = compute_mean_average_precision
}
