local classic = require 'classic'
local cudnn = require 'cudnn'
local cunn = require 'cunn'
local cutorch = require 'cutorch'
local optim = require 'optim'
local paths = require 'paths'
local torch = require 'torch'

local evaluator = require 'evaluator'
local image_util = require 'util/image_util'

local Trainer = classic.class('Trainer')

function Trainer:_init(args)
    --[[
    Args:
        model
        criterion
        data_loader
        input_dimension_permutation: Array, default nil.
            Specifies what each dimension in the input tensor corresponds to.
            By default, the input dimension order is
              (sequence_length, batch_size, num_channels, width, height)
            A permutation of [2, 3, 1, 4, 5], for example, results in
              (batch_size, num_channels, seuquence_length, width, height)
        pixel_mean
        batch_size
        crop_size
        learning_rates: Array of tables containing keys 'start_epoch',
            'learning_rate'. E.g.
                [{start_epoch: 1, learning_rate: 1e-2},
                 {start_epoch: 6, learning_rate: 1e-3}]
            will use a learning rate of 1e-2 for the first 5 epochs, then switch
            to a learning rate of 1e-3.
        num_labels
        momentum
        weight_decay
        optim_config: Optional
        optim_state: Optional
    ]]--
    self.model = args.model
    self.criterion = args.criterion
    self.data_loader = args.data_loader
    -- Only use input permutation if it is not the identity.
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
    self.weight_decay = args.weight_decay
    self.learning_rates = args.learning_rates

    -- Preallocate GPU inputs.
    self.gpu_inputs = torch.CudaTensor()
    self.gpu_labels = torch.CudaTensor()

    if args.optim_config then
        self.optimization_config = args.optim_config
    else
        self.optimization_config = {
            learningRateDecay = 0.0,
            momentum = args.momentum,
            dampening = 0.0,
            learningRate = nil, -- set by update_optim_config
            weightDecay = nil -- set by update_optim_config
        }
    end
    if args.optim_state then
        self.optimization_state = args.optim_state
    else
        self.optimization_state = {}
    end
    -- These variables view into the model's parameters, so that changes to the
    -- model's parameters are automatically reflected in them, and vice versa.
    self.model_parameters, self.model_grad_parameters =
        self.model:getParameters()

    -- Prefetch the next batch.
    self.data_loader:fetch_batch_async(self.batch_size)
end

function Trainer:update_optim_config(epoch)
    local learning_rate, regime_was_updated = self:_epoch_learning_rate(epoch)
    self.epoch_base_learning_rate = learning_rate
    if regime_was_updated then
        self.optimization_config.learningRate = learning_rate
        self.optimization_config.weightDecay = self.weightDecay
        self.optimization_state = {}
    end
    return regime_was_updated
end

function Trainer:train_batch()
    --[[
    Train on a batch of data

    Returns:
        loss: Output of criterion:forward on this batch.
        outputs (Tensor): Output of model:forward on this batch. The tensor
            size should be either (sequence_length, batch_size, num_labels) or
            (batch_size, num_labels), depending on the model.
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
            images[{step, sequence}] = image_util.augment_image_train(
                img:typeAs(images), self.crop_size, self.crop_size,
                self.pixel_mean)
        end
    end

    if self.input_dimension_permutation then
        images = images:permute(unpack(self.input_dimension_permutation))
    end

    self.gpu_inputs:resize(images:size()):copy(images)
    self.gpu_labels:resize(labels:size()):copy(labels)

    -- TODO(achald): Allow smaller computational batch sizes while maintaining
    -- optimization batch size (i.e. accumulate gradients across computational
    -- batch sizes).
    local loss, outputs
    local function model_forward_backward(_)
        self.model:zeroGradParameters()
        outputs = self.model:forward(self.gpu_inputs)
        -- If the output of the network is a single prediction for the sequence,
        -- compare it to the label of the last frame.
        if (outputs:size(1) == 1 or outputs:dim() == 2) and
                self.gpu_labels:size(1) ~= 1 then
            self.gpu_labels = self.gpu_labels[self.gpu_labels:size(1)]
        end
        loss = self.criterion:forward(outputs, self.gpu_labels)
        local criterion_gradients = self.criterion:backward(
            outputs, self.gpu_labels)
        self.model:backward(self.gpu_inputs, criterion_gradients)
        return loss, self.model_grad_parameters
    end

    -- Updates self.model_parameters (and, in turn, the parameters of
    -- self.model) in place.
    optim.sgd(model_forward_backward, self.model_parameters,
              self.optimization_config, self.optimization_state)
    return loss, outputs, labels
end

function Trainer:train_epoch(epoch, num_batches)
    self.model:clearState()
    self.model:training()
    self:update_optim_config(epoch)
    local epoch_timer = torch.Timer()
    local batch_timer = torch.Timer()

    local predictions = torch.Tensor(
        num_batches * self.batch_size, self.num_labels)
    local groundtruth = torch.ByteTensor(
        num_batches * self.batch_size, self.num_labels)

    local loss_epoch = 0
    for batch_index = 1, num_batches do
        batch_timer:reset()
        collectgarbage()
        local loss, curr_predictions, curr_groundtruth = self:train_batch()
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

        local current_mean_average_precision =
            evaluator.compute_mean_average_precision(
                predictions[{{1, epoch_index_start + self.batch_size - 1}}],
                groundtruth[{{1, epoch_index_start + self.batch_size - 1}}])

        print(string.format(
              '%s: Epoch: [%d] [%d/%d] \t Time %.3f Loss %.4f ' ..
              'epoch mAP %.4f LR %.0e',
              os.date('%X'), epoch, batch_index, num_batches,
              batch_timer:time().real, loss,
              current_mean_average_precision,
              self.epoch_base_learning_rate))
    end

    local mean_average_precision = evaluator.compute_mean_average_precision(
        predictions, groundtruth)

    print(string.format(
        '%s: Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t' ..
        'average loss (per batch): %.5f \t mAP: %.5f',
        os.date('%X'), epoch, epoch_timer:time().real, loss_epoch / num_batches,
        mean_average_precision))
end

function Trainer:save(directory, epoch)
    --[[
    Save model, optimization config, and optimization config to a directory.
    ]]--
    -- Clear intermediate states in the model before saving to disk to minimize
    -- disk space usage.
    self.model:clearState()
    torch.save(paths.concat(directory, 'model_' .. epoch .. '.t7'), self.model)
    torch.save(paths.concat(directory, 'optim_config_' .. epoch .. '.t7'),
               self.optimization_config)
    torch.save(paths.concat(directory, 'optim_state_' .. epoch .. '.t7'),
               self.optimization_state)
end

function Trainer:_epoch_learning_rate(epoch)
    --[[
    Compute learning rate and weight decay regime for a given epoch.

    Args:
        epoch (num)
    Returns:
        params: Contains params.learning_rate and params.weight_decay
        is_new_regime: True if this marks the beginning of new parameters.
    --]]

    local regime
    for i = 1, #self.learning_rates - 1 do
        local start_epoch = self.learning_rates[i].start_epoch
        local end_epoch = self.learning_rates[i+1].start_epoch
        if epoch >= start_epoch and epoch < end_epoch then
            regime = self.learning_rates[i]
            break
        end
    end
    if regime == nil then
        regime = self.learning_rates[#self.learning_rates]
    end
    local is_new_regime = epoch == regime.start_epoch
    return regime.learning_rate, is_new_regime
end

local SequentialTrainer, SequentialTrainerSuper = classic.class(
    'SequentialTrainer', Trainer)
function SequentialTrainer:_init(args)
    if args.input_dimension_permutation ~= nil then
        for i = 1, #args do
            if args.input_dimension_permutation[i] ~= i then
                error('SequentialTrainer does not support ' ..
                      'input_dimension_permutation')
            end
        end
    end
    SequentialTrainerSuper._init(self, args)
    assert(self.batch_size == 1,
          'Currently, SequentialTrainer only supports batch size = 1. ' ..
          'See the "recurrent_batched_training" branch for some WIP on ' ..
          'allowing the batch size to be greater than 1.')
    assert(torch.isTypeOf(self.model, 'nn.Sequencer'),
           'SequentialTrainer requires that the input model be decorated ' ..
           'with nn.Sequencer.')
    assert(torch.isTypeOf(self.criterion, 'nn.SequencerCriterion'),
           'SequentialTrainer expects SequencerCriterion.')
    self.model:remember('both')
    self.max_backprop_steps = args.max_backprop_steps
end

function SequentialTrainer:train_batch()
    --[[
    Train on a batch of data

    Returns:
        loss: Output of criterion:forward on this batch.
        outputs (Tensor): Output of model:forward on this batch. The tensor
            size should be either (sequence_length, 1, num_labels). The
            sequence_length may be shorter at the end of the sequence (if the
            sequence ends before we get enough frames).
        labels (Tensor): True labels. Same size as the outputs.
    ]]--
    local images_table, labels = self.data_loader:load_batch(1 --[[batch size]])
    if images_table[1][1] == nil then
        -- The sequence ended at the end of the last batch; reset the model and
        -- start loading the next sequence in the next batch.
        self.model:forget()
        local images_table, labels = self.data_loader:load_batch(1 --[[batch size]])
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
        if img == nil then
            -- We're out of frames for this sequence.
            num_valid_steps = step - 1
            break
        else
            -- Process image after converting to the default Tensor type.
            -- (Originally, it is a ByteTensor).
            images[step] = image_util.augment_image_train(
                img:typeAs(images), self.crop_size, self.crop_size,
                self.pixel_mean)
        end
    end
    if num_valid_steps ~= num_steps then
        labels = labels[{1, num_valid_steps}]
        images = images[{1, num_valid_steps}]
    end

    self.gpu_inputs:resize(images:size()):copy(images)
    self.gpu_labels:resize(labels:size()):copy(labels)

    local loss, outputs
    local function model_forward_backward(_)
        self.model:zeroGradParameters()
        -- Should be of shape (sequence_length, batch_size, num_classes)
        outputs = self.model:forward(self.gpu_inputs)
        loss = self.criterion:forward(outputs, labels)
        local criterion_gradients = self.criterion:backward(
            outputs, loss)
        self.model:backward(self.gpu_inputs, criterion_gradients)
        return loss, self.model_grad_parameters
    end

    -- Updates self.model_parameters (and, in turn, the parameters of
    -- self.model) in place.
    optim.sgd(model_forward_backward, self.model_parameters,
              self.optimization_config, self.optimization_state)
    if num_valid_steps ~= num_steps then
        self.model:forget()
    end
    return loss, outputs, labels
end

function SequentialTrainer:train_epoch(epoch, num_batches)
    self.model:clearState()
    self.model:training()
    self:update_optim_config(epoch)
    local epoch_timer = torch.Timer()
    local batch_timer = torch.Timer()

    local predictions = {}
    local groundtruth = {}

    local loss_epoch = 0
    for batch_index = 1, num_batches do
        batch_timer:reset()
        collectgarbage()
        local loss, batch_predictions, batch_groundtruth = self:train_batch()
        loss_epoch = loss_epoch + loss

        local num_steps = torch.isTensor(batch_predictions) and
            batch_predictions:size(1) or #batch_predictions
        for step = 1, num_steps do
            table.insert(predictions,
                         batch_predictions[step][1 --[[sequence index]]])
            table.insert(groundtruth,
                         batch_groundtruth[step][1 --[[sequence index]]])
        end

        local current_mean_average_precision =
            evaluator.compute_mean_average_precision(
                torch.cat(predictions, 2):t(),
                torch.cat(groundtruth, 2):t())

        print(string.format(
              '%s: Epoch: [%d] [%d/%d] \t Time %.3f Loss %.4f ' ..
              'epoch mAP %.4f LR %.0e',
              os.date('%X'), epoch, batch_index, num_batches,
              batch_timer:time().real, loss,
              current_mean_average_precision,
              self.epoch_base_learning_rate))
    end

    local mean_average_precision = evaluator.compute_mean_average_precision(
        predictions, groundtruth)

    print(string.format(
        '%s: Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t' ..
        'average loss (per batch): %.5f \t mAP: %.5f',
        os.date('%X'), epoch, epoch_timer:time().real, loss_epoch / num_batches,
        mean_average_precision))
end

return {Trainer = Trainer, SequentialTrainer = SequentialTrainer}
