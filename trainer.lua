local classic = require 'classic'
local cudnn = require 'cudnn'
local cunn = require 'cunn'
local cutorch = require 'cutorch'
local optim = require 'optim'
local paths = require 'paths'
local torch = require 'torch'
require 'nnlr'

local evaluator = require 'evaluator'
local image_util = require 'util/image_util'
local log = require 'util/log'
local END_OF_SEQUENCE = require('data_loader').END_OF_SEQUENCE

local Trainer = classic.class('Trainer')

function Trainer:_init(args)
    --[[
    Trains a model on images.

    This is by no means a general purpose trainer class. It assumes a number of
    things about the model and inputs, as described below.

    By default, the model is assumed to take in inputs of size
        (sequence_length, batch_size, num_channels, crop_size, crop_size)
    The sequence_length and num_channels can be arbitrary (they depend on the
    data loaders), but batch_size must be specified. The exact order of the
    dimensions can be changed by specifying input_dimension_permutation (see
    below).

    By default, the model will be passed the entire input tensor above, but this
    can be changed with computational_batch_size and backprop_rho (which can
    be used for truncated backprop). See doc for these parameters below.

    Args:
        model
        criterion
        train_data_loader
        val_data_loader
        input_dimension_permutation: Array, default nil.
            Specifies what each dimension in the input tensor corresponds to.
            By default, the input dimension order is
              (sequence_length, batch_size, num_channels, width, height)
            A permutation of [2, 3, 1, 4, 5], for example, results in
              (batch_size, num_channels, seuquence_length, width, height)
        pixel_mean
        batch_size
        computational_batch_size
        backprop_rho (int): Optional. If specified, sequences will be fed to the
            model in chunks of backprop_rho steps, and after all the chunks
            have been processed, `model:forget()` will be called. This can be
            used for truncated back-propagation with a model that maintains
            state across forward/backward calls.
        crop_size
        learning_rates: Array of tables containing keys 'start_epoch',
            'learning_rate'. E.g.
                [{start_epoch: 1, learning_rate: 1e-2},
                 {start_epoch: 6, learning_rate: 1e-3}]
            will use a learning rate of 1e-2 for the first 5 epochs, then switch
            to a learning rate of 1e-3.
        gradient_clip (float)
        momentum (float)
        weight_decay (float)
        use_nnlr (bool): If true, use nnlr to train with layer
            wise learning rates. Otherwise, use the same learning rate for all
            layers. (Default: False)
        optim_config: Optional
        optim_state: Optional
    ]]--
    self.model = args.model

    self.gradient_clip = args.gradient_clip
    self.criterion = args.criterion
    self.train_data_loader = args.train_data_loader
    self.val_data_loader = args.val_data_loader
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
    self.computational_batch_size = args.computational_batch_size or
                                    args.batch_size
    self.backprop_rho = args.backprop_rho
    self.crop_size = args.crop_size
    self.weight_decay = args.weight_decay
    self.learning_rates = args.learning_rates
    self.use_nnlr = args.use_nnlr == nil and false or args.use_nnlr

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
    log.info('Getting model parameters')
    -- These variables view into the model's parameters, so that changes to the
    -- model's parameters are automatically reflected in them, and vice versa.
    self.model_parameters, self.model_grad_parameters =
        self.model:getParameters()
    log.info('Got model parameters')

    -- Prefetch the next batch.
    self.train_data_loader:fetch_batch_async(self.batch_size)
    self.val_data_loader:fetch_batch_async(self.batch_size)
    self.num_labels = self.train_data_loader:num_labels()

    self.cleared_model = self.model
    if torch.isTypeOf(self.cleared_model, 'nn.DataParallelTable') then
        self.cleared_model = self.cleared_model:get(1)
    end
    self.cleared_model = self.cleared_model:sharedClone()
    self.cleared_model:clearState()
end

function Trainer:update_optim_config(epoch)
    local learning_rate, regime_was_updated = self:_epoch_learning_rate(epoch)
    self.epoch_base_learning_rate = learning_rate

    if self.use_nnlr and self.optimization_config.learningRates == nil then
        -- For whatever reason, optim.sgd will use (decay_fn(learningRate) *
        -- learningRates) as the vector of learning rates, but will use
        -- weightDecays (ignoring weightDecay the scalar) as the vector of
        -- weight decays. So we need to supply a vector of learning rate
        -- multipliers and a vector of weight decays here, and supply a
        -- base `learningRate` below when the regime is updated.
        log.info('Using layerwise learning rates')
        local layer_learning_rate_multipliers, layer_weight_decays =
            self.model:getOptimConfig(
                1 --[[base lr multiplier]],
                self.weight_decay --[[base weight decay]])
        self.optimization_config.learningRates =
            layer_learning_rate_multipliers
        self.optimization_config.weightDecays =
            layer_weight_decays
    end

    if regime_was_updated then
        if self.use_nnlr then
            self.optimization_config.learningRate = learning_rate
        else
            self.optimization_config.learningRate = learning_rate
            self.optimization_config.weightDecay = self.weight_decay
        end
        self.optimization_state = nil
        collectgarbage()
        collectgarbage()
        self.optimization_state = {}
    end
    return regime_was_updated
end

function Trainer:train_epoch(epoch, num_batches)
    self:_train_or_evaluate_epoch(epoch, num_batches, true --[[train_mode]])
end

function Trainer:evaluate_epoch(epoch, num_batches)
    self:_train_or_evaluate_epoch(epoch, num_batches, false --[[train_mode]])
end

function Trainer:save(directory, epoch)
    --[[
    Save model, optimization config, and optimization config to a directory.
    ]]--
    -- Clear intermediate states in the model before saving to disk to minimize
    -- disk space usage.
    torch.save(
        paths.concat(directory, 'model_' .. epoch .. '.t7'), self.cleared_model)
    torch.save(paths.concat(directory, 'optim_config_' .. epoch .. '.t7'),
               self.optimization_config)
    torch.save(paths.concat(directory, 'optim_state_' .. epoch .. '.t7'),
               self.optimization_state)
    collectgarbage()
    collectgarbage()
end

function Trainer:_train_or_evaluate_batch(train_mode)
    local data = train_mode and self.train_data_loader or self.val_data_loader
    local images, labels = self:_load_batch(data, train_mode)
    local loss = 0
    local outputs
    local function forward_backward()
        if train_mode then
            self.model:zeroGradParameters()
        end
        for i = 1, math.ceil(self.batch_size / self.computational_batch_size) do
            local start_index = (i - 1) * self.computational_batch_size + 1
            local end_index = math.min(
                i * self.computational_batch_size, self.batch_size)
            local chunk_loss, chunk_outputs =
                self:_forward_backward(
                    images[{{}, {start_index, end_index}}],
                    labels[{{}, {start_index, end_index}}],
                    train_mode)
            loss = loss + chunk_loss
            if outputs == nil then
                outputs = chunk_outputs:clone()
            else
                assert(chunk_outputs:dim() == 3,
                       'Unknown output size:\n' ..
                       tostring(chunk_outputs:size()))
                -- Outputs should be of size
                -- (sequence_length, batch_size, num_labels).
                -- Concatenate across the second dimension.
                outputs = torch.cat(outputs, chunk_outputs, 2 --[[batch dim]])
            end
        end
        if self.gradient_clip ~= nil then
            self.model_grad_parameters:clamp(
                -self.gradient_clip, self.gradient_clip)
        end
        return loss, self.model_grad_parameters
    end

    if train_mode then
        -- Updates self.model_parameters (and, in turn, the parameters of
        -- self.model) in place.
        optim.sgd(forward_backward, self.model_parameters,
                  self.optimization_config, self.optimization_state)
    else
        forward_backward()
    end
    return loss, outputs, labels
end

function Trainer:_train_or_evaluate_epoch(epoch, num_batches, train_mode)
    if train_mode then
        self.model:training()
        self:update_optim_config(epoch)
    else
        self.model:evaluate()
    end

    local epoch_timer = torch.Timer()
    local batch_timer = torch.Timer()

    local predictions = torch.CudaTensor(
        num_batches * self.batch_size, self.num_labels)
    local groundtruth = torch.ByteTensor(
        num_batches * self.batch_size, self.num_labels)

    local loss_epoch = 0
    for batch_index = 1, num_batches do
        batch_timer:reset()
        collectgarbage()
        collectgarbage()

        local loss, curr_predictions, curr_groundtruth =
            self:_train_or_evaluate_batch(train_mode)
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
                      {}}] = curr_predictions
        groundtruth[{{epoch_index_start,
                      epoch_index_start + self.batch_size - 1},
                      {}}] = curr_groundtruth

        if train_mode then
            local log_string = string.format(
                'Epoch: [%d] [%d/%d] \t Time %.3f Loss %.4f',
                epoch, batch_index, num_batches,
                batch_timer:time().real, loss)
            if batch_index % 10 == 0 then
                local current_mean_average_precision =
                    evaluator.compute_mean_average_precision(
                        predictions[{{1, epoch_index_start + self.batch_size - 1}}],
                        groundtruth[{{1, epoch_index_start + self.batch_size - 1}}])
                log_string = log_string .. string.format(
                    ' epoch mAP %.4f', current_mean_average_precision)
            end
            log_string = log_string .. string.format(
                ' LR %.0e', self.epoch_base_learning_rate)
            log.info(log_string)
        end
    end

    local mean_average_precision = evaluator.compute_mean_average_precision(
        predictions, groundtruth)
    predictions = nil
    groundtruth = nil
    collectgarbage()
    collectgarbage()

    local mode_str = train_mode and 'TRAINING' or 'EVALUATION'

    log.info(string.format(
        'Epoch: [%d][%s SUMMARY] Total Time(s): %.2f\t' ..
        'average loss (per batch): %.5f \t mAP: %.5f',
        epoch, mode_str, epoch_timer:time().real, loss_epoch /
        num_batches, mean_average_precision))
end

function Trainer:_load_batch(data_loader, train_mode)
    local images_table, labels = data_loader:load_batch(self.batch_size)
    -- Prefetch the next batch.
    data_loader:fetch_batch_async(self.batch_size)

    local num_steps = #images_table
    local num_channels = images_table[1][1]:size(1)
    local images = torch.Tensor(num_steps, self.batch_size, num_channels,
                                self.crop_size, self.crop_size)
    local augment = train_mode and image_util.augment_image_train
                               or image_util.augment_image_eval
    local sequence_states = {}
    for step, step_images in ipairs(images_table) do
        for sequence, img in ipairs(step_images) do
            -- Process image after converting to the default Tensor type.
            -- (Originally, it is a ByteTensor).
            images[{step, sequence}], sequence_states[sequence] = augment(
                img:typeAs(images),
                self.crop_size,
                self.crop_size,
                self.pixel_mean,
                sequence_states[sequence])
        end
    end
    return images, labels
end

function Trainer:_forward_backward(images, labels, train_mode)
    --[[
    Run forward (and optionally backward) pass on images.

    Args:
        images ((sequence_length, batch_size, num_channels, width, height))
        labels: Subset of output of data_loader:load_batch()
        train_mode (bool): If true, perform backward pass as well.
    ]]--
    local num_images = images:size(2)
    local sequence_length = images:size(1)
    local sequence_chunk = self.backprop_rho or sequence_length

    -- This sequence chunking code is similar to what we do for
    -- computational_batch_size in _train_or_evaluate_batch, but there isn't an
    -- easy way to share the code.
    local loss = 0
    local outputs
    for i = 1, math.ceil(sequence_length / sequence_chunk) do
        local start_index = (i - 1) * sequence_chunk + 1
        local end_index = math.min(i * sequence_chunk, sequence_length)
        local chunk_images = images[{{start_index, end_index}, {}}]
        local chunk_labels = labels[{{start_index, end_index}, {}}]
        if self.input_dimension_permutation then
            chunk_images = chunk_images:permute(
                unpack(self.input_dimension_permutation))
        end
        self.gpu_inputs:resize(chunk_images:size()):copy(chunk_images)
        self.gpu_labels:resize(chunk_labels:size()):copy(chunk_labels)

        local current_outputs = self.model:forward(self.gpu_inputs)

        if current_outputs:dim() == 2 then
            current_outputs = nn.utils.addSingletonDimension(current_outputs, 1)
        end
        if outputs == nil then
            outputs = current_outputs:clone()
        else
            outputs = torch.cat(outputs, current_outputs, 1 --[[sequence]])
        end

        -- If the output of the network is a single prediction for the sequence,
        -- compare it to the label of the last frame.
        if current_outputs:size(1) == 1 and self.gpu_labels:size(1) ~= 1 then
            log.info('Only one output from network, but multiple GT labels.')
            self.gpu_labels = self.gpu_labels[self.gpu_labels:size(1)]
        end
        loss = loss + self.criterion:forward(current_outputs, self.gpu_labels)

        if train_mode then
            local criterion_gradients = self.criterion:backward(
                current_outputs, self.gpu_labels)
            if criterion_gradients:norm() <= 1e-10 and loss >= 1e-10 then
                log.info(string.format(
                    'Criterion gradients small: %.2f; Loss: %.2f',
                    criterion_gradients:norm(), loss))
            end
            self.model:backward(self.gpu_inputs,
                                criterion_gradients,
                                num_images / self.batch_size)
        end
    end
    -- Forget state for next set of sequences. This is necessary since we use
    -- truncated backpropagation above to compute gradients over long sequences
    -- from small chunks (of length self.backprop_rho).
    if torch.isTypeOf(self.model, 'nn.DataParallelTable') then
        -- https://github.com/Element-Research/rnn/issues/404
        self.model.impl:exec(function(m) m:forget() end)
    else
        self.model:forget()
    end

    -- The loss is averaged by the computational batch size; we want to
    -- average by the actual batch size.
    loss = loss * (num_images / self.batch_size)

    return loss, outputs
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

local SequentialTrainer, parent = classic.class('SequentialTrainer', Trainer)
function SequentialTrainer:_init(args)
    if args.input_dimension_permutation ~= nil then
        for i = 1, #args do
            if args.input_dimension_permutation[i] ~= i then
                error('SequentialTrainer does not support ' ..
                      'input_dimension_permutation')
            end
        end
    end
    parent._init(self, args)
    assert(self.batch_size == 1,
          'Currently, SequentialTrainer only supports batch size = 1. ' ..
          'See the "recurrent_batched_training" branch for some WIP on ' ..
          'allowing the batch size to be greater than 1.')
    assert(self.model:findModules('nn.Sequencer') ~= nil,
           'SequentialTrainer requires that the input model be decorated ' ..
           'with nn.Sequencer.')
    assert(torch.isTypeOf(self.criterion, 'nn.SequencerCriterion'),
           'SequentialTrainer expects SequencerCriterion.')
    self.model:remember('both')
end

function SequentialTrainer:_train_or_evaluate_batch(train_mode)
    --[[
    Train or evaluate on a batch of data.

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
    local data_loader
    if train_mode then
        self.model:zeroGradParameters()
        data_loader = self.train_data_loader
    else
        data_loader = self.val_data_loader
    end

    local images_table, labels = data_loader:load_batch(1 --[[batch size]])
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
    data_loader:fetch_batch_async(1 --[[batch size]])

    local num_steps = #images_table
    local num_channels = images_table[1][1]:size(1)
    local images = torch.Tensor(num_steps, 1 --[[batch size]], num_channels,
                                self.crop_size, self.crop_size)
    local num_valid_steps = num_steps
    local augment = train_mode and image_util.augment_image_train
                               or image_util.augment_image_eval
    local augment_state = nil
    for step, step_images in ipairs(images_table) do
        local img = step_images[1]
        if img == END_OF_SEQUENCE then
            -- We're out of frames for this sequence.
            num_valid_steps = step - 1
            break
        else
            -- Process image after converting to the default Tensor type.
            -- (Originally, it is a ByteTensor).
            images[step], augment_state = augment(
                img:typeAs(images), self.crop_size, self.crop_size,
                self.pixel_mean, augment_state)
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

    local loss, outputs
    if train_mode then
        local function model_forward_backward(_)
            -- Should be of shape (sequence_length, batch_size, num_classes)
            outputs = self.model:forward(self.gpu_inputs)
            loss = self.criterion:forward(outputs, self.gpu_labels)
            local criterion_gradients = self.criterion:backward(
                outputs, self.gpu_labels)
            if criterion_gradients:norm() <= 1e-5 and loss >= 1e-10 then
                log.info(string.format(
                    'Criterion gradients small: %.2f; Loss: %.2f',
                    criterion_gradients:norm(), loss))
            end
            self.model:backward(self.gpu_inputs, criterion_gradients)
            if self.gradient_clip ~= nil then
                self.model_grad_parameters:clamp(
                    -self.gradient_clip, self.gradient_clip)
            end
            return loss, self.model_grad_parameters
        end

        -- Updates self.model_parameters (and, in turn, the parameters of
        -- self.model) in place.
        optim.sgd(model_forward_backward, self.model_parameters,
                self.optimization_config, self.optimization_state)
    else
        -- Should be of shape (sequence_length, batch_size, num_classes)
        outputs = self.model:forward(self.gpu_inputs)
        loss = self.criterion:forward(outputs, self.gpu_labels)
    end
    if sequence_ended then
        self.model:forget()
    end
    return loss, outputs, labels, sequence_ended
end

function SequentialTrainer:_train_or_evaluate_epoch(
    epoch, num_sequences, train_mode)
    if train_mode then
        self.model:training()
        self:update_optim_config(epoch)
    else
        self.model:evaluate()
    end
    local epoch_timer = torch.Timer()
    local batch_timer = torch.Timer()

    local predictions, groundtruth

    local epoch_loss = 0
    for sequence = 1, num_sequences do
        batch_timer:reset()
        collectgarbage()
        local sequence_ended = false
        local sequence_predictions, sequence_groundtruth
        local sequence_loss = 0
        local num_steps_in_sequence = 0
        io.write(sequence)
        while not sequence_ended do
            local loss, batch_predictions, batch_groundtruth, sequence_ended_ =
                self:_train_or_evaluate_batch(train_mode)
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
            io.write('>')
        end
        io.write('x\n')
        epoch_loss = epoch_loss + sequence_loss
        if train_mode then
            local sequence_mean_average_precision =
                evaluator.compute_mean_average_precision(
                    sequence_predictions, sequence_groundtruth)
            log.info(string.format(
                'Epoch: [%d] [%d/%d] \t Time %.3f Loss %.4f ' ..
                'seq mAP %.4f LR %.0e',
                epoch, sequence, num_sequences,
                batch_timer:time().real, sequence_loss,
                sequence_mean_average_precision, self.epoch_base_learning_rate))
        end
        if predictions == nil then
            predictions = sequence_predictions
            groundtruth = sequence_groundtruth
        else
            predictions = torch.cat(predictions, sequence_predictions, 1)
            groundtruth = torch.cat(groundtruth, sequence_groundtruth, 1)
        end
        collectgarbage()
        collectgarbage()
    end

    local mean_average_precision = evaluator.compute_mean_average_precision(
        predictions, groundtruth)

    local mode_str = train_mode and 'TRAINING' or 'EVALUATION'
    log.info(string.format(
        'Epoch: [%d][%s SUMMARY] Total Time(s): %.2f\t' ..
        'average loss (per batch): %.5f \t mAP: %.5f',
        epoch, mode_str, epoch_timer:time().real,
        epoch_loss / num_sequences, mean_average_precision))
    collectgarbage()
    collectgarbage()
end

return {Trainer = Trainer, SequentialTrainer = SequentialTrainer}
