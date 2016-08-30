local classic = require 'classic'
local cudnn = require 'cudnn'
local cunn = require 'cunn'
local cutorch = require 'cutorch'
local image = require 'image'
local optim = require 'optim'
local paths = require 'paths'
local torch = require 'torch'

local evaluator = require 'evaluator'
local DataLoader = require('data_loader').DataLoader

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
    self.optimization_config = {
        learningRate = nil, -- set by update_learning_rate
        learningRateDecay = 0.0,
        momentum = args.momentum,
        dampening = 0.0,
        weightDecay = args.weight_decay
    }
    self.learning_rates = args.learning_rates

    -- Preallocate GPU inputs.
    self.gpu_inputs = torch.CudaTensor()
    self.gpu_labels = torch.CudaTensor()

    self.optimization_state = {}
    -- These variables view into the model's parameters, so that changes to the
    -- model's parameters are automatically reflected in them, and vice versa.
    self.model_parameters, self.model_grad_parameters =
        self.model:getParameters()

    -- Prefetch the next batch.
    self.data_loader:fetch_batch_async(self.batch_size)
end

function Trainer:update_learning_rate(epoch)
    local learning_rate, regime_was_updated = self:_epoch_learning_rate(epoch)
    if regime_was_updated then
        self.optimization_config.learningRate = learning_rate
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
        for batch_index, img in ipairs(step_images) do
            -- Process image after converting to the default Tensor type.
            -- (Originally, it is a ByteTensor).
            images[{step, batch_index}] = self:_process(img:typeAs(images))
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
    self:update_learning_rate(epoch)
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

        print(string.format(
              '%s: Epoch: [%d] [%d/%d] \t Time %.3f Loss %.4f LR %.0e',
              os.date('%X'), epoch, batch_index, num_batches,
              batch_timer:time().real, loss,
              self.optimization_config.learningRate))
    end

    local mean_average_precision = evaluator.compute_mean_average_precision(
        predictions, groundtruth)
    print(string.format(
        '%s: Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t' ..
        'average loss (per batch): %.5f \t mAP: %.5f',
        os.date('%X'), epoch, epoch_timer:time().real, loss_epoch / num_batches,
        mean_average_precision))
    collectgarbage()
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

function Trainer:_process(img)
    -- Avoid wrap around for ByteTensors, which are unsigned.
    assert(img:type() ~= torch.ByteTensor():type())

    -- Randomly crop.
    local width = img:size(3)
    local height = img:size(2)
    local x_origin = math.random(width - self.crop_size)
    local y_origin = math.random(height - self.crop_size)
    img = image.crop(img, x_origin, y_origin,
                     x_origin + self.crop_size, y_origin + self.crop_size)

    -- Mirror horizontally with probability 0.5.
    if torch.uniform() > 0.5 then
        img = image.hflip(img)
    end

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

function Trainer:_epoch_learning_rate(epoch)
    --[[
    Compute learning rate and weight decay regime for a given epoch.

    Args:
        epoch (num)
    Returns:
        params: Contains params.learning_rate and params.weight_decay
        is_new_regime: True if this marks the beginning of new parameters.
    --]]

    --[[
    Arbitrary learning rate policy modified from
    https://github.com/soumith/imagenet-multiGPU.torch
    except with learning rates divided by 10 because we're fine tuning.
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

return {Trainer = Trainer}
