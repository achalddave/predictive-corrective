local classic = require 'classic'
local cudnn = require 'cudnn'
local cunn = require 'cunn'
local cutorch = require 'cutorch'
local image = require 'image'
local optim = require 'optim'
local paths = require 'paths'

local evaluator = require 'evaluator'
local DataLoader = require('data_loader').DataLoader

local Trainer = classic.class('Trainer')

function Trainer:_init(args)
    --[[
    Args:
        model
        criterion
        data_loader
        pixel_mean
        batch_size
        crop_size
        num_labels
        momentum
    ]]--
    self.model = args.model
    self.criterion = args.criterion
    self.data_loader = args.data_loader
    self.pixel_mean = torch.Tensor(args.pixel_mean)
    self.batch_size = args.batch_size
    self.crop_size = args.crop_size
    self.num_labels = args.num_labels
    self.optimization_config = {
        learningRate = nil, -- set by _epoch_regime
        learningRateDecay = 0.0,
        momentum = args.momentum,
        dampening = 0.0,
        weightDecay = nil -- set by _epoch_regime
    }

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

function Trainer:update_regime(epoch)
    regime, regime_was_updated = self:_epoch_regime(epoch)
    if regime_was_updated then
        self.optimization_config.learningRate = regime.learning_rate
        self.optimization_config.weightDecay = regime.weight_decay
        self.optimization_state = {}
    end
    return regime_was_updated
end

function Trainer:train_batch()
    --[[
    Train on a batch of data

    Returns:
        loss: Output of criterion:forward on this batch.
        outputs: Output of model:forward on this batch.
        labels: True labels.
    ]]--
    local images_table, labels_table = self.data_loader:load_batch(
        self.batch_size)

    -- Fetch the next batch.
    self.data_loader:fetch_batch_async(self.batch_size)
    local images = torch.Tensor(#images_table, images_table[1]:size(1),
                                self.crop_size, self.crop_size)
    local labels = torch.ByteTensor(#labels_table, self.num_labels)
    for i, img in ipairs(images_table) do
        -- Process image after converting to the default Tensor type.
        -- (Originally, it is a ByteTensor).
        images[i] = self:_process(img:typeAs(images))
        labels[i] = DataLoader.labels_to_tensor(
            labels_table[i], self.num_labels)
    end

    self.gpu_inputs:resize(images:size()):copy(images)
    self.gpu_labels:resize(labels:size()):copy(labels)

    local loss, outputs
    local function model_forward_backward(_)
        self.model:zeroGradParameters()
        outputs = self.model:forward(self.gpu_inputs)
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
    self.model:training()
    self:update_regime(epoch)
    local epoch_timer = torch.Timer()
    local batch_timer = torch.Timer()

    local predictions = torch.Tensor(
        num_batches * self.batch_size, self.num_labels)
    local groundtruth = torch.ByteTensor(
        num_batches * self.batch_size, self.num_labels)

    loss_epoch = 0
    for batch_index = 1, num_batches do
        batch_timer:reset()
        collectgarbage()
        cutorch.synchronize()
        local loss, curr_predictions, curr_groundtruth = self:train_batch()
        cutorch.synchronize()
        loss_epoch = loss_epoch + loss

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

function Trainer:save(directory)
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

function Trainer:_epoch_regime(epoch)
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
    TODO(achald): Is this ideal?
    TODO(achald): Allow this to be configured from outside Trainer.
    --]]
    local regimes = {
        -- start,   LR,     WD
        {  1,     1e-2,   5e-4 },
        {  6,     1e-3,   5e-4 },
        { 12,     1e-4,   5e-4 },
        { 18,     1e-5,   5e-4 },
        { 24,     1e-6,   5e-4 },
    }

    local regime
    for i = 1, #regimes do
        if i == #regimes or
                (epoch >= regimes[i][1] and epoch < regimes[i+1][1]) then
            regime = regimes[i]
            break
        end
    end
    local is_new_regime = epoch == regime[1]
    return {
        learning_rate = regime[2],
        weight_decay = regime[3]
    }, is_new_regime
end

return {Trainer = Trainer}
