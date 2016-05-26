--[[
-- Fine-tunes an ImageNet-pretrained VGG-16 network on MultiTHUMOS data.
--]]

local argparse = require 'argparse'
local class = require 'class'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local nn = require 'nn'
local torch = require 'torch'
local yaml = require 'yaml'

local data_loader = require 'data_loader'
local evaluate = require 'evaluate'
local train = require 'train'

local parser = argparse() {
    description = 'Fine tune ImageNet-pretrained VGG-16 network on MultiTHUMOS.'
}
parser:argument('config', 'Config file')

local args = parser:parse()
local config = yaml.loadpath(args.config)
-- TODO(achald): Validate config.

cutorch.setDevice(config.gpu or 1)
torch.manualSeed(config.seed)
cutorch.manualSeed(config.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- Load model
local model
if config.model_init ~= nil then
    model = torch.load(config.model_init)
else
    model = require(config.model_layout)
end
-- https://groups.google.com/forum/#!topic/torch7/HiBymc9NfIY
model = model:cuda()
local criterion = nn.BCECriterion():cuda()
print 'Loaded model'

local optimization_config = {
    learningRate = config.learning_rate,
    learningRateDecay = 0.0,
    momentum = config.momentum,
    dampening = 0.0,
    weightDecay = config.weight_decay
}
local optimization_state

local train_loader = data_loader.DataLoader(
    config.train_lmdb, config.train_lmdb_without_images)
local val_loader = data_loader.DataLoader(
    config.val_lmdb, config.val_lmdb_without_images)

local trainer = train.Trainer {
    model = model,
    criterion = criterion,
    data_loader = train_loader,
    pixel_mean = config.pixel_mean,
    epoch_size = config.epoch_size,
    batch_size = config.batch_size,
    crop_size = config.crop_size,
    num_labels = config.num_labels,
    momentum = config.momentum
}
local evaluator = evaluate.Evaluator {
    model = model,
    criterion = criterion,
    data_loader = val_loader,
    pixel_mean = config.pixel_mean,
    batch_size = config.batch_size,
    crop_size = config.crop_size,
    num_labels = config.num_labels
}

print('Initialized trainer and evaluator.')
for i = 1, config.num_epochs do
    print(('Training epoch %d'):format(i))
    epoch = config.init_epoch + i - 1
    trainer:train_epoch(epoch, config.epoch_size)
    trainer:save(config.cache_dir)
    evaluator:evaluate_epoch(epoch, config.epoch_size)
end
