--[[
-- Fine-tunes an ImageNet-pretrained VGG-16 network on MultiTHUMOS data.
--]]

local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local paths = require 'paths'
local nn = require 'nn'
local torch = require 'torch'
local yaml = require 'yaml'

local data_loader = require 'data_loader'
local evaluator = require 'evaluator'
local trainer = require 'trainer'

local parser = argparse() {
    description = 'Fine tune ImageNet-pretrained VGG-16 network on MultiTHUMOS.'
}
parser:argument('config', 'Config file')
parser:argument('cache_base',
                'Directory to save model snapshots, logging, etc. to.')

local args = parser:parse()
local config = yaml.loadpath(args.config)
-- TODO(achald): Validate config.

-- Create cache_base
if not paths.dirp(args.cache_base) and not paths.mkdir(args.cache_base) then
    print('Error creating cache base dir:', args.cache_base)
    os.exit()
end
local cache_dir = paths.concat(args.cache_base, os.date('%X'))
if not paths.mkdir(cache_dir) then
    print('Error making cache dir:', cache_dir)
    os.exit()
end
print('Saving run information to', cache_dir)

-- Save config to cache_dir
function copy_file_naive(in_path, out_path)
    -- TODO(achald): Use a library function, if one exists.
    local in_file = io.open(in_path, 'r')
    local in_contents = in_file:read('*all')
    in_file:close()
    local out_file = io.open(out_path, 'w')
    out_file:write(in_contents)
    out_file:close()
end
copy_file_naive(args.config, paths.concat(cache_dir, 'config.yaml'))

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
local criterion = nn.MultiLabelSoftMarginCriterion():cuda()
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
    config.train_lmdb, config.train_lmdb_without_images, config.num_labels)
local val_loader = data_loader.DataLoader(
    config.val_lmdb, config.val_lmdb_without_images, config.num_labels)

local trainer = trainer.Trainer {
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
local evaluator = evaluator.Evaluator {
    model = model,
    criterion = criterion,
    data_loader = val_loader,
    pixel_mean = config.pixel_mean,
    -- During evaluation, we can handle a slightly larger batch size since we
    -- don't use as much memory.
    batch_size = torch.round(1.5 * config.batch_size),
    crop_size = config.crop_size,
    num_labels = config.num_labels
}

print('Initialized trainer and evaluator.')
for i = 1, config.num_epochs do
    print(('Training epoch %d'):format(i))
    epoch = config.init_epoch + i - 1
    trainer:train_epoch(epoch, config.epoch_size)
    trainer:save(cache_dir)
    evaluator:evaluate_epoch(epoch, config.epoch_size)
end
