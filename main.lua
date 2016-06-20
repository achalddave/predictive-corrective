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
local save_run_info = require 'save_run_info'

local parser = argparse() {
    description = 'Fine tune ImageNet-pretrained VGG-16 network on MultiTHUMOS.'
}
parser:argument('config', 'Config file')
parser:argument('cache_base',
                'Directory to save model snapshots, logging, etc. to.')

local args = parser:parse()
local config = yaml.loadpath(args.config)

if config.data_paths_config ~= nil then
    local data_paths = yaml.loadpath(config.data_paths_config)
    local train_path = data_paths[config.train_split]
    local val_path = data_paths[config.val_split]
    config.train_lmdb = train_path.with_images
    config.train_lmdb_without_images = train_path.without_images
    config.val_lmdb = val_path.with_images
    config.val_lmdb_without_images = val_path.without_images
end
-- TODO(achald): Validate config.

-- Create cache_base
if not paths.dirp(args.cache_base) and not paths.mkdir(args.cache_base) then
    print('Error creating cache base dir:', args.cache_base)
    os.exit()
end
local cache_dir = paths.concat(args.cache_base, os.date('%m-%d-%y-%H-%m-%S'))
if not paths.mkdir(cache_dir) then
    print('Error making cache dir:', cache_dir)
    os.exit()
end
save_run_info.save_git_info(cache_dir)
print('Saving run information to', cache_dir)

-- Save config to cache_dir
save_run_info.copy_file_naive(args.config,
                              paths.concat(cache_dir, 'config.yaml'))

cutorch.setDevice(config.gpus[1])
torch.manualSeed(config.seed)
cutorch.manualSeed(config.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- Load model
local single_model
if config.model_init ~= nil then
    single_model = torch.load(config.model_init)
else
    single_model = require(config.model_layout)
end
local model = nn.DataParallelTable(1)
for _, gpu in ipairs(config.gpus) do
    cutorch.setDevice(gpu)
    model:add(single_model:clone():cuda(), gpu)
end
cutorch.setDevice(config.gpus[1])
-- https://groups.google.com/forum/#!topic/torch7/HiBymc9NfIY
model = model:cuda()
local criterion = nn.MultiLabelSoftMarginCriterion():cuda()
print 'Loaded model'

local sampling_strategies = {
    permuted = data_loader.PermutedSampler,
    balanced = data_loader.BalancedSampler
}
local train_sampler = sampling_strategies[config.sampling_strategy:lower()](
    config.train_lmdb_without_images,
    config.num_labels,
    config.sampling_strategy_options)
local val_sampler = data_loader.PermutedSampler(
    config.val_lmdb_without_images, config.num_labels)

local train_loader = data_loader.DataLoader(
    config.train_lmdb, train_sampler, config.num_labels)
local val_loader = data_loader.DataLoader(
    config.val_lmdb, val_sampler, config.num_labels)

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
