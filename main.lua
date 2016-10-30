--[[
-- Fine-tunes an ImageNet-pretrained VGG-16 network on MultiTHUMOS data.
--
-- Example usage:
--    th main.lua \
--      config/config-vgg.yaml \
--      model_output_dir/ \
--      | tee model_output_dir/training.log
--]]

local argparse = require 'argparse'
local cudnn = require 'cudnn'
local cutorch = require 'cutorch'
local lyaml = require 'lyaml'
local nn = require 'nn'
local paths = require 'paths'
local torch = require 'torch'
require 'rnn'
require 'last_step_criterion'
require 'classic'
require 'classic.torch'

local data_loader = require 'data_loader'
local evaluator = require 'evaluator'
local experiment_saver = require 'util/experiment_saver'
local trainer = require 'trainer'
require 'layers/init'

local parser = argparse() {
    description = 'Fine tune ImageNet-pretrained VGG-16 network on MultiTHUMOS.'
}
parser:argument('config', 'Config file')
parser:argument('cache_base',
                'Directory to save model snapshots, logging, etc. to.')
parser:option('--experiment_id_file',
              'Path to text file containing the experiment id for this run.' ..
              'The id in this file will be incremented by this program.')
              :count(1)
              :default('/data/achald/MultiTHUMOS/models/next_experiment_id.txt')
parser:flag('--decorate_sequencer',
            'If specified, decorate model with nn.Sequencer.' ..
            'This is necessary if the model does not expect a table as ' ..
            'input.')

local args = parser:parse()
local config = lyaml.load(io.open(args.config, 'r'):read('*a'))

if config.data_paths_config ~= nil then
    local data_paths = lyaml.load(
        io.open(config.data_paths_config, 'r'):read('*a'))
    local train_path = data_paths[config.train_split]
    local val_path = data_paths[config.val_split]
    config.train_lmdb = train_path.with_images
    config.train_lmdb_without_images = train_path.without_images
    config.val_lmdb = val_path.with_images
    config.val_lmdb_without_images = val_path.without_images
end

config.sequence_length = config.sequence_length == nil
                         and 1
                         or config.sequence_length
config.step_size = config.step_size == nil and 1 or config.step_size
if config.input_dimension_permutation == nil then
    config.input_dimension_permutation = {1, 2, 3, 4, 5}
end
if config.use_boundary_frames == nil then
    config.use_boundary_frames = false
end
-- TODO(achald): Validate config.

-- Create cache_base
if not paths.dirp(args.cache_base) and not paths.mkdir(args.cache_base) then
    print('Error creating cache base dir:', args.cache_base)
    os.exit()
end
local cache_dir = paths.concat(args.cache_base, os.date('%m-%d-%y-%H-%M-%S'))
if not paths.mkdir(cache_dir) then
    print('Error making cache dir:', cache_dir)
    os.exit()
end
experiment_saver.save_git_info(cache_dir)
print('Saving run information to', cache_dir)

-- Save config to cache_dir
experiment_saver.copy_file(args.config, paths.concat(cache_dir, 'config.yaml'))
if config.data_paths_config ~= nil then
    experiment_saver.copy_file(
        config.data_paths_config,
        paths.concat(cache_dir, paths.basename(config.data_paths_config)))
end

local experiment_id = experiment_saver.read_and_increment_experiment_id(
    args.experiment_id_file)
experiment_id_output = io.open(
    paths.concat(cache_dir, 'experiment-id.txt'), 'w')
experiment_id_output:write(experiment_id)
experiment_id_output:close()
print('===')
print('Experiment id:', experiment_id)
print('===')

cutorch.setDevice(config.gpus[1])
math.randomseed(config.seed)
torch.manualSeed(config.seed)
cutorch.manualSeedAll(config.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- Load model
local single_model
assert(config.model_init ~= nil, 'Initial model must be specified.')
experiment_saver.copy_file(config.model_init,
                           paths.concat(cache_dir, 'model_init.t7'))
print('Loading model from' .. config.model_init)
single_model = torch.load(config.model_init)

if torch.isTypeOf(single_model, 'nn.DataParallelTable') then
    print('Getting first of DataParallelTable.')
    single_model = single_model:get(1)
end
if args.decorate_sequencer then
    if torch.isTypeOf(single_model, 'nn.Sequencer') then
        print('WARNING: --decorate_sequencer on model that is already ' ..
              'nn.Sequencer!')
    end
    single_model = nn.Sequencer(single_model)
end
local batch_dimension = 2 -- by default
for i = 1, 5 do
    if config.input_dimension_permutation[i] == batch_dimension then
        -- batch_dimension will be permuted to be i'th dimension
        batch_dimension = i
        break
    end
end
local model = nn.DataParallelTable(batch_dimension)
for _, gpu in ipairs(config.gpus) do
    cutorch.setDevice(gpu)
    model:add(single_model:clone():cuda(), gpu)
end
cutorch.setDevice(config.gpus[1])
-- https://groups.google.com/forum/#!topic/torch7/HiBymc9NfIY
model = model:cuda()
local criterion = nn.MultiLabelSoftMarginCriterion():cuda()
if config.criterion_wrapper == nil then
    if torch.isTypeOf(single_model, 'nn.Sequencer') then
        print('nn.Sequencer models are wrapped by LastStepCriterion if ' ..
            'config.criterion_wrapper is not set.')
        config.criterion_wrapper = 'last_step_criterion'
    else
        config.criterion_wrapper = ''
    end
end

if config.criterion_wrapper:lower() == 'last_step_criterion' then
    criterion = nn.LastStepCriterion(criterion)
elseif config.criterion_wrapper:lower() == 'sequencer_criterion' then
    criterion = nn.SequencerCriterion(criterion)
end
print('Loaded model')

local sampling_strategies = {
    permuted = data_loader.PermutedSampler,
    balanced = data_loader.BalancedSampler,
    sequential = data_loader.SequentialSampler
}

config.sampling_strategy_options = config.sampling_strategy_options == nil and
    {} or config.sampling_strategy_options

if sampling_strategies[config.sampling_strategy:lower()] ==
        sampling_strategies.sequential then
    config.sampling_strategy_options.batch_size = config.batch_size
end

local train_sampler = sampling_strategies[config.sampling_strategy:lower()](
    config.train_lmdb_without_images,
    config.num_labels,
    config.sequence_length,
    config.step_size,
    config.use_boundary_frames,
    config.sampling_strategy_options)
local val_sampler = data_loader.PermutedSampler(
    config.val_lmdb_without_images,
    config.num_labels,
    config.sequence_length,
    config.step_size,
    config.use_boundary_frames)

local train_loader = data_loader.DataLoader(
    config.train_lmdb, train_sampler, config.num_labels)
local val_loader = data_loader.DataLoader(
    config.val_lmdb, val_sampler, config.num_labels)

local optim_config, optim_state
if config.optim_config ~= nil and config.optim_state ~= nil then
    optim_config = torch.load(config.optim_config)
    optim_state = torch.load(config.optim_state)
    print('Loading optim_config, optim_state from disk.')
elseif not (config.optim_config == nil and config.optim_state == nil) then
    error('optim_config and optim_state must either both be specified, ' ..
          'or both left empty')
end

local trainer_class, evaluator_class
if sampling_strategies[config.sampling_strategy:lower()] ==
        sampling_strategies.sequential then
    trainer_class = trainer.SequentialTrainer
    evaluator_class = evaluator.SequentialEvaluator
else
    trainer_class = trainer.Trainer
    evaluator_class = evaluator.Evaluator
end

local trainer = trainer_class {
    model = model,
    criterion = criterion,
    data_loader = train_loader,
    input_dimension_permutation = config.input_dimension_permutation,
    pixel_mean = config.pixel_mean,
    batch_size = config.batch_size,
    crop_size = config.crop_size,
    num_labels = config.num_labels,
    learning_rates = config.learning_rates,
    momentum = config.momentum,
    weight_decay = config.weight_decay,
    optim_config = optim_config,
    optim_state = optim_state
}
local evaluator = evaluator_class {
    model = model,
    criterion = criterion,
    data_loader = val_loader,
    input_dimension_permutation = config.input_dimension_permutation,
    pixel_mean = config.pixel_mean,
    batch_size = config.batch_size,
    crop_size = config.crop_size,
    num_labels = config.num_labels
}

print('Initialized trainer and evaluator.')
print('Config:')
print(config)
trainer:save(cache_dir, 0)
for epoch = config.init_epoch, config.num_epochs do
    print(('Training epoch %d'):format(epoch))
    trainer:train_epoch(epoch, config.epoch_size)
    trainer:save(cache_dir, epoch)
    collectgarbage()
    collectgarbage()

    evaluator:evaluate_epoch(epoch, config.epoch_size)
    collectgarbage()
    collectgarbage()
end
