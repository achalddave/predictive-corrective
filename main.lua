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
local signal = require 'posix.signal'

require 'nnlr'
require 'rnn'
require 'classic'
require 'classic.torch'

local data_loader = require 'data_loader'
local data_source = require 'data_source'
local experiment_saver = require 'util/experiment_saver'
local log = require 'util/log'
local trainer = require 'trainer'
require 'last_step_criterion'
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
parser:flag('--debug', "Indicates that we are only debugging; " ..
            "Speeds up some things, such as not saving models to disk.")

local args = parser:parse()
local config = lyaml.load(io.open(args.config, 'r'):read('*a'))

function normalize_config(config)
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

    if config.computational_batch_size == nil then
        config.computational_batch_size = config.batch_size
    end
    if config.val_batch_size == nil then
        config.val_batch_size = config.batch_size
    end
    if config.val_epoch_size == nil then
        config.val_epoch_size = config.epoch_size
    end

    if config.sampling_strategy_options == nil then
        config.sampling_strategy_options = {}
    end
    if config.sampling_strategy:lower() == 'sequential' and
            config.sampling_strategy_options.batch_size == nil then
        config.sampling_strategy_options.batch_size = config.batch_size
    end

    if (config.optim_config == nil) ~= (config.optim_state == nil) then
        error('optim_config and optim_state must either both be specified, ' ..
            'or both left empty')
    end

    if config.learning_rate_multipliers == nil then
        config.learning_rate_multipliers = {}
    end
    return config
end

config = normalize_config(config)

-- Create cache_base
if not paths.dirp(args.cache_base) and not paths.mkdir(args.cache_base) then
    log.error('Error creating cache base dir:', args.cache_base)
    os.exit()
end
local cache_dir = paths.concat(args.cache_base, os.date('%m-%d-%y-%H-%M-%S'))
if not paths.mkdir(cache_dir) then
    log.error('Error making cache dir:', cache_dir)
    os.exit()
end
log.outfile = paths.concat(cache_dir, 'training.log')
experiment_saver.save_git_info(cache_dir)
log.info('Saving run information to', cache_dir)

-- Save config to cache_dir
experiment_saver.copy_file(args.config, paths.concat(cache_dir, 'config.yaml'))
if config.data_paths_config ~= nil then
    experiment_saver.copy_file(
        config.data_paths_config,
        paths.concat(cache_dir, paths.basename(config.data_paths_config)))
end

local experiment_id = experiment_saver.read_and_increment_experiment_id(
    args.experiment_id_file)
local experiment_id_output = io.open(
    paths.concat(cache_dir, 'experiment-id.txt'), 'w')
experiment_id_output:write(experiment_id)
experiment_id_output:close()
log.info('===')
log.info('Experiment id:', experiment_id)
log.info('===')

cutorch.setDevice(config.gpus[1])
math.randomseed(config.seed)
torch.manualSeed(config.seed)
cutorch.manualSeedAll(config.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- Load model
local single_model
assert(config.model_init ~= nil, 'Initial model must be specified.')
if not args.debug then
    experiment_saver.copy_file(config.model_init,
                               paths.concat(cache_dir, 'model_init.t7'))
end
log.info('Loading model from ' .. config.model_init)
single_model = torch.load(config.model_init)
single_model:clearState()
if config.criterion_wrapper == nil then
    if torch.isTypeOf(single_model, 'nn.Sequencer') then
        log.info('nn.Sequencer models are wrapped by LastStepCriterion if ' ..
               'config.criterion_wrapper is not set.')
        config.criterion_wrapper = 'last_step_criterion'
    else
        config.criterion_wrapper = ''
    end
end

-- Increase learning rate of last nn.Linear layer.
for _, multiplier_spec in ipairs(config.learning_rate_multipliers) do
    local layers = single_model:findModules(multiplier_spec.name)
    layers[multiplier_spec.index]:learningRate('weight', multiplier_spec.weight)
                                 :learningRate('bias', multiplier_spec.bias)
    log.info(string.format(
        'Multiplier for %s layer, index %d. Weight: %d, Bias: %d',
        multiplier_spec.name,
        multiplier_spec.index,
        multiplier_spec.weight,
        multiplier_spec.bias))
end

-- -- Increase dropout probability.
if config.dropout_p ~= nil then
   local dropout_layers = single_model:findModules('nn.Dropout')
   for _, layer in ipairs(dropout_layers) do
       local previous_p = layer.p
       layer.p = config.dropout_p
       log.info(string.format('Increasing dropout probability from %.2f to %.2f',
                            previous_p, layer.p))
   end
end

if torch.isTypeOf(single_model, 'nn.DataParallelTable') then
    log.info('Getting first of DataParallelTable.')
    single_model = single_model:get(1)
end
if args.decorate_sequencer then
    if torch.isTypeOf(single_model, 'nn.Sequencer') then
        log.info('WARNING: --decorate_sequencer on model that is already ' ..
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
model:add(single_model, config.gpus)
cutorch.setDevice(config.gpus[1])
model = model:cuda()

local criterion = nn.MultiLabelSoftMarginCriterion():cuda()

single_model = nil
collectgarbage()
collectgarbage()

if config.criterion_wrapper:lower() == 'last_step_criterion' then
    criterion = nn.LastStepCriterion(criterion)
elseif config.criterion_wrapper:lower() == 'sequencer_criterion' then
    criterion = nn.SequencerCriterion(criterion)
elseif config.criterion_wrapper ~= '' then
    error('Unknown criterion wrapper', config.criterion_wraper)
end
log.info('Loaded model')

local sampling_strategies = {
    permuted = data_loader.PermutedSampler,
    balanced = data_loader.BalancedSampler,
    sequential = data_loader.SequentialSampler
}

local train_source = data_source.LabeledVideoFramesLmdbSource(
    config.train_lmdb, config.train_lmdb_without_images, config.num_labels)
local val_source = data_source.LabeledVideoFramesLmdbSource(
    config.val_lmdb, config.val_lmdb_without_images, config.num_labels)
log.info('Loaded data sources')

local train_sampler
if config.train_sampler_init then
    train_sampler = torch.load(config.train_sampler_init)
    log.info('Loaded train sampler from disk.')
else
    train_sampler = sampling_strategies[config.sampling_strategy:lower()](
        train_source,
        config.sequence_length,
        config.step_size,
        config.use_boundary_frames,
        config.sampling_strategy_options)
    log.info('Initialized train sampler')
end

local val_sampler
if config.sampling_strategy:lower() == 'sequential' then
    val_sampler = data_loader.SequentialSampler(
        val_source,
        config.sequence_length,
        config.step_size,
        config.use_boundary_frames,
        config.sampling_strategy_options)
else
    val_sampler = data_loader.PermutedSampler(
        val_source,
        config.sequence_length,
        config.step_size,
        config.use_boundary_frames)
end
log.info('Initialized val sampler')

local train_loader = data_loader.DataLoader(train_source, train_sampler)
local val_loader = data_loader.DataLoader(val_source, val_sampler)
log.info('Initialized data loaders')

local optim_config, optim_state
if config.optim_config ~= nil and config.optim_state ~= nil then
    optim_config = torch.load(config.optim_config)
    optim_state = torch.load(config.optim_state)
    log.info('Loading optim_config, optim_state from disk.')
end

local trainer_class
if sampling_strategies[config.sampling_strategy:lower()] ==
        sampling_strategies.sequential then
    trainer_class = trainer.SequentialTrainer
else
    trainer_class = trainer.Trainer
end

local trainer = trainer_class {
    model = model,
    criterion = criterion,
    train_data_loader = train_loader,
    val_data_loader = val_loader,
    input_dimension_permutation = config.input_dimension_permutation,
    pixel_mean = config.pixel_mean,
    batch_size = config.batch_size,
    computational_batch_size = config.computational_batch_size,
    crop_size = config.crop_size,
    num_labels = config.num_labels,
    learning_rates = config.learning_rates,
    momentum = config.momentum,
    weight_decay = config.weight_decay,
    optim_config = optim_config,
    optim_state = optim_state,
    use_nnlr = (#config.learning_rate_multipliers ~= 0)
}

log.info('Initialized trainer.')
if not args.debug then
    log.info('Config:')
    log.info(config)
    trainer:save(cache_dir, 0)
end
collectgarbage()
collectgarbage()

local epoch = config.init_epoch
function save_intermediate(epoch)
    trainer:save(cache_dir, epoch)
    torch.save(paths.concat(cache_dir, 'sampler_' .. epoch .. '.t7'),
            train_sampler)
end

if not args.debug then
    signal.signal(signal.SIGINT, function(signum)
        log.info('Caught ctrl-c, saving model')
        save_intermediate(epoch - 1)
        os.exit(signum)
    end)
end

while epoch <= config.num_epochs do
    log.info(('Training epoch %d'):format(epoch))
    trainer:train_epoch(epoch, config.epoch_size)
    if not args.debug and (epoch % 5 == 0 or epoch == config.init_epoch) then
        -- TODO: Add a signal handler that saves the model on SIGINT/ctrl-c.
        save_intermediate(epoch)
    end
    collectgarbage()
    collectgarbage()

    trainer:evaluate_epoch(epoch, config.val_epoch_size)
    collectgarbage()
    collectgarbage()
    epoch = epoch + 1
end
