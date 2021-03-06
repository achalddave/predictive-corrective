--[[
-- Train a model on image data.
--
-- Example usage:
--    th main.lua \
--      config/config-vgg.yaml \
--      model_output_dir/
--
-- Config options:
--
-- Note that the types listed are YAML types (e.g. list instead of table).
--
-- # General
-- seed (int): Random seed
-- gpus (list): List of GPUs to use
--
-- # Data options
-- data_source_class (str): Data source class to use. See data_source.lua for
--     possible classes. (Default: 'DiskFramesHdf5LabelsDataSource')
-- train_source_options (object): Options to pass train data source.
-- val_source_options (object): Options to pass val data source.
-- num_labels (int): Number of labels.
-- crop_size (int): Size to crop image to before passing ot network.
-- pixel_mean (list of floats): Mean pixel to subtract from images.
--
-- If data_source_class is a data source from lmdb_data_source, the following
-- options are used to construct the data source:
--     data_paths_config (str): Yaml file configuring datasets/splits.
--         Maps keys of dataset/split names to an object containing
--         keys 'with_images' and 'without_images' (whose values are the
--         LMDBs with and without images)
--     train_split (str): Name of dataset/split to use for training. Must
--         be a key in data_paths_config.
--     val_split (str): Name of dataset/split to use for evaluation. Must
--         be a key in data_paths_config.
--
-- # Training options
-- num_epochs (int): Number of epochs to train.
-- epoch_size (int): Number of batches in one epoch.
-- val_epoch_size (int): Number of batches in one evaluation epoch.
--     (Default: epoch_size)
-- batch_size (int): Mini batch size for mini-batch SGD.
-- computational_batch_size (int): The *computational* batch size:
--     how many examples should we pass at once to the network and
--     compute gradients. See Trainer documentation for details.
--     (Default: batch_size)
-- criterion_wrapper (string): Either 'sequencer_criterion' or
--     'last_step_criterion', which will wrap the criterion with
--     nn.SequencerCriterion or nn.LastStepCriterion.
-- sampler_class (string): Sampler class to use. See samplers.lua for possible
--     classes. (Default: 'PermutedSampler')
-- sampler_options (object): Options to be passed to the
--     sampler. See sampler documentation in data_loader.lua.
-- sequence_length (int): Number of steps in a sequence. See Trainer
--     for details. (Default: 1)
-- backprop_rho (int): Number of steps for truncated backprop through time. See
--     Trainer for details. (Default: sequence_length)
-- step_size (int): Step size for sequence. This is equal to 1 + the
--     number of frames between consecutive steps in the sequence.
--     (Default: 1)
-- input_dimension_permutation (list): See Trainer for details.
--     (Default: {1, 2, 3, 4, 5})
-- use_boundary_frames (bool): Whether to use sequences
--     that go beyond video boundaries. See data_loader.lua for
--     details. (Default: false)
-- checkpoint_every: Save a model checkpoint every `checkpoint_every` epoch.
--     (Default: 1)
-- evaluate_every: Evaluate the model every `evaluate_every` epoch.
--     (Default: 1)
--
-- # Optimization options
-- momentum (float)
-- weight_decay (float)
-- gradient_clip (float)
-- learning_rates (list of objects): List containing objects of the
--     form { start_epoch: (int), learning_rate: (float) }.
-- learning_rate_multipliers (list of objects): List specifiying learning rate
--     multipliers to use for some layers. Contains objects of the form
--     { name: (type of layer), index: (int), weight: (float), bias: (float) }.
--     e.g. {name: 'nn.Linear', index: 3, weight: 10, bias: 10} specifies a
--     10x multiplier on the weight and bias of the 3rd nn.Linear layer in the
--     model. (Default: {})
-- dropout_p (float): If specified, update all dropout probabilities
--     for model to this value.
-- optim_config (str): If specified, load optim_config from disk. Note that
--     if this is specified, optim_state must also be specified (and vice
--     versa). (Default: None)
-- optim_state (str): If specified, load optim_state from disk. (Default: None)
--
-- # Model options
-- model_init (str): Path to initial model
-- init_epoch (int): Initial epoch to start training with. Useful for
--     re-starting training.
-- decorate_sequencer (bool): If specified, decorate model with
--     nn.Sequencer.
-- sequencer_remember (string): One of {None, 'eval', 'train', 'neither',
--     'both'}. If not empty or nil, call `:remember(sequencer_remember)` on the
--     model, which is assumed to implement the method (mainly for
--     nn.Sequencer). (Default '')
--]]

local argparse = require 'argparse'
local cutorch = require 'cutorch'
local lyaml = require 'lyaml'
local nn = require 'nn'
local paths = require 'paths'
local torch = require 'torch'
local signal = require 'posix.signal'
local __ = require 'moses'

require 'cudnn'
require 'nnlr'
require 'rnn'
require 'classic'
require 'classic.torch'

local data_loader = require 'data_loader'
local data_source = require 'data_source'
local experiment_saver = require 'util/experiment_saver'
local log = require 'util/log'
local samplers = require 'samplers'
local trainer = require 'trainer'
require 'last_step_criterion'
require 'layers/init'
require 'util/strict'

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
parser:flag('--debug', "Indicates that we are only debugging; " ..
            "Speeds up some things, such as not saving models to disk.")

local args = parser:parse()
if not paths.filep(args.config) then
    log.error(string.format('Config file %s does not exist', args.config))
    os.exit()
end
local config = lyaml.load(io.open(args.config, 'r'):read('*a'))

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
do
    experiment_saver.copy_file(
        args.config, paths.concat(cache_dir, 'config.yaml'))
    local new_config = lyaml.load(io.open(args.config, 'r'):read('*a'))
    assert(__.isEqual(config, new_config),
           'Config updated before it could be copied!')
end
if config.data_paths_config ~= nil then
    experiment_saver.copy_file(
        config.data_paths_config,
        paths.concat(cache_dir, paths.basename(config.data_paths_config)))
end

local function normalize_config(config)
    -- Normalize config files.
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
        log.warn('CONFIG: use_boundary_frames not specified, setting to ' ..
                 tostring(config.use_boundary_frames))
    end
    if config.checkpoint_every == nil then
        config.checkpoint_every = 1
    end
    if config.evaluate_every == nil then
        config.evaluate_every = 1
    end

    if config.computational_batch_size == nil then
        config.computational_batch_size = config.batch_size
    end

    if config.sampling_strategy ~= nil then
        error('config.sampling_strategy[_options] no longer supported. ' ..
              'Please use sampler_class and sampler_options instead.')
    end
    if config.sampling_strategy_options ~= nil then
        error('config.sampling_strategy no longer supported. Please use ' ..
              'sampler_class and sampler_options instead.')
    end
    if config.sampler_class == nil then
        config.sampler_class = 'PermutedSampler'
        log.warn('CONFIG: sampler_class not specified, using ' ..
                 config.sampler_class)
    end

    if config.sampler_options == nil then
        config.sampler_options = {}
    end

    if (config.optim_config == nil) ~= (config.optim_state == nil) then
        error('optim_config and optim_state must either both be specified, ' ..
              'or both left empty')
    end

    if config.learning_rate_multipliers == nil then
        config.learning_rate_multipliers = {}
    end

    if config.data_source_class == nil then
        config.data_source_class = 'DiskFramesHdf5LabelsDataSource'
        log.warn('CONFIG: data_source_class not specified, using ' ..
                 config.data_source_class)
    end
    if config.train_source_options == nil and config.data_source_options ~= nil
        then
        log.warn('DEPRECATED: data_source_options is deprecated. Use ' ..
                 'train/val_source_options instead.')
        config.train_source_options = config.data_source_options
        assert(config.val_source_options == nil)
        config.val_source_options = config.data_source_options
    end

    return config
end

config = normalize_config(config)

do -- Write normalized config to file
    local normalized_config_out = io.open(
        paths.concat(cache_dir, 'normalized-config.yaml'), 'w')
    normalized_config_out:write(lyaml.dump({config}))
    normalized_config_out:close()
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
    -- HACK: Copy model init, but if it's a link, just copy the link instead of
    -- the source file. I'm assuming here that if the model is a symlink, then
    -- it points to some canonical directory of init models that won't be
    -- deleted or moved, and so the link will not break. Of course, this isn't
    -- always true, but this saves me 100s of GBs since each model is >1GB.
    experiment_saver.copy_file(config.model_init,
                               paths.concat(cache_dir, 'model_init.t7'),
                               true --[[preserve]])
end
log.info('Loading model from ' .. config.model_init)
single_model = torch.load(config.model_init)
if torch.isTypeOf(single_model, 'nn.DataParallelTable') then
    log.debug('Getting first of DataParallelTable.')
    single_model = single_model:get(1)
end
-- TODO(achald): XXX HACK! XXX
if config.reinitialize_rate ~= nil then
    single_model:apply(function(m)
        if torch.isTypeOf(m, 'nn.InputCouplerRecurrent') or
                torch.isTypeOf(m, 'nn.InitUpdateRecurrent') or
                torch.isTypeOf(m, 'nn.CCumSumRecurrent') then
            log.info('Updating', torch.type(m), config.reinitialize_rate)
            m.reinitialize_rate = config.reinitialize_rate
        end
    end)
end
single_model:clearState()


if config.criterion_wrapper == nil then
    if torch.isTypeOf(single_model, 'nn.Sequencer') then
        log.warn('Adding LastStepCriterion wrapper for ' ..
                 'nn.Sequencer model since config.criterion_wrapper is unset.')
        config.criterion_wrapper = 'last_step_criterion'
    else
        config.criterion_wrapper = ''
    end
end

-- Increase learning rate of specified layers.
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

-- Increase dropout probability.
if config.dropout_p ~= nil then
   local dropout_layers = single_model:findModules('nn.Dropout')
   for _, layer in ipairs(dropout_layers) do
       local previous_p = layer.p
       layer.p = config.dropout_p
       log.info(string.format('Updating dropout probability from %.2f to %.2f',
                              previous_p, layer.p))
   end
end

if config.decorate_sequencer then
    if torch.isTypeOf(single_model, 'nn.Sequencer') then
        log.warn('WARNING: decorating sequencer on model that is already ' ..
                 'nn.Sequencer!')
    end
    single_model = nn.Sequencer(single_model)
end
if config.sequencer_remember ~= nil then
    single_model:remember(config.sequencer_remember)
    log.debug('Calling :remember with "' .. config.sequencer_remember .. '"')
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

local train_source, val_source
if config.data_source_class == 'LabeledVideoFramesLmdbSource' or
    config.data_source_class == 'PositiveVideosLmdbSource' or
    config.data_source_class == 'SubsampledLmdbSource' then
    -- Legacy code
    require 'lmdb_data_source'
    train_source = data_source[config.data_source_class](
        config.train_lmdb,
        config.train_lmdb_without_images,
        config.num_labels,
        config.train_source_options)
    val_source = data_source[config.data_source_class](
        config.val_lmdb,
        config.val_lmdb_without_images,
        config.num_labels,
        config.val_source_options)
else
    train_source = data_source[config.data_source_class](
        config.train_source_options)
    val_source = data_source[config.data_source_class](
        config.val_source_options)
end
log.info('Loaded data sources')

local train_sampler
if config.train_sampler_init then
    train_sampler = torch.load(config.train_sampler_init)
    log.info('Loaded train sampler from disk.')
else
    train_sampler = samplers[config.sampler_class](
        train_source,
        config.sequence_length,
        config.step_size,
        config.use_boundary_frames,
        config.sampler_options)
    log.info('Initialized train sampler')
end

local val_sampler
if config.sampler_class == 'SequentialSampler' then
    log.warn('CONFIG: Using SequentialSampler for evaluation.')
    val_sampler = samplers.SequentialSampler(
        val_source,
        config.sequence_length,
        config.step_size,
        config.use_boundary_frames,
        config.sampler_options)
else
    log.warn('CONFIG: Using PermutedSampler for evaluation.')
    val_sampler = samplers.PermutedSampler(
        val_source,
        config.sequence_length,
        config.step_size,
        config.use_boundary_frames,
        {replace = false})
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
if config.sampler_class == 'SequentialSampler' then
    log.warn('CONFIG: Using SequentialTrainer for SequentialSampler.')
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
    backprop_rho = config.backprop_rho,
    crop_size = config.crop_size,
    learning_rates = config.learning_rates,
    gradient_clip = config.gradient_clip,
    momentum = config.momentum,
    weight_decay = config.weight_decay,
    optim_config = optim_config,
    optim_state = optim_state,
    use_nnlr = (#config.learning_rate_multipliers ~= 0)
}
log.info('Initialized trainer.')

local epoch = config.init_epoch
local function save_intermediate(epoch)
    trainer:save(cache_dir, epoch)
    torch.save(paths.concat(cache_dir, 'sampler_' .. epoch .. '.t7'),
               train_sampler)
end

if not args.debug then
    log.info('Config:', config)
    save_intermediate(0)
    collectgarbage()
    collectgarbage()

    signal.signal(signal.SIGINT, function(signum)
        log.info('Caught ctrl-c, saving model')
        save_intermediate(epoch)
        os.exit(signum)
    end)
end

local function train_eval_loop()
    -- -- XXX HACK XXX
    -- log.warn('=========================')
    -- log.warn('Using PCA augmentation!!!')
    -- log.warn('=========================')
    -- assert(string.match(config.data_paths_config,
    --                     'config/local_data_paths_charades.yaml'))
    while epoch <= config.num_epochs do
        log.info(('Training epoch %d'):format(epoch))
        trainer:train_epoch(epoch, config.epoch_size)
        collectgarbage()
        collectgarbage()

        if not args.debug and (epoch % config.checkpoint_every == 0 or
                               epoch == config.init_epoch) then
            save_intermediate(epoch)
        end
        collectgarbage()
        collectgarbage()

        if epoch % config.evaluate_every == 0 then
            trainer:evaluate_epoch(epoch, config.val_epoch_size)
        end
        collectgarbage()
        collectgarbage()
        epoch = epoch + 1
    end
end

collectgarbage()
collectgarbage()
-- TODO(achald): Wrap in pcall, save model on error.
train_eval_loop()
-- Proof of concept of pcall below. The issue is that I can't figure out how to
-- raise the exact same error that would have been raised in train_eval_loop.
-- local successful, err = pcall(train_eval_loop)
-- if successful then
--     log.info('Model successfuly trained.')
-- else
--     log.info('Saving model before exiting due to error:')
--     log.error(err)
--     log.error(type(err))
--     if not args.debug then
--         save_intermediate(epoch - 1)
--     end
--     os.exit(1)
-- end
