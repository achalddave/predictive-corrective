--[[
Helper class to load data and labels from an LMDB containing LabeledVideoFrames
as values.
]]--

local classic = require 'classic'
local threads = require 'threads'
require 'torch'
require 'classic.torch'

local log = require 'util/log' -- luacheck: no unused

local DataLoader = classic.class('DataLoader')

function DataLoader:_init(data_source_obj, sampler)
    --[[
    Args:
        data_source: Data source object.
        sampler (Sampler): Sampler used for batches
    ]]--
    self.data_source = data_source_obj
    self.sampler = sampler
    self._prefetched_data = {
        batch_images = nil,
        batch_labels = nil
    }
    self._prefetching_thread = threads.Threads(1)
end

function DataLoader:num_labels()
    return self.sampler:num_labels()
end

function DataLoader:num_samples()
    return self.sampler:num_samples()
end

function DataLoader:load_batch(batch_size, return_keys)
    --[[
    Load a batch of images and labels.

    Args:
        batch_size (num)
        return_keys (bool): Whether to return the keys from the batch.
            Default false.

    Returns:
        images (Array of array of ByteTensors): Contains image sequences for
            the batch. Each element is a step in the sequence, so that
            #images = sequence_length, #images[1] = batch_size.
        labels (ByteTensor): Contains label ids. Size is
            (sequence_length, batch_size, num_labels)
        keys (Array of array of strings): Only returned if return_keys is
            True.
    ]]--
    return_keys = return_keys == nil and false or return_keys
    if not self:_data_fetched() then
        self:fetch_batch_async(batch_size)
        self._prefetching_thread:synchronize()
    end

    local batch_images = self._prefetched_data.batch_images
    local batch_labels = self._prefetched_data.batch_labels
    local batch_keys = self._prefetched_data.batch_keys
    self._prefetched_data.batch_images = nil
    self._prefetched_data.batch_labels = nil
    self._prefetched_data.batch_keys = nil
    if return_keys then
        return batch_images, batch_labels, batch_keys
    else
        return batch_images, batch_labels
    end
end

function DataLoader:fetch_batch_async(batch_size)
    --[[ Load a batch, store it for returning in next call to load_batch. ]]--
    if self:_data_fetched() then
        return
    end

    local batch_keys = self.sampler:sample_keys(batch_size)
    local data_source_obj = self.data_source

    self._prefetching_thread:addjob(
        function()
            require 'torch'
            require 'classic'
            require 'classic.torch'
            require 'data_source'
        end,
        function()
            local batch_images, batch_labels = data_source_obj:load_data(
                batch_keys)
            self._prefetched_data = {
                batch_images = batch_images,
                batch_labels = batch_labels,
                batch_keys = batch_keys
            }
        end)
end

function DataLoader:_data_fetched()
    -- Wait for possible fetching thread to finish.
    self._prefetching_thread:synchronize()
    return self._prefetched_data.batch_images ~= nil
end

return { DataLoader = DataLoader }
