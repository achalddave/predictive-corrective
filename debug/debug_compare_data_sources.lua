local argparse = require 'argparse'
local image = require 'image'
local torch = require 'torch'

local data_source = require 'data_source'
-- local data_loader = require 'data_loader'
local log = require 'util/log'
local samplers = require 'samplers'
require 'lmdb_data_source'

local parser = argparse() {
    description = 'Compare lmdb loader to disk loader'
}
parser:argument('groundtruth_lmdb')
parser:argument('groundtruth_without_images_lmdb')
parser:argument('frames_root')
parser:argument('labels_hdf5')
parser:option('--num_labels'):default(65):convert(tonumber)

local args = parser:parse()

local lmdb_source = data_source.LabeledVideoFramesLmdbSource(
    args.groundtruth_lmdb,
    args.groundtruth_without_images_lmdb,
    args.num_labels)

local disk_source = data_source.DiskFramesHdf5LabelsDataSource(
    args.frames_root,
    args.labels_hdf5)

-- print('loading label map')
-- local key_label_map_1 = disk_source:key_label_map()
-- print('loaded label map')
-- print('loading label map')
-- local key_label_map = lmdb_source:key_label_map()
-- print('loaded label map')

-- local lmdb_video_keys = lmdb_source:video_keys()
-- local disk_video_keys = disk_source:video_keys()
-- for video, video_keys in pairs(lmdb_source:video_keys()) do
--     if (#video_keys ~= #disk_video_keys[video]) then
--         print('Mismatch counts!', video, #video_keys, #disk_video_keys[video])
--     end
--     print(video, 'matches!')
--     for i, key in ipairs(video_keys) do
--         if (key ~= disk_video_keys[video][i]) then
--             print('Mismatch name!', video, i, key, disk_video_keys[video][i])
--         end
--     end
-- end

local sampler = samplers.PermutedSampler(
    lmdb_source,
    1 --[[sequence_length]],
    1 --[[step_size]],
    true --[[use_boundary_frames]],
    {replace = false})

local keys = sampler:sample_keys(64)
local images1, labels1 = lmdb_source:load_data(keys)
local images2, labels2 = disk_source:load_data(keys)

for i, _ in ipairs(images1) do
    for j, _ in ipairs(images1[i]) do
        print(keys[i][j])
        print(images1[i][j]:float():norm())
        print(images2[i][j]:float():norm())
        -- if i == 1 and j == 1 then
        --     print(keys[i][j])
        --     image.save('lmdb.png', images1[i][j])
        --     image.save('disk.png', images2[i][j])
        -- end
        assert(torch.all(torch.eq(images1[i][j], images2[i][j])))
    end
end
assert(torch.all(torch.eq(labels1, labels2)))
