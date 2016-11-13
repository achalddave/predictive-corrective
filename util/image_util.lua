local image = require 'image'
local torch = require 'torch'
require 'cudnn'
require 'cutorch'

local AUGMENT_MODE = { TRAIN = 1, EVAL = 2 }

local function random_crop(img, crop_width, crop_height, state)
    --[[ Randomly crop an image.
    --
    -- Args:
    --     img
    --     crop_width, crop_height
    --     state: Optional opaque object returned by an earlier call to this
    --         function. If specified, then the same random crop coordinates
    --         will be used to crop this img as in the earlier call.
    --
    -- Returns:
    --      cropped_img
    --      state
    --]]
    local width = img:size(3)
    local height = img:size(2)
    if state == nil then
        state = {}
        state.x_origin = math.random(width - crop_width)
        state.y_origin = math.random(height - crop_height)
    end
    img = image.crop(img, state.x_origin, state.y_origin,
                     state.x_origin + crop_width, state.y_origin + crop_width)
    return img, state
end

local function random_mirror(img, state)
    --[[ Mirror an image with 0.5 probability.
    --
    -- Args:
    --     img
    --     state: Optional opaque object returned by an earlier call to this
    --         function. If specified, then the img will be mirrored iff the
    --         original img was mirrored.
    -- Returns:
    --     maybe_mirrored_img
    --     state
    --]]
    if state == nil then
        state = {}
        state.mirror = torch.uniform() > 0.5
    end
    if state.mirror then
        img = image.hflip(img)
    end
    return img, state
end

local function subtract_pixel_mean(img, pixel_mean)
    --[[ Subtract a pixel-wise mean from the image.
    --
    -- Args:
    --     img
    --     pixel_mean (torch.Tensor): Tensor containing 3 numbers (the mean for
    --         each of the R, G, and B channels).
    -- Returns:
    --     mean_subtracted_img
    --]]
    for channel = 1, 3 do
        -- Subtract mean
        img[{{channel}, {}, {}}]:add(-pixel_mean[channel])
     end
     return img
end

local function augment_image(
    img, crop_width, crop_height, pixel_mean, mode, state)
    --[[ Process image by cropping, mirroring, and subtracting mean.
    --
    -- Args:
    --     img
    --     crop_width, crop_height
    --     pixel_mean
    --     mode: One of AUGMENT_MODE.TRAIN or AUGMENT_MODE.EVAL.
    --     state: Optional opaque object returned by an earlier call to
    --         _process. In mode = TRAIN, providing state leads to this img
    --         being augmented in the same way as the earlier call to this
    --         function. This is useful, for example, if we want to process a
    --         sequence of frames in the same manner (same random crops, same
    --         mirror, etc.).
    --
    -- Returns:
    --     img
    --     state
    --]]
    -- Avoid wrap around for ByteTensors, which are unsigned.
    assert(img:type() ~= torch.ByteTensor():type())
    assert(mode == AUGMENT_MODE.EVAL or mode == AUGMENT_MODE.TRAIN)

    if state == nil then state = {} end
    -- Randomly crop.
    if mode == AUGMENT_MODE.EVAL then
        img = image.crop(
            img, "c" --[[center crop]], crop_width, crop_height)
    else
        img, state.crop_state = random_crop(
            img, crop_width, crop_height, state.crop_state)
    end

    -- Mirror horizontally with probability 0.5.
    if mode == AUGMENT_MODE.TRAIN then
        img, state.mirror_state = random_mirror(img, state.mirror_state)
    end

    -- Subtract mean.
    img = subtract_pixel_mean(img, pixel_mean)

    -- TODO: Do we need to divide by STD?

    return img, state
end

local function augment_image_eval(img, crop_width, crop_height, pixel_mean)
    return augment_image(
        img, crop_width, crop_height, pixel_mean, AUGMENT_MODE.EVAL)
end

local function augment_image_train(
        img, crop_width, crop_height, pixel_mean, state)
    return augment_image(
        img, crop_width, crop_height, pixel_mean, AUGMENT_MODE.TRAIN, state)
end

return {
    augment_image_train = augment_image_train,
    augment_image_eval = augment_image_eval,
    subtract_pixel_mean = subtract_pixel_mean
}
