from __future__ import division

import argparse

import torchfile
import numpy as np

from colorsys import hsv_to_rgb
from moviepy.editor import VideoClip, VideoFileClip, clips_array
from moviepy.video.io.bindings import mplfig_to_npimage
from PIL import Image


def extract_max_var_filter(activations):
    max_var_filter = None
    max_var = -float('inf')
    num_filters = activations.shape[1]
    for i in range(activations.shape[1]):
        curr_var = activations[:, i].var()
        if curr_var > max_var:
            max_var = curr_var
            max_var_filter = i
    print 'Max var filter: ', max_var_filter
    return activations[:, max_var_filter]


def compute_offset_maxpool_avg(activations, offset):
    """
    Maxpool 'left' frame and average it with 'right' frame.

    Specifically, replace each pixel in the left frame with the max value in a
    3x3 neighborhood, then average this updated left frame with the right
    frame.

    Args:
        activations (np.array, shape (num_frames, height, width))
        offset (int)
    """
    activations = activations.copy()
    if offset == 0:
        # Needs to be special cased as activations[:-0] returns an empty array.
        return activations
    # Set activations_i = activations_i - activations_{i - offset}
    pooled_activations = activations.copy()
    for i in range(1, activations.shape[1]):
        for j in range(1, activations.shape[2]):
            pooled_activations[:, i, j] = map(
                np.max, activations[:, i - 1:i + 2, j - 1:j + 2])
    activations[offset:] = 0.5 * (
        activations[offset:] + pooled_activations[:-offset])
    return activations


def compute_offset_avg(activations, offset):
    activations = activations.copy()
    if offset == 0:
        # Needs to be special cased as activations[:-0] returns an empty array.
        return activations
    # Set activations_i = activations_i - activations_{i - offset}
    activations[offset:] = (activations[offset:] + activations[:-offset]) / 2
    return activations


def compute_offset_diff(activations, offset):
    """
    >>> fib = np.array([1, 1, 2, 3, 5, 8, 13])
    >>> diff0 = np.array([0, 0, 0, 0, 0, 0, 0])
    >>> diff1 = np.array([0, 0, 1, 1, 2, 3, 5])
    >>> diff2 = np.array([0, 0, 1, 2, 3, 5, 8])
    >>> diff3 = np.array([0, 0, 0, 2, 4, 6, 10])
    >>> diff6 = np.array([0, 0, 0, 0, 0, 0, 12])
    >>> diff7 = np.array([0, 0, 0, 0, 0, 0, 0])
    >>> diff15 = np.array([0, 0, 0, 0, 0, 0, 0])
    >>> assert(all(compute_offset_diff(fib, 0) == diff0))
    >>> assert(all(compute_offset_diff(fib, 1) == diff1))
    >>> assert(all(compute_offset_diff(fib, 2) == diff2))
    >>> assert(all(compute_offset_diff(fib, 3) == diff3))
    >>> assert(all(compute_offset_diff(fib, 6) == diff6))
    >>> assert(all(compute_offset_diff(fib, 7) == diff7))
    >>> assert(all(compute_offset_diff(fib, 15) == diff15))
    """
    activations = activations.copy()
    if offset == 0:
        # Needs to be special cased as activations[:-0] returns an empty array.
        activations[:] = 0
        return activations
    # Set activations_i = activations_i - activations_{i - offset}
    activations[offset:] = activations[offset:] - activations[:-offset]
    activations[:offset] = 0
    return activations


def compute_offset_hdiffmap(activations, offset):
    activations = compute_offset_vdiffmap(
        activations.transpose((0, 2, 1)), offset)
    return activations.transpose((0, 2, 1))


def compute_offset_vdiffmap(activations, offset):
    vdiffs = np.zeros((3, activations.shape[0], activations.shape[1],
                       activations.shape[2]))
    right_frames = activations[offset:]
    left_frames = activations[:-offset]
    # vdiffs[0, f, r] = activations[f, r] - activations[f - offset, r - 1]
    vdiffs[0, offset:, 1:] = right_frames[:, 1:] - left_frames[:, :-1]
    # vdiffs[1, f, r] = activations[f, r] - activations[f - offset, r]
    vdiffs[1, offset:] = right_frames - left_frames
    # vdiffs[2, f, r] = activations[f, r] - activations[f - offset, r + 1]
    vdiffs[2, offset:, :-1] = right_frames[:, :-1] - left_frames[:, 1:]
    return abs(vdiffs).argmin(axis=0).astype(float)


def cosine_similarity(left_activations, right_activations):
    """
    Compute cosine similarity between activations.

    Args:
        left_activations (np.array): 4D array of shape
            (num_frames, num_filters, height, width)
        right_activations (np.array): 4D array of shape
            (num_frames, num_filters, height, width)
    Returns:
        distances (np.array): 3D array of shape
            (num_frames, height, width)
    """
    # dot_products[t, i, j] = \sum_f (left_activations[t, f, i, j] *
    #                                 right_activations[t, f, i, j])
    dot_products = (left_activations * right_activations).sum(axis=1)
    left_norms = np.linalg.norm(left_activations, axis=1)
    right_norms = np.linalg.norm(right_activations, axis=1)
    return (dot_products / left_norms) / right_norms


def compute_offset_alignment(activations, offset):
    """
    Compute alignment between frames separated by an offset.

    Computes alignment by computing the nearest neighbor in a 3x3 spatial
    neighborhood in the left frame for each pixel in the right frame, using
    cosine distance on the feature activations.

    Args:
        activations (np.array, shape (num_frames, num_filters, height, width))
        offset (int)
    """
    # TOP_LEFT implies the pixel is aligned with the pixel to the top left in
    # the next frame.
    num_frames = activations.shape[0]
    height = activations.shape[2]
    width = activations.shape[3]
    similarities = np.zeros((9, num_frames - offset, height, width))
    # Shape (num_filters, num_frames - offset, height, width)
    left_frames = activations[:-offset]
    right_frames = activations[offset:]
    (CENTER, LEFT, TOP_LEFT, TOP, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM,
     BOTTOM_LEFT) = range(9)

    colors = np.zeros((9, 3))
    colors[CENTER] = hsv_to_rgb(0, 1, 0)
    colors[RIGHT] = hsv_to_rgb(0, 1, 1)
    colors[TOP_RIGHT] = hsv_to_rgb(0.125, 1, 1)
    colors[TOP] = hsv_to_rgb(0.25, 1, 1)
    colors[TOP_LEFT] = hsv_to_rgb(0.375, 1, 1)
    colors[LEFT] = hsv_to_rgb(0.5, 1, 1)
    colors[BOTTOM_LEFT] = hsv_to_rgb(0.625, 1, 1)
    colors[BOTTOM] = hsv_to_rgb(0.75, 1, 1)
    colors[BOTTOM_RIGHT] = hsv_to_rgb(0.875, 1, 1)

    # I'm very sad about this code, but can't seem to improve it.
    similarities[CENTER] = cosine_similarity(left_frames, right_frames)
    similarities[LEFT, :, :, 1:] = cosine_similarity(
        left_frames[:, :, :, 1:], right_frames[:, :, :, :-1])
    similarities[RIGHT, :, :, :-1] = cosine_similarity(
        left_frames[:, :, :, :-1], right_frames[:, :, :, 1:])
    similarities[TOP, :, 1:] = cosine_similarity(
        left_frames[:, :, 1:], right_frames[:, :, :-1])
    similarities[BOTTOM, :, :-1] = cosine_similarity(
        left_frames[:, :, :-1], right_frames[:, :, 1:])

    similarities[BOTTOM_LEFT, :, :-1, 1:] = cosine_similarity(
            left_frames[:, :, :-1, 1:], right_frames[:, :, 1:, :-1])
    similarities[TOP_LEFT, :, 1:, 1:] = cosine_similarity(
            left_frames[:, :, 1:, 1:], right_frames[:, :, :-1, :-1])
    similarities[TOP_RIGHT, :, 1:, :-1] = cosine_similarity(
            left_frames[:, :, 1:, :-1], right_frames[:, :, :-1, 1:])
    similarities[BOTTOM_RIGHT, :, :-1, :-1] = cosine_similarity(
            left_frames[:, :, :-1, :-1], right_frames[:, :, 1:, 1:])

    # (num_frames, height, width)
    best_align = np.argmax(similarities, axis=0)
    output = np.zeros((num_frames, height, width, 3))
    for i in range(3):
        output[:-offset, :, :, i] = colors[best_align, i]
    return output


def main():
    # Use first line of file docstring as description.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--activations_t7', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument(
        '--filter',
        help="""Either a number, or one of "max", "avg", "max_var".
                Ignored if --offset_action is align.""")
    parser.add_argument(
        '--offset_action',
        choices=['diff', 'avg', 'maxpool-avg', 'vdiffmap', 'hdiffmap', 'align'
                 ],
        help="""Optional action to perform on frames at --offset.""")
    parser.add_argument(
            '--offset',
            type=int,
            help='Show difference between activations at diff_offset.',
            default=1)
    parser.add_argument('--frames_per_second', type=int, required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    # (num_frames, num_filters, height, width) array.
    all_activations = torchfile.load(args.activations_t7)
    if args.offset_action != 'align':
        if args.filter == 'max_var':
            activations = extract_max_var_filter(all_activations)
        elif args.filter == 'max':
            activations = all_activations.max(axis=1)
        elif args.filter == 'avg':
            activations = all_activations.mean(axis=1)
        else:
            filter_index = None
            try:
                filter_index = int(args.filter)
            except ValueError:
                print('Invalid --filter option, see --help.')
                raise
            activations = all_activations[:, filter_index]

    if args.offset_action == 'diff':
        activations = compute_offset_diff(activations, args.offset)
    elif args.offset_action == 'avg':
        activations = compute_offset_avg(activations, args.offset)
    elif args.offset_action == 'maxpool-avg':
        activations = compute_offset_maxpool_avg(activations, args.offset)
    elif args.offset_action == 'vdiffmap':
        activations = compute_offset_vdiffmap(activations, args.offset)
    elif args.offset_action == 'hdiffmap':
        activations = compute_offset_hdiffmap(activations, args.offset)
    elif args.offset_action == 'align':
        # Pass all filter activations for alignment.
        activations = compute_offset_alignment(all_activations, args.offset)

    min_value = activations.min()
    max_value = activations.max()
    activations = (activations - min_value) / (max_value - min_value) * 255

    old_height = activations.shape[1]
    old_width = activations.shape[2]
    new_width, new_height = None, None
    if old_height > old_width:
        new_height = 500.
        new_width = new_height / old_height * old_width
    else:
        new_width = 500.
        new_height = new_width / old_width * old_height

    def make_frame(t_second):
        frame_index = int(t_second * args.frames_per_second)
        frame_activations = activations[frame_index]
        if frame_activations.ndim != 3 or frame_activations.shape[2] == 1:
            # Replicate frame_activations into 3 channels.
            frame_activations = np.tile(
                frame_activations[:, :, np.newaxis], (1, 1, 3))
        image = Image.fromarray(np.uint8(frame_activations))
        image = image.resize((int(new_width), int(new_height)))
        return np.array(image)

    num_frames = activations.shape[0]
    duration = num_frames / args.frames_per_second

    activations_clip = VideoClip(make_frame, duration=duration)
    video_clip = VideoFileClip(args.video)
    video_clip = video_clip.resize((new_height, new_width))

    final_clip = clips_array([[activations_clip, video_clip]])
    final_clip.write_videofile(args.output, fps=args.frames_per_second)


if __name__ == "__main__":
    main()
