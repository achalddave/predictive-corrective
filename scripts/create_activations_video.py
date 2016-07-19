import argparse

import torchfile
import numpy as np

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
    current_frames = activations[offset:]
    previous_frames = activations[:-offset]
    # vdiffs[0, f, r] = activations[f, r] - activations[f - offset, r - 1]
    vdiffs[0, offset:, 1:] = current_frames[:, 1:] - previous_frames[:, :-1]
    # vdiffs[1, f, r] = activations[f, r] - activations[f - offset, r]
    vdiffs[1, offset:] = current_frames - previous_frames
    # vdiffs[2, f, r] = activations[f, r] - activations[f - offset, r + 1]
    vdiffs[2, offset:, :-1] = current_frames[:, :-1] - previous_frames[:, 1:]
    return abs(vdiffs).argmin(axis=0).astype(float)


def main():
    # Use first line of file docstring as description.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--activations_t7', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument(
        '--filter',
        help='Either a number, or one of "max", "avg", "max_var"',
        required=True)
    parser.add_argument(
            '--offset_action',
            choices=['diff', 'avg', 'maxpool-avg', 'vdiffmap', 'hdiffmap'],
            help="""Optional action to perform on frames at --offset.""")
    parser.add_argument(
            '--offset',
            type=int,
            help='Show difference between activations at diff_offset.',
            default=1)
    parser.add_argument('--frames_per_second', type=int, required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    activations = torchfile.load(args.activations_t7)
    if args.filter == 'max_var':
        activations = extract_max_var_filter(activations)
    elif args.filter == 'max':
        activations = activations.max(axis=1)
    elif args.filter == 'avg':
        activations = activations.mean(axis=1)
    else:
        try:
            filter_index = int(args.filter)
        except ValueError:
            print('--filter must be int, or one of "max_var", "max", "avg"')
            raise
        activations = activations[:, filter_index]

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
        # Replicate frame_activations into 3 channels.
        frame_activations_channels = np.tile(
            frame_activations[:, :, np.newaxis], (1, 1, 3))
        image = Image.fromarray(np.uint8(frame_activations_channels))
        image = image.resize((int(new_width), int(new_height)))
        return np.array(image)

    num_frames = activations.shape[0]
    duration = num_frames / args.frames_per_second

    activations_clip = VideoClip(make_frame, duration=duration)
    video_clip = VideoFileClip(args.video)
    video_clip = video_clip.resize((new_height, new_width))

    final_clip = clips_array([[activations_clip, video_clip]])
    final_clip.write_videofile(args.output)


if __name__ == "__main__":
    main()
