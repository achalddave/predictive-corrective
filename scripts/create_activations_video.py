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
    min_value = activations.min()
    max_value = activations.max()
    activations = (activations - min_value) / (max_value - min_value) * 256

    old_height = activations[1].shape[0]
    old_width = activations[1].shape[1]
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
