from __future__ import division

import argparse
import logging
import random
import subprocess
import sys
from collections import defaultdict, namedtuple
from os import path

import h5py
import jinja2
import lmdb
import matplotlib.pyplot as plt
import numpy as np

from video_util import video_frames_pb2

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                    datefmt='%H:%M:%S')

FramePrediction = namedtuple('FramePrediction',
                             ['video_name', 'frame_number', 'groundtruth',
                              'left_predictions', 'right_predictions'])


def parse_frame_key(frame_key):
    """
    >>> parse_frame_key("video_validation_0000-123")
    ('video_validation_0000', 123)
    """
    frame_number = int(frame_key.split('-')[-1])
    video_name = '-'.join(frame_key.split('-')[:-1])
    return (video_name, frame_number)


def read_groundtruth_lmdb(groundtruth_without_images_lmdb):
    """Read groundtruth for frames from LMDB of LabeledVideoFrame.

    Returns:
        groundtruth (dict): Maps video name to numpy array of shape
            (num_video_frames, num_labels), which is one-hot encoding of labels
            per frame.
        label_names (list): List of label names
    """
    lmdb_environment = lmdb.open(groundtruth_without_images_lmdb)
    # Map video name to groundtruth per frame.
    groundtruth = defaultdict(dict)
    video_num_frames = defaultdict(lambda: 0)
    label_names = {}
    with lmdb_environment.begin().cursor() as read_cursor:
        for frame_key, frame_data in read_cursor:
            video_name, frame_number = parse_frame_key(frame_key)
            frame_number -= 1  # Frame numbers are 1-indexed.
            video_frame = video_frames_pb2.LabeledVideoFrame()
            video_frame.ParseFromString(frame_data)
            labels = []
            for label in video_frame.label:
                labels.append(label.id)
                label_names[label.id] = label.name
            groundtruth[video_name][frame_number] = labels
    label_names = [x[1] for x in sorted(label_names.items())]

    # Map videos to np.array of shape (num_frames, num_labels)
    groundtruth_onehot = {}
    for video, frame_predictions in groundtruth.items():
        num_frames = len(frame_predictions)
        groundtruth_onehot[video] = np.zeros((num_frames, len(label_names)))
        for frame_number, label_ids in frame_predictions.items():
            for label_id in label_ids:
                groundtruth_onehot[video][frame_number][label_ids] = 1
    return groundtruth_onehot, label_names


def compute_frames_of_interest(left_predictions_hdf5, right_predictions_hdf5,
                               groundtruth_without_images_lmdb, diff_method):
    with h5py.File(left_predictions_hdf5, 'r') as f:
        left_predictions = {key: np.array(f[key]) for key in f.keys()}
    with h5py.File(right_predictions_hdf5, 'r') as f:
        right_predictions = {key: np.array(f[key]) for key in f.keys()}

    # Maps video name to (num_frames, num_labels) array.
    groundtruth, label_names = read_groundtruth_lmdb(
        groundtruth_without_images_lmdb)

    assert set(groundtruth.keys()) == set(left_predictions.keys())
    # List of (video_name, frame_index) tuples.
    frames_of_interest = []
    for video_name, video_labels in groundtruth.items():
        if (left_predictions[video_name].shape != video_labels.shape) or (
                right_predictions[video_name].shape != video_labels.shape):
            print('%s: (Left: %s, Right: %s, GT: %s)' %
                  (video_name, left_predictions[video_name].shape,
                   right_predictions[video_name].shape, video_labels.shape))

        left_guesses = left_predictions[video_name] > 0.5
        right_guesses = right_predictions[video_name] > 0.5
        num_frames = left_guesses.shape[0]

        left_wrong_frames = set(np.nonzero(left_guesses != video_labels)[0])
        right_wrong_frames = set(np.nonzero(right_guesses != video_labels)[0])
        all_frames = set(range(num_frames))
        if diff_method == 'both_wrong':
            selected_frames = left_wrong_frames | right_wrong_frames
        elif diff_method == 'both_right':
            selected_frames = all_frames - (left_wrong_frames |
                                            right_wrong_frames)
        elif diff_method == 'left_better':
            selected_frames = right_wrong_frames - left_wrong_frames
        elif diff_method == 'right_better':
            selected_frames = left_wrong_frames - right_wrong_frames
        frames_of_interest.extend([(video_name, frame)
                                   for frame in selected_frames])
    # List of FramePrediction objects
    frame_prediction_of_interest = []
    for video_name, frame in frames_of_interest:
        groundtruth_labels = [
            label_names[i] for i in groundtruth[video_name][frame].nonzero()[0]
        ]
        left_labels = [
            label_names[i]
            for i in np.where(left_predictions[video_name][frame] > 0.5)[0]
        ]
        right_labels = [
            label_names[i]
            for i in np.where(right_predictions[video_name][frame] > 0.5)[0]
        ]
        frame_prediction_of_interest.append(
            FramePrediction(video_name=video_name,
                            frame_number=frame,
                            groundtruth=groundtruth_labels,
                            left_predictions=left_labels,
                            right_predictions=right_labels))
    return frame_prediction_of_interest


def main(left_predictions_hdf5, right_predictions_hdf5,
         groundtruth_without_images_lmdb, frames_dir, num_output, diff_method,
         left_name, right_name, output, command_string):
    frames_of_interest = compute_frames_of_interest(
        left_predictions_hdf5, right_predictions_hdf5,
        groundtruth_without_images_lmdb, diff_method)

    jinja_env = jinja2.Environment(
        loader=jinja2.PackageLoader(__name__, 'templates'))
    root_template = jinja_env.get_template('index.html')

    diff_template = jinja_env.get_template('prediction_diff.html')
    diff_templates_combined = ''
    for frame_prediction in random.sample(frames_of_interest, num_output):
        video_name = frame_prediction.video_name
        frame_number = frame_prediction.frame_number
        image = path.join(frames_dir, video_name, 'frame%04d.png' %
                          frame_number)
        diff_templates_combined += diff_template.render({
            'image': image,
            'left_name': left_name,
            'right_name': right_name,
            'left_predictions': ', '.join(frame_prediction.left_predictions),
            'right_predictions': ', '.join(frame_prediction.right_predictions),
            'groundtruth': ', '.join(frame_prediction.groundtruth)
        })
    with open(output, 'wb') as f:
        f.write(root_template.render(frame_diffs=diff_templates_combined,
                                     command_string=command_string))


if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--left_predictions',
                        required=True,
                        help='HDF5 containing predictions.')
    parser.add_argument('--right_predictions',
                        required=True,
                        help='HDF5 containing predictions.')
    parser.add_argument(
        '--groundtruth_without_images_lmdb',
        required=True,
        help='LMDB containing LabeledVideoFrames without video data.')
    parser.add_argument('--frames_dir',
                        required=True,
                        help='Directory containing video frames.')
    parser.add_argument(
        '--diff_method',
        choices=['both_wrong', 'both_right', 'left_better', 'right_better'],
        required=True)
    parser.add_argument('--output_html', required=True)

    # Optional arguments
    parser.add_argument('--left_name', default='Left')
    parser.add_argument('--right_name', default='Right')
    parser.add_argument('--num_output', type=int, default=100)

    args = parser.parse_args()

    # Courtesy <http://stackoverflow.com/a/12411695/1291812>
    command = '%s %s' % (sys.argv[0], subprocess.list2cmdline(sys.argv[1:]))
    main(left_predictions_hdf5=args.left_predictions,
         left_name=args.left_name,
         right_predictions_hdf5=args.right_predictions,
         right_name=args.right_name,
         groundtruth_without_images_lmdb=args.groundtruth_without_images_lmdb,
         frames_dir=args.frames_dir,
         diff_method=args.diff_method,
         num_output=args.num_output,
         output=args.output_html,
         command_string=command)
