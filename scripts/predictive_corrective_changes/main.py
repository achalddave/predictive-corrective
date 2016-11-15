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

FramePrediction = namedtuple('FramePrediction', ['video_name', 'frames',
                                                 'groundtruth', 'predictions'])


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


def compute_frames_of_interest(predictions_hdf5,
                               groundtruth_without_images_lmdb,
                               sequence_length=4):
    """
    Find a set of `sequence_length` consecutive frames where
        - we predict action A at first, then stop predicting action A
        - ideally, our predictions are correct in each frame
    """
    with h5py.File(predictions_hdf5, 'r') as f:
        predictions = {key: np.array(f[key]) for key in f.keys()}

    # Maps video name to (num_frames, num_labels) array.
    groundtruth, label_names = read_groundtruth_lmdb(
        groundtruth_without_images_lmdb)

    assert set(groundtruth.keys()) == set(predictions.keys())

    # List of (video_name, start_frame) tuples.
    frames_of_interest = []
    for video_name, video_labels in groundtruth.items():
        # (num_frames, num_labels)
        guesses = (predictions[video_name] > 0)
        num_frames = guesses.shape[0]

        for frame in range(0, num_frames, sequence_length):
            end_frame = min(frame + sequence_length, num_frames)
            # If we don't match with groundtruth, skip this.
            matching_labels = np.where(np.all(
                video_labels[frame:end_frame] == guesses[frame:end_frame],
                axis=0))[0]
            if not len(matching_labels):
                continue

            # We are only interested in frames where predictions change.
            switching_labels = set()
            for other_frame in range(frame + 1, end_frame):
                assert np.all(guesses[other_frame, matching_labels] ==
                              video_labels[other_frame, matching_labels])
                # Indices into matching_labels
                curr_switching_labels = np.where(guesses[other_frame,
                    matching_labels] != guesses[frame, matching_labels])[0]
                # Actual label indices.
                curr_switching_labels = [matching_labels[i]
                                         for i in curr_switching_labels]
                if len(curr_switching_labels):
                    switching_labels.update(curr_switching_labels)
            if len(switching_labels):
                for label in switching_labels:
                    # Add the first frame in the sequence to frames of
                    # interest.
                    assert(video_labels[frame, label] == guesses[frame, label])
                    frames_of_interest.append((video_name, frame, label))

    # List of FramePrediction objects
    frame_prediction_of_interest = []
    for video_name, start_frame, label in frames_of_interest:
        num_frames = groundtruth[video_name].shape[0]
        groundtruth_labels = []
        predicted_labels = []
        for frame in range(start_frame, min(start_frame + sequence_length,
                                            num_frames)):
            if groundtruth[video_name][frame, label]:
                assert(predictions[video_name][frame, label] > 0)
                groundtruth_labels.append(label_names[label])
                predicted_labels.append(label_names[label])
            else:
                groundtruth_labels.append('{}')
                predicted_labels.append('{}')

        frame_prediction_of_interest.append(FramePrediction(
            video_name=video_name,
            frames=list(range(start_frame, start_frame + sequence_length)),
            groundtruth=groundtruth_labels,
            predictions=predicted_labels))
    return frame_prediction_of_interest


def main(predictions_hdf5, groundtruth_without_images_lmdb, frames_dir,
         num_output, name, output, command_string):
    frames_of_interest = compute_frames_of_interest(
        predictions_hdf5, groundtruth_without_images_lmdb)
    for interest in frames_of_interest: print(interest)

    jinja_env = jinja2.Environment(
        loader=jinja2.PackageLoader(__name__, 'templates'))
    root_template = jinja_env.get_template('index.html')

    diff_template = jinja_env.get_template('prediction_changes.html')
    diff_templates_combined = ''
    print(len(frames_of_interest))
    for frame_prediction in random.sample(frames_of_interest, num_output):
        video_name = frame_prediction.video_name
        images = [
            path.join(frames_dir, video_name, 'frame%04d.png' % (x + 1))
            for x in frame_prediction.frames
        ]
        diff_templates_combined += diff_template.render({
            'image0': images[0],
            'image1': images[1],
            'image2': images[2],
            'image3': images[3],
            'name': name,
            'predictions': ' | '.join(frame_prediction.predictions),
            'groundtruth': ' | '.join(frame_prediction.groundtruth),
            'frame_id': '%s-%s' % (video_name, frame_prediction.frames[-1] + 1)
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
    parser.add_argument('--predictions',
                        required=True,
                        help='HDF5 containing predictions.')
    parser.add_argument(
        '--groundtruth_without_images_lmdb',
        required=True,
        help='LMDB containing LabeledVideoFrames without video data.')
    parser.add_argument('--frames_dir',
                        required=True,
                        help='Directory containing video frames.')
    parser.add_argument('--output_html', required=True)

    # Optional arguments
    parser.add_argument('--name', default='Predictive-Corrective')
    parser.add_argument('--num_output', type=int, default=100)
    parser.add_argument('--seed', default=0)

    args = parser.parse_args()

    print 'Using seed: ', args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Courtesy <http://stackoverflow.com/a/12411695/1291812>
    command = '%s %s' % (sys.argv[0], subprocess.list2cmdline(sys.argv[1:]))
    main(predictions_hdf5=args.predictions,
         groundtruth_without_images_lmdb=args.groundtruth_without_images_lmdb,
         frames_dir=args.frames_dir,
         num_output=args.num_output,
         name=args.name,
         output=args.output_html,
         command_string=command)
