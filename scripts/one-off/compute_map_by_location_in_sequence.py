# Compute mean average precision based on the index of a frame in a window.
#
# For example, if the sequence length is 4, compute 4 mean average precisions.
# - mAP for frames 0, 4,  8, ...
# - mAP for frames 1, 5,  9, ...
# - mAP for frames 2, 6, 10, ...
# - mAP for frames 3, 7, 11, ...
#
# This was a one-off script to create one of the figures in my supplementary
# material for CVPR 2017. I tried this with a model that "reinitialized" every
# 4th frame, so I could see how the mean average precision changed based on the
# number of "past frames" that were in the memory.

from __future__ import division
import argparse
from collections import defaultdict

import h5py
import lmdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from video_util import video_frames_pb2


def compute_average_precision(groundtruth, predictions):
    """
    Computes average precision for a binary problem.

    See:
    <https://en.m.wikipedia.org/wiki/Information_retrieval#Average_precision>

    This is what sklearn.metrics.average_precision_score should do, but it is
    broken:
    https://github.com/scikit-learn/scikit-learn/issues/5379
    https://github.com/scikit-learn/scikit-learn/issues/6377

    Args:
        groundtruth (array-like): Binary vector indicating whether each sample
            is positive or negative.
        predictions (array-like): Contains scores for each sample.

    Returns:
        Average precision.

    >>> predictions = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    >>> groundtruth = [1, 0, 1, 1, 1, 1, 0, 0, 0, 1]
    >>> expected_ap = (1. + 2/3. + 3/4. + 4/5. + 5/6. + 6/10.) / 6
    >>> ap = compute_average_precision(groundtruth, predictions)
    >>> assert ap == expected_ap, ('Expected %s, received %s'
    ...                            % (ap, expected_ap))

    >>> predictions = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    >>> groundtruth = [1, 0, 0, 0, 1, 1, 0, 0, 0, 1]
    >>> expected_ap = (1. + 2/5. + 3/6. + 4/10.) / 4
    >>> ap = compute_average_precision(groundtruth, predictions)
    >>> assert ap == expected_ap, ('Expected %s, received %s'
    ...                            % (ap, expected_ap))
    """
    sorted_indices = sorted(
        range(len(predictions)),
        key=lambda x: predictions[x],
        reverse=True)

    average_precision = 0
    true_positives = 0
    if sum(predictions) == 0:
        print 'No predictions!'
    for num_guesses, index in enumerate(sorted_indices):
        if groundtruth[index]:
            true_positives += 1
            precision = true_positives / (num_guesses + 1)
            average_precision += precision
    average_precision /= sum(groundtruth)
    return average_precision


def parse_frame_key(frame_key):
    """
    >>> parse_frame_key("video_validation_0000-123")
    ('video_validation_0000', 123)
    """
    frame_number = int(frame_key.split('-')[-1])
    video_name = '-'.join(frame_key.split('-')[:-1])
    return (video_name, frame_number)


# TODO: Share this with predictive_corrective_changes/main.py and
# prediction_diffs/main.py.
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


def main(predictions_hdf5, groundtruth_without_images_lmdb, sequence_length):
    groundtruth_by_video, label_names = read_groundtruth_lmdb(
        groundtruth_without_images_lmdb)

    with h5py.File(predictions_hdf5, 'r') as f:
        predictions_by_video = {key: np.array(f[key]) for key in f.keys()}

    predictions = [{} for _ in range(sequence_length)]
    groundtruth = [{} for _ in range(sequence_length)]
    mean_ap = 0
    for video_name in groundtruth_by_video.keys():
        num_frames = predictions_by_video[video_name].shape[0]
        frame_selectors = [np.arange(i, num_frames, sequence_length) for i in range(sequence_length)]
        for label in range(len(label_names)):
            for i in range(sequence_length):
                curr_predictions = predictions_by_video[video_name][frame_selectors[i], label]
                curr_groundtruth = groundtruth_by_video[video_name][frame_selectors[i], label]
                if label not in predictions[i]:
                    predictions[i][label] = curr_predictions
                    groundtruth[i][label] = curr_groundtruth
                else:
                    predictions[i][label] = np.hstack((predictions[i][
                        label], curr_predictions))
                    groundtruth[i][label] = np.hstack((groundtruth[i][
                        label], curr_groundtruth))

    aps = [0. for _ in range(sequence_length)]
    for label in range(len(label_names)):
        for i in range(sequence_length):
            aps[i] += compute_average_precision(groundtruth[i][label],
                                                predictions[i][label])
    for i in range(sequence_length):
        aps[i] /= len(label_names)
    return aps


if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('predictions_hdf5')
    parser.add_argument('groundtruth_without_images_lmdb')
    parser.add_argument('--sequence_length', default=4, type=int)

    args = parser.parse_args()
    print(main(args.predictions_hdf5, args.groundtruth_without_images_lmdb, args.sequence_length))
