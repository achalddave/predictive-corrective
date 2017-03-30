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
from scripts.util import (compute_average_precision, parse_frame_key,
                          read_groundtruth_lmdb)


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
