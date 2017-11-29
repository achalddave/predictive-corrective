"""Compute mAP with predictions in HDF5 file and a groundtruth LMDB."""
from __future__ import division
import argparse

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from scripts.util import compute_average_precision, read_groundtruth_lmdb


def main(predictions_hdf5,
         groundtruth_without_images_lmdb,
         selected_frames=None):
    groundtruth_by_video, label_names = read_groundtruth_lmdb(
        groundtruth_without_images_lmdb)

    with h5py.File(predictions_hdf5, 'r') as f:
        predictions_by_video = {key: np.array(f[key]) for key in f.keys()}

    predictions = {}
    groundtruth = {}
    for video_name in groundtruth_by_video.keys():
        if selected_frames is not None:
            predictions_by_video[video_name] = predictions_by_video[
                video_name][selected_frames[video_name], :]
            groundtruth_by_video[video_name] = groundtruth_by_video[
                video_name][selected_frames[video_name], :]
        for label in range(len(label_names)):
            if label not in predictions:
                predictions[label] = predictions_by_video[video_name][:, label]
                groundtruth[label] = groundtruth_by_video[video_name][:, label]
            else:
                predictions[label] = np.hstack((predictions[
                    label], predictions_by_video[video_name][:, label]))
                groundtruth[label] = np.hstack((groundtruth[
                    label], groundtruth_by_video[video_name][:, label]))

    aps = np.zeros(len(label_names))
    accuracies = np.zeros(len(label_names))
    for label in range(len(label_names)):
        # Old code used to set the first few frames' prediction to be exactly
        # -1. Set these values to be a much smaller negative value instead
        # (since we are dealing with the predictions before the sigmoid).
        #
        # We can safely assume that we are only changing the values that were
        # artificially set to -1, since the output of the network for any frame
        # will basically never be exactly -1.
        predictions[label][predictions[label] == -1] = -1e9
        accuracies[label] = (
            groundtruth[label] ==
            (predictions[label] > 0)).sum() / groundtruth[label].shape[0]
        aps[label] = compute_average_precision(groundtruth[label],
                                               predictions[label])
    print('Average accuracy: %s' % accuracies.mean())
    return aps


if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('predictions_hdf5')
    parser.add_argument('groundtruth_without_images_lmdb')
    parser.add_argument('--selected_frames')
    parser.add_argument('--output_aps')

    args = parser.parse_args()

    selected_frames = None
    if args.selected_frames is not None:
        import collections
        import csv
        with open(args.selected_frames, 'rb') as f:
            reader = csv.reader(f)
            selected_frames = collections.defaultdict(list)
            for video, frame in reader:
                selected_frames[video].append(int(frame))
            print(selected_frames)

    aps = main(args.predictions_hdf5, args.groundtruth_without_images_lmdb,
               selected_frames)
    if args.output_aps is not None:
        np.save(args.output_aps, aps)
    print 'mAP:', aps.mean()
