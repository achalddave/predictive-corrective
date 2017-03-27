"""Compute mAP with predictions in HDF5 file and a groundtruth LMDB."""
from __future__ import division
import argparse
from collections import defaultdict

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from scripts.util import compute_average_precision, read_groundtruth_lmdb


def main(predictions_hdf5, groundtruth_without_images_lmdb):
    groundtruth_by_video, label_names = read_groundtruth_lmdb(
        groundtruth_without_images_lmdb)

    with h5py.File(predictions_hdf5, 'r') as f:
        predictions_by_video = {key: np.array(f[key]) for key in f.keys()}

    predictions = {}
    groundtruth = {}
    for video_name in groundtruth_by_video.keys():
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
    for label in range(len(label_names)):
        aps[label] = compute_average_precision(groundtruth[label], predictions[label])
    return aps


if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('predictions_hdf5')
    parser.add_argument('groundtruth_without_images_lmdb')
    parser.add_argument('--output_aps')

    args = parser.parse_args()
    aps = main(args.predictions_hdf5, args.groundtruth_without_images_lmdb)
    if args.output_aps is not None:
        np.save(args.output_aps, aps)
    print 'mAP:', aps.mean()
