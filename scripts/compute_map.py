"""Compute mAP with predictions in HDF5 file and a groundtruth LMDB."""
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
    Computes average precision for a binary problem. This is based off of the
    PASCAL VOC implementation.
    Args:
        groundtruth (array-like): Binary vector indicating whether each sample
            is positive or negative.
        predictions (array-like): Contains scores for each sample.
    Returns:
        Average precision.
    """
    predictions = np.asarray(predictions)
    groundtruth = np.asarray(groundtruth, dtype=float)

    sorted_indices = np.argsort(predictions)[::-1]
    predictions = predictions[sorted_indices]
    groundtruth = groundtruth[sorted_indices]
    # The false positives are all the negative groundtruth instances, since we
    # assume all instances were 'retrieved'. Ideally, these will be low scoring
    # and therefore in the end of the vector.
    false_positives = 1 - groundtruth

    tp = np.cumsum(groundtruth)      # tp[i] = # of positive examples up to i
    fp = np.cumsum(false_positives)  # fp[i] = # of false positives up to i

    num_positives = tp[-1]

    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    recalls = tp / num_positives

    # Append end points of the precision recall curve.
    precisions = np.concatenate(([0.], precisions, [0.]))
    recalls = np.concatenate(([0.], recalls, [1.]))

    # Set precisions[i] = max(precisions[j] for j >= i)
    # This is because (for j > i), recall[j] >= recall[i], so we can always use
    # a lower threshold to get the higher recall and higher precision at j.
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    # Find points where recall value changes.
    c = np.where(recalls[1:] != recalls[:-1])[0]

    ap = np.sum((recalls[c + 1] - recalls[c]) * precisions[c + 1])

    return ap


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


def main(predictions_hdf5, groundtruth_without_images_lmdb):
    groundtruth_by_video, label_names = read_groundtruth_lmdb(
        groundtruth_without_images_lmdb)

    with h5py.File(predictions_hdf5, 'r') as f:
        predictions_by_video = {key: np.array(f[key]) for key in f.keys()}

    predictions = {}
    groundtruth = {}
    mean_ap = 0
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

    mean_ap = 0.0
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
