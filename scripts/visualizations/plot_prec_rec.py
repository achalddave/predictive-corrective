from __future__ import division
import argparse

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from scripts.util import compute_average_precision, read_groundtruth_lmdb

LABELS_LIST = "/data/all/MultiTHUMOS/class_list.txt"


def plot_prec_rec(left_predictions_hdf5, right_predictions_hdf5,
                  groundtruth_by_video, label_name, left_name,
                  right_name, output_dir):
    label_map = {}  # Map label name to id
    with open(LABELS_LIST) as f:
        for line in f:
            label_map[line.split(' ')[1].strip()] = int(line.split(' ')[0]) - 1
    label = label_map[label_name]

    with h5py.File(left_predictions_hdf5, 'r') as f:
        left_predictions_by_video = {key: np.array(f[key]) for key in f.keys()}
    with h5py.File(right_predictions_hdf5, 'r') as f:
        right_predictions_by_video = {
            key: np.array(f[key])
            for key in f.keys()
        }

    plt.style.use('ggplot')

    def helper(predictions_by_video, name):
        predictions = None
        groundtruth = None
        for video_name in groundtruth_by_video.keys():
            if predictions is None:
                predictions = predictions_by_video[video_name][:, label]
                groundtruth = groundtruth_by_video[video_name][:, label]
            else:
                predictions = np.hstack(
                    (predictions, predictions_by_video[video_name][:, label]))
                groundtruth = np.hstack(
                    (groundtruth, groundtruth_by_video[video_name][:, label]))

        precision, recall, _ = precision_recall_curve(groundtruth, predictions)
        ap = compute_average_precision(groundtruth, predictions)
        print('Average precision for %s, %s-%s:' %
              (name, label + 1, label_name), ap)
        for j in range(1, len(precision)):
            precision[j] = max((precision[j], precision[j-1]))
        plt.plot(recall, precision, label='%s' % label_name, linewidth=3)
        plt.xlabel('Recall', fontsize=30)
        plt.ylabel('Precision', fontsize=30)

    matplotlib.rcParams.update({'font.size': 25})
    plt.clf()
    plt.title(label_name, fontsize=30)
    helper(left_predictions_by_video, left_name)
    helper(right_predictions_by_video, right_name)
    plt.tight_layout()
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.gca().set_axis_bgcolor('white')
    # plt.gca().grid(color='0.8')
    if output_dir:
        plt.savefig(
            '%s/%s.pdf' % (output_dir, label_name), bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('left_predictions_hdf5')
    parser.add_argument('right_predictions_hdf5')
    parser.add_argument('groundtruth_without_images_lmdb')
    parser.add_argument('labels', help='comma separated list of labels')
    parser.add_argument('--left_name', default='Left')
    parser.add_argument('--right_name', default='Right')
    parser.add_argument('--output_dir', default=None)

    args = parser.parse_args()
    groundtruth_by_video = read_groundtruth_lmdb(
        args.groundtruth_without_images_lmdb)[0]
    for label in args.labels.split(','):
        print(label)
        plot_prec_rec(
            args.left_predictions_hdf5,
            args.right_predictions_hdf5,
            groundtruth_by_video,
            label,
            left_name=args.left_name,
            right_name=args.right_name,
            output_dir=args.output_dir)
