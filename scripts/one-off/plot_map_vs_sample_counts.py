"""Plot mAP and # of samples in batches for each label.

Plot a bar chart containing two bars for each label: the mAP, and a count of
the number of samples containing that label during training."""

import argparse

import numpy as np
import torchfile
from matplotlib import pyplot as plt
from mpldatacursor import datacursor
from scipy.stats import pearsonr
# plt.style.use('ggplot')

def plot_bars(mean_aps, label_counts, label_names=None):
    """
    Args:
        mean_aps (vector, shape (num_labels))
        label_counts (vector, shape (num_labels + 1))
    """
    label_counts /= label_counts.sum()
    index_spacer = 3

    num_labels = mean_aps.shape[0]
    print(num_labels)
    indices = np.arange(num_labels + 1) * index_spacer

    old_mean_aps = mean_aps
    mean_aps = np.zeros(num_labels + 1)
    mean_aps[:num_labels] = old_mean_aps

    bar_width = 1

    print(pearsonr(mean_aps[:num_labels], label_counts[:num_labels]))
    sorted_labels = np.array(sorted(
        range(num_labels + 1), key=lambda i: mean_aps[i]))

    fig, ax = plt.subplots()
    ap_bar = ax.bar(indices,
                    mean_aps[sorted_labels],
                    bar_width,
                    color='r',
                    label='AP')

    ax2 = ax.twinx()
    counts_bar = ax2.bar(indices + bar_width,
                         label_counts[sorted_labels],
                         bar_width,
                         color='b',
                         label='Counts')
    ax.set_xticks(np.arange(1, index_spacer*(num_labels + 1), index_spacer))
    ax.set_xticklabels(sorted_labels + 1)
    ax.legend((ap_bar[0], counts_bar[0]), ('AP', 'Counts'))
    if label_names is not None:
        label_names = label_names + ['Background']
        def formatter(**kwargs):
            return 'x: {}, label: {}, name: {}'.format(
                kwargs['x'], sorted_labels[int(kwargs['x'] / index_spacer)],
                label_names[sorted_labels[int(kwargs['x'] / index_spacer)]])
        datacursor(formatter=formatter)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('label_aps', help="""
        `.npy` file containing a (num_labels) vector of AP per label.""")
    parser.add_argument('label_counts', help="""
        `.t7` file containing a (num_labels+1) Tensor containing counts of
        samples for each label (plus background), as output by
        scripts/batch_label_distribution.lua""")
    parser.add_argument('--label_mapping',
                        help="""
                        File containing lines of the form "<class_int_id>
                        <class_name>".""")

    args = parser.parse_args()
    label_names = None
    if args.label_mapping is not None:
        with open(args.label_mapping) as f:
            ids = []
            labels = []
            for line in f:
                ids.append(int(line.strip().split(' ')[0]))
                labels.append(line.strip().split(' ')[1])
        zero_indexed = 0 in ids
        if not zero_indexed:
            ids = [original_id - 1 for original_id in ids]
        label_names = [x[1] for x in sorted(zip(ids, labels))]
        print(label_names)

    mean_aps = np.load(args.label_aps)[:20]
    label_counts = torchfile.load(args.label_counts)
    label_counts = np.hstack((label_counts[:20], label_counts[-1]))
    plot_bars(mean_aps, label_counts, label_names)
