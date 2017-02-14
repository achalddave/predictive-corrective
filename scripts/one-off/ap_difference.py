import argparse

import numpy as np
from matplotlib import pyplot as plt
from mpldatacursor import datacursor


def plot_difference_bars(baseline,
                         experiment,
                         label_names=None,
                         baseline_name='Baseline',
                         experiment_name='Experiment'):
    print('Plotting')
    experiment *= 100
    baseline *= 100
    difference = (experiment - baseline) / baseline

    num_labels = experiment.shape[0]
    indices = np.arange(num_labels)

    bar_width = 0.3
    sorted_labels = np.array(sorted(
        range(num_labels), key=lambda i: difference[i]))

    fig, ax = plt.subplots()
    baseline_bar = ax.bar(indices,
                          baseline[sorted_labels],
                          bar_width,
                          label=baseline_name, color='b')
    experiment_bar = ax.bar(indices + bar_width,
                            experiment[sorted_labels],
                            bar_width,
                            label=experiment_name, color='r')
    # difference_bar = ax.bar(indices,
    #                         difference,
    #                         bar_width,
    #                         label='Experiment - baseline')
    ax.set_xticks(indices + bar_width)
    ax.set_xticklabels(sorted_labels + 1)
    ax.legend()

    if label_names is not None:
        def formatter(**kwargs):
            label = sorted_labels[int(kwargs['x'])]
            return 'label: {}, baseline: {:.2f}, experiment: {:.2f}'.format(
                label_names[label], baseline[label], experiment[label])
        datacursor(formatter=formatter, bbox={'alpha': 1})
    print 'Most improved:'
    print '=============='
    for label in sorted_labels[-5:]:
        print '{}:\t {:.2f} \t=>\t {:.2f}'.format(label_names[label],
                                            baseline[label], experiment[label])
    print 'Least improved:'
    print '==============='
    for label in sorted_labels[:5]:
        print '{}:\t {:.2f} \t=>\t {:.2f}'.format(label_names[label],
                                            baseline[label], experiment[label])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('baseline', help="""
        `.npy` file containing a (num_labels) vector of AP per label.""")
    parser.add_argument('experiment', help="""
        `.npy` file containing a (num_labels) vector of AP per label.""")
    parser.add_argument('--baseline_name',
                        default='baseline',
                        help="""User friendly name for baseline.""")
    parser.add_argument('--experiment_name',
                        default='experiment',
                        help="""User friendly name for experiment.""")
    parser.add_argument('--label_mapping',
                        help="""
                        File containing lines of the form "<class_int_id>
                        <class_name>".""")


    args = parser.parse_args()
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

    baseline_aps = np.load(args.baseline)
    experiment_aps = np.load(args.experiment)

    plot_difference_bars(baseline_aps, experiment_aps, label_names,
                         args.baseline_name, args.experiment_name)
