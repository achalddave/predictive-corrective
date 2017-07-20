"""Sample activations from consecutive pairs of frames.

This is used to analyze the correlation of activations across time."""

from __future__ import division

import argparse
import logging
import pickle
import subprocess
import sys

import numpy as np


def _set_logging(logging_filepath):
    log_formatter = logging.Formatter('%(asctime)s.%(msecs).03d: %(message)s',
                                      datefmt='%H:%M:%S')

    file_handler = logging.FileHandler(logging_filepath)
    file_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(file_handler)
    logging.info('Writing log file to %s', logging_filepath)


def _log_info(args):
    logging.info('Command line arguments: %s', sys.argv)
    logging.info('Parsed arguments: %s', args)

    def run(command):
        logging.info('Running command: %s', ' '.join(command))
        logging.info(subprocess.check_output(command))
    run([
        'git', '--no-pager', 'diff',
        'scripts/sample_activation_correlations.py'
    ])
    run(['git', '--no-pager', 'rev-parse', 'HEAD'])


def sample(args):
    import torchfile

    activations_t7 = args.activations_t7
    sampled_activations = args.sampled_activations
    num_samples = args.num_samples

    _set_logging(sampled_activations + '.log')
    _log_info(args)

    # (num_frames, num_filters, height, width) array.
    all_activations = torchfile.load(activations_t7)
    num_frames, num_filters, height, width = all_activations.shape
    sampled_frames = np.random.randint(num_frames-1, size=num_samples)
    sampled_filters = np.random.randint(num_filters, size=num_samples)
    sampled_rows = np.random.randint(height, size=num_samples)
    sampled_cols = np.random.randint(width, size=num_samples)
    sampled_points = zip(sampled_frames, sampled_filters, sampled_rows,
                         sampled_cols)
    frame1_values = np.zeros(num_samples)
    frame2_values = np.zeros(num_samples)
    for i, (frame, filter_index, row, col) in enumerate(sampled_points):
        frame1_values[i] = all_activations[frame, filter_index, row, col]
        frame2_values[i] = all_activations[frame+1, filter_index, row, col]

    with open(args.sampled_activations, 'wb') as f:
        pickle.dump({'frame1': frame1_values, 'frame2': frame2_values}, f)


def plot(args):
    from matplotlib import pyplot as plt

    sampled_activations = args.sampled_activations
    output_plot = args.output_plot

    _set_logging(output_plot + '.log')
    _log_info(args)

    with open(sampled_activations, 'r') as f:
        data = pickle.load(f)

    frame1_values = data['frame1']
    frame2_values = data['frame2']
    plt.clf()

    plt.scatter(frame1_values, frame2_values, alpha=0.1, rasterized=True)
    plt.tight_layout()
    plt.savefig(output_plot, bbox_inches='tight')
    plt.show()


def main():
    # Use first line of file docstring as description.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(dest='command')

    # create the parser for the "a" command
    sample_parser = subparsers.add_parser('sample', help='Sample activations')
    sample_parser.add_argument('--activations_t7', required=True)
    sample_parser.add_argument('--num_samples', required=True, type=int)
    sample_parser.add_argument('--sampled_activations', required=True)

    sample_parser = subparsers.add_parser('plot', help='Plot correlations')
    sample_parser.add_argument('--sampled_activations', required=True)

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                        datefmt='%H:%M:%S')
    print(args)

    # args.func(args) basically does this, but I want to be explicit.
    if args.command == 'sample':
        sample(args)
    elif args.command == 'plot':
        plot(args)


if __name__ == "__main__":
    main()
