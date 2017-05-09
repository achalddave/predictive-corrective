# Plot loss/accuracy/test error from training log file.
#
# Alternatives
# ============
# I spent a few hours trying to come up with a better way to do this that
# doesn't involve parsing the plain text training.log file. Unfortunately,
# there's no good framework for logging and plotting (similar to TensorBoard)
# in Torch. Here's a brief summary of the options:
#
# - optim.Logger: Logs and plots but
#     * must pre-specify a list of names that # will be logged,
#       all of which must be present at each log call, and
#     * does not have a way of plotting from a log file.
#
# - torrvision/crayon: Interface to tensorboard for lua.
#     * Requires that a server be running whenever attempting to log.
#
# - TeamHG-Memex/tensorboard_logger: Interface to tensorboard for python
#     * No Lua interface. I considered porting part of this to lua, but I don't
#       know how stable tensorboard's summary objects are between updates to
#       Tensorflow.
#
# - Manually write a JSON/CSV file: This doesn't seem particularly better than
#   parsing the text file. Ultimately, I will probably change the format of the
#   JSON/CSV file over time, and it involves adding more code to the Trainer
#   class. I'll consider doing this later, but it will probably take some
#   planning to get right.

import argparse
import re

import numpy as np
from matplotlib import pyplot as plt

int_pattern = '[0-9]*'
float_pattern = '[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

train_pattern = (
        r'.*Epoch: \[(?P<epoch>{int})\]\s*'
        r'\[(?P<iter>{int})/(?P<total_iter>{int})\]\s*'
        r'Time (?P<time>{flt})\s*'
        r'Loss (?P<loss>{flt})\s*'
        r'(?:epoch mAP (?P<epoch_map>{flt})\s*)?'
        r'LR (?P<lr>{flt})').format(int=int_pattern, flt=float_pattern)

summary_pattern = (
        r'.*Epoch: \[(?P<epoch>{int})\]'
        r'\[(?P<mode>TRAINING|EVALUATION) SUMMARY\]\s*'
        r'Total Time\(s\): (?P<time>{flt})\s*'
        r'average loss \(per batch\): (?P<loss>{flt})\s*'
        r'mAP: (?P<mAP>{flt})\s*').format(int=int_pattern, flt=float_pattern)

train_re = re.compile(train_pattern)
summary_re = re.compile(summary_pattern)

def match_training(line):
    """
    >>> example = ('[INFO  11:04:30]: Epoch: [1] [52/593] 	 '
    ...            'Time 1.569 Loss 0.1064 LR 5e-04	')
    >>> sorted(match_training(example).items())
    [('epoch', 1), ('iter', 52), ('loss', 0.1064), ('lr', 0.0005), \
('time', 1.569), ('total_iter', 593)]
    """
    match = train_re.match(line)
    if match:
        match = match.groupdict()
        match['epoch'] = int(match['epoch'])
        match['iter'] = int(match['iter'])
        match['loss'] = float(match['loss'])
        match['lr'] = float(match['lr'])
        match['time'] = float(match['time'])
        match['total_iter'] = int(match['total_iter'])

        if match['epoch_map'] == None:
            del match['epoch_map']
        else:
            match['epoch_map'] = float(match['epoch_map'])
    return match


def match_summary(line):
    """
    >>> example = ('[INFO  12:16:53]: Epoch: [1][TRAINING SUMMARY] '
    ...            'Total Time(s): 4901.42	average loss (per batch): 0.23667 '
    ...            'mAP: 0.18515	')
    >>> sorted(match_summary(example).items())
    [('epoch', '1'), ('loss', '0.23667'), ('mAP', '0.18515'), \
('time', '4901.42'), ('train', True)]
    >>> example = ('[INFO  21:52:45]: Epoch: [5][EVALUATION SUMMARY] '
    ...            'Total Time(s): 15425.03	average loss (per batch): 2.85945 '
    ...            'mAP: 0.18608	')
    >>> sorted(match_summary(example).items())
    [('epoch', '5'), ('loss', '2.85945'), ('mAP', '0.18608'), \
('time', '15425.03'), ('train', False)]
    """
    match = summary_re.match(line)
    if match:
        match = match.groupdict()
        match['train'] = match['mode'] == 'TRAINING'
        del match['mode']
    return match

def moving_average(a, n=3):
    """From http://stackoverflow.com/a/14314054/1291812 ."""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('training_log')

    args = parser.parse_args()

    training = []
    train_summary = []
    eval_summary = []

    with open(args.training_log) as f:
        for line in f:
            training_line = match_training(line)
            if training_line:
                training.append(training_line)
                continue
            summary_line = match_summary(line)
            if summary_line:
                summary = summary_line
                training_mode = summary['train']
                del summary['train']
                if training_mode:
                    train_summary.append(summary)
                else:
                    eval_summary.append(summary)
    y = [x['loss'] for x in training]

    plt.plot(y)
    plt.show()
