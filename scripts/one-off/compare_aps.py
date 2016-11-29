import argparse


def get_aps(log_file):
    with open(log_file) as f:
        lines = [line.strip() for line in f.readlines()]
    # AP lines are of the form "Class <number>\t AP: <float>"
    return [float(line.split('AP: ')[-1])
            for line in lines if line.startswith('Class')]

if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('evaluation_log_1')
    parser.add_argument('evaluation_log_2')
    args = parser.parse_args()

    aps1 = get_aps(args.evaluation_log_1)
    aps2 = get_aps(args.evaluation_log_2)
    print(aps1)
    print(aps2)
    diffs = [ap2 - ap1 for (ap1, ap2) in zip(aps1, aps2)]
    sorted_diffs = sorted(enumerate(diffs), key=lambda x: x[1])
    for i, diff in sorted_diffs:
        print('Class %s change: %s' % (i + 1, diff))
