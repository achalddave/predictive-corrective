from __future__ import division
import argparse

if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('evaluation_log')

    args = parser.parse_args()
    num_init = 0
    num_frames = 0
    with open(args.evaluation_log) as f:
        for line in f:
            if not line.startswith('Reinitialized'): continue
            num_init += int(line.split(' ')[1])
            num_frames += int(line.split(' ')[4])
    print(num_init, num_frames, num_init / num_frames)
