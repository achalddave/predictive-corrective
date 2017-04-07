from __future__ import division, print_function
import argparse
import re

if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('evaluation_log')
    parser.add_argument('keyword', default='Reinitialized', nargs='?')

    args = parser.parse_args()
    num_matched = 0
    keyword = args.keyword
    keyword_regex = re.compile(
        '.*{} ([0-9]*) out of ([0-9]*) times.'.format(keyword))
    num_frames_regex = re.compile('.*Finished [0-9]*/([0-9]*)')
    num_frames = None
    with open(args.evaluation_log) as f:
        for line in f:
            if num_frames is None:
                num_frames_match = num_frames_regex.match(line)
                if num_frames_match:
                    num_frames = int(num_frames_match.group(1))
            match = keyword_regex.match(line)
            if match:
                num_matched += int(match.group(1))
    print('{percent:.2f}% ({matched}/{total}) {keyword}'.format(
        keyword=keyword.lower(),
        matched=num_matched,
        total=num_frames,
        percent=100.0 * num_matched / num_frames))
