import argparse
import json
from math import ceil, floor

import numpy as np

from scripts.util import read_groundtruth_lmdb

if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--labels_json')
    parser.add_argument('--output_frame_counts')
    parser.add_argument('--output_segment_counts')
    parser.add_argument('--output_average_lengths')
    parser.add_argument('--frames_per_second', default=10, type=int)
    parser.add_argument('--label_mapping',
                        help="""
                        File containing lines of the form "<class_int_id>
                        <class_name>".""")
    args = parser.parse_args()

    label_names = None
    label_ids = {}
    assert(args.label_mapping is not None)
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
    label_ids = {x[1]: x[0] for x in zip(ids, labels)}

    segment_counts = np.zeros(len(label_names))
    frame_counts = np.zeros(len(label_names))
    label_lengths = np.zeros(len(label_names))
    with open(args.labels_json) as f:
        annotations = json.load(f)
    for annotation in annotations:
        segment_counts[label_ids[annotation['category']]] += 1
        # TODO(achald): This doesn't give us the same answer as if we count the
        # frames from the LMDB! Why?
        end_frame = annotation['end_seconds'] * args.frames_per_second
        start_frame = annotation['start_seconds'] * args.frames_per_second
        num_frames = ceil(end_frame) - floor(start_frame)
        frame_counts[label_ids[annotation['category']]] += num_frames
    if args.output_segment_counts is not None:
        np.save(args.output_segment_counts, segment_counts)
    if args.output_frame_counts is not None:
        np.save(args.output_frame_counts, frame_counts)
    if args.output_average_lengths is not None:
        np.save(args.output_average_lengths, frame_counts / segment_counts)
