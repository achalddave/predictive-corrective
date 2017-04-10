"""Average predictions from two predictions HDF5 files."""

from __future__ import division
import argparse

import h5py

from scripts.util import compute_average_precision, read_groundtruth_lmdb

def merge_hdf5(input_paths, output_path):
    with h5py.File(output_path, 'w') as output_hdf5:
        for input_path in input_paths:
            with h5py.File(input_path) as input_hdf5:
                for key, value in input_hdf5.items():
                    if key not in output_hdf5:
                        output_hdf5[key] = value.value / len(input_paths)
                    else:
                        output_hdf5[key][...] += (value.value /
                                                  len(input_paths))

if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('predictions_hdf5', nargs='+')
    parser.add_argument('output_predictions_hdf5')

    args = parser.parse_args()
    merge_hdf5(args.predictions_hdf5, args.output_predictions_hdf5)
