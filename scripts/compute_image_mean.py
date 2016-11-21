"""Compute mean for each image channel from a dataset."""

from __future__ import division

import argparse
import random

import lmdb
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from video_util import video_frames_pb2

map_size = 200e9

def compute_image_mean(path):
    image_sum = None
    with lmdb.open(path, map_size=map_size) as env, \
            env.begin().cursor() as lmdb_cursor:
        num_entries = env.stat()['entries']
        progress = tqdm(total=num_entries)
        i = 0
        while lmdb_cursor.next():
            video_frame = video_frames_pb2.LabeledVideoFrame()
            video_frame.ParseFromString(lmdb_cursor.value())
            image_proto = video_frame.frame.image
            # Shape (width, height, num_channels)
            image = np.fromstring(image_proto.data,
                                  dtype=np.uint8).reshape(
                                      image_proto.channels, image_proto.height,
                                      image_proto.width).transpose((1, 2, 0))
            if image_sum is None:
                image_sum = image.astype(float)
            else:
                image_sum += image.astype(float)
            progress.update()
            if i % 100000 == 0:
                print((image_sum / i).mean(axis=(0, 1)))
            i += 1
    return (image_sum / num_entries).mean(axis=(0, 1))


if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('lmdb')

    args = parser.parse_args()

    print('Image mean:')
    print(compute_image_mean(args.lmdb))
