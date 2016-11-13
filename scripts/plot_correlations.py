from __future__ import division

import argparse
import collections
import logging
import random

import lmdb
import matplotlib.pyplot as plt
import numpy as np

from video_util import video_frames_pb2


def read_images(groundtruth_lmdb, video_name):
    """
    Returns:
        images (np.array, shape (num_frames, width, height, channels))
    """
    lmdb_environment = lmdb.open(groundtruth_lmdb)
    images = []
    with lmdb_environment.begin().cursor() as read_cursor:
        frame = 1
        while True:
            video_frame = video_frames_pb2.LabeledVideoFrame()
            frame_data = read_cursor.get('%s-%d' % (video_name, frame))
            if frame_data is None:
                break
            video_frame.ParseFromString(frame_data)
            image_proto = video_frame.frame.image
            image = np.fromstring(image_proto.data,
                                  dtype=np.uint8).reshape(
                                      image_proto.channels, image_proto.height,
                                      image_proto.width).transpose((1, 2, 0))
            images.append(image)
            frame += 1
    return np.asarray(images)

if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('groundtruth_lmdb')
    parser.add_argument('video_name')
    parser.add_argument('--frame', type=int)

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                        datefmt='%H:%M:%S')

    # This is in bgr format.
    images = read_images(args.groundtruth_lmdb, args.video_name).astype(float)
    # For now, let's just use one channel.
    images = images[:, :, :, 0]
    # Downsample

    num_frames = images.shape[0]

    if args.frame is not None:
        frame_indices = [args.frame]
    else:
        frame_indices = list(range(1, num_frames - 1))
        random.shuffle(frame_indices)
    for frame in frame_indices:
        frame1_values = images[frame].reshape(-1)
        frame2_values = images[frame + 1].reshape(-1)
        diff_values = frame2_values - frame1_values

        plt.clf()
        plt.suptitle("Frame %d and %d (Red Channel)" % (frame, frame + 1))
        ax1 = plt.subplot(2, 2, 1)
        ax1.scatter(frame1_values, frame2_values, alpha=0.01)
        ax1.set_xlim(0, 255)
        ax1.set_ylim(0, 255)
        ax1.set_xlabel('$x_1$', fontsize=20)
        ax1.set_ylabel('$x_2$', fontsize=20)

        # Attempt at setting colors based on (x, y) occurrences counts.
        # scatter_values = [(frame1_values[i], frame2_values[i])
        #                   for i in range(frame1_values.shape[0])]
        # counter = collections.Counter(scatter_values)
        # scatter_x, scatter_y = zip(*counter.keys())
        # max_val = max(counter.values())
        # print(counter.values()[:10])
        # print(max_val)
        # print([x / max_val for x in counter.values()][:10])
        # ax1.scatter(scatter_x,
        #             scatter_y,
        #             s=2,
        #             vmin=-50,
        #             vmax=max_val,
        #             c=counter.values(),
        #             linewidths=0,
        #             cmap='Blues')

        ax2 = plt.subplot(2, 2, 3)
        ax2.scatter(frame1_values, diff_values, alpha=0.01)
        ax2.set_xlim(0, 255)
        ax2.set_ylim(-255, 255)
        ax2.set_xlabel('$x_1$', fontsize=20)
        ax2.set_ylabel('$x_2 - x_1$', fontsize=20)

        ax3 = plt.subplot(2, 2, 2)
        ax3.imshow(images[frame], cmap='gray', interpolation='none')
        ax3.set_title('Frame 1 ($x_1$)', fontsize=15)
        ax3.axis('off')

        ax4 = plt.subplot(2, 2, 4)
        ax4.imshow(images[frame + 1], cmap='gray', interpolation='none')
        ax4.set_title('Frame 2 ($x_2$)', fontsize=15)
        ax4.axis('off')

        plt.waitforbuttonpress()
