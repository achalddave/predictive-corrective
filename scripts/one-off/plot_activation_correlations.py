import argparse
import pickle

from matplotlib import pyplot as plt
import numpy as np


if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--single_frame_activations', required=True)
    parser.add_argument('--correction_activations', required=True)

    args = parser.parse_args()

    with open(args.single_frame_activations, 'rb') as f:
        single_frame = pickle.load(f)

    with open(args.correction_activations, 'rb') as f:
        corrections = pickle.load(f)

    print('Loaded')

    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(single_frame['frame1'],
                single_frame['frame2'],
                s=70,
                alpha=0.7,
                rasterized=True,
                marker='.')
    # ax1.set_xlim(0, 255)
    # ax1.set_ylim(0, 255)
    ax1.set_xlabel('$z_t$', fontsize=16)
    ax1.set_ylabel('$z_{t+1}$', fontsize=16)
    ax1.set_title('VGG-16 activations')
    ax1.tick_params(axis='both',
                    which='both',
                    bottom='off',
                    top='off',
                    left='off',
                    right='off',
                    labelbottom='off',
                    labelleft='off')
    x0, x1 = ax1.get_xlim()
    y0, y1 = ax1.get_ylim()
    ax1.set_aspect((x1 - x0) / (y1 - y0))

    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(corrections['frame1'],
                corrections['frame2'],
                s=70,
                alpha=0.7,
                rasterized=True,
                marker='.')
    # ax2.set_xlim(0, 255)
    # ax2.set_ylim(0, 255)
    ax2.set_xlabel('$z_t$', fontsize=16)
    ax2.set_ylabel('$z_{t+1}$', fontsize=16)
    ax2.set_title('Our corrections')
    ax2.tick_params(axis='both',
                    which='both',
                    bottom='off',
                    top='off',
                    left='off',
                    right='off',
                    labelbottom='off',
                    labelleft='off')
    x0, x1 = ax2.get_xlim()
    y0, y1 = ax2.get_ylim()
    ax2.set_aspect((x1 - x0) / (y1 - y0))

    plt.suptitle('conv4-3 activations', y=0.93)
    plt.subplots_adjust(top=0.5, wspace=0.9)

    plt.tight_layout()

    plt.savefig('tmp.pdf', bbox_inches='tight', dpi=600)
    plt.show()

    """
    indices = np.random.choice(mat['first_frame'].shape[1], 500)
    frame1_values = mat['first_frame'][:, indices]
    frame2_values = mat['second_frame'][:, indices]
    diff_values = mat['diff'][:, indices]

    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(frame1_values, frame2_values, s=70, alpha=0.7, rasterized=True, marker='.')
    ax1.set_xlim(0, 255)
    ax1.set_ylim(0, 255)
    ax1.set_xlabel('frame$_1$', fontsize=16)
    ax1.set_ylabel('frame$_2$', fontsize=16)
    ax1.set_aspect(0.7);

    ax2 = plt.subplot(1, 2, 2)
    ax2.scatter(frame1_values, diff_values, s=70, alpha=0.7, rasterized=True, marker='.')
    ax2.set_xlim(0, 255)
    ax2.set_ylim(-255, 255)
    ax2.set_xlabel('frame$_1$', fontsize=16)
    ax2.set_ylabel('frame$_2$ - frame$_1$', fontsize=16)
    ax2.set_aspect(0.35);

    plt.tight_layout()
    # plt.subplots_adjust(top=0.99,wspace=0.9)

    plt.savefig(args.output,bbox_inches='tight')

    plt.show()
    """
