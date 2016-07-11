import argparse

import torchfile
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('torch_file')
    parser.add_argument('output_file', nargs='?')
    parser.add_argument('--hide',
                        help="Save the plot but don't show it.",
                        action='store_true')

    args = parser.parse_args()

    weights = torchfile.load(args.torch_file)
    plt.set_cmap('gray')
    plt.imshow(weights, interpolation='none')
    plt.colorbar()
    if not args.hide:
        plt.show()
    if args.output_file:
        plt.savefig(args.output_file, transparent=True)


if __name__ == "__main__":
    main()
