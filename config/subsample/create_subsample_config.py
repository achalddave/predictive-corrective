import argparse
import re

if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--subsample_rate', required=True, type=int)
    parser.add_argument('--config_output', required=True)
    parser.add_argument('--config_template', default='config-subsample-template.yaml')
    # Default is number of images in train+val set.
    parser.add_argument('--num_images', default=556241, type=int)

    args = parser.parse_args()

    with open(args.config_template, 'r') as f:
        template = f.read()

    # We can't use yaml to parse the template, since the template itself is not
    # valid yaml. Instead, hackily parse the config file to get the batch size.
    batch_size = int(re.search(r'batch_size: *([0-9]*) *\n', template).group(1))
    subsample = args.subsample_rate

    def update_template(var, value):
        return template.replace(var, str(value))

    template = update_template('{{SUBSAMPLE_RATE}}', subsample)
    template = update_template('{{NUM_EPOCHS}}', 2 * subsample)
    template = update_template(
        '{{EPOCH_SIZE}}',
        int(round(args.num_images / batch_size / subsample / 2)))
    template = update_template('{{EVALUATE_EVERY}}', subsample)

    # Reduce learning rate by 10 every 2.5 epochs.
    lr_step = subsample * 5
    for i in range(5):
        template = update_template('{{LR_EPOCH%d}}' % (i + 1), lr_step * i + 1)
    with open(args.config_output, 'w') as f:
        f.write(template)
