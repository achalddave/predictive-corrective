# Predictive Corrective Networks for Action Detection

This is the source code for training and evaluating
["Predictive-Corrective
Networks"](http://www.achaldave.com//projects/predictive-corrective/).

Please file an issue if you run into any problems, or [contact
me](mailto:achald@cs.cmu.edu).

## Download models and data

To download models+data, run

    bash download.sh

This will create a directory with the following structure

    data/
        <dataset>/
            models/
                vgg16-init.t7: Initial VGG-16 model pre-trained on Imagenet.
                vgg16-trained.t7: Trained VGG-16 single-frame model.
                pc_c33-1_fc7-8.t7: Trained predictive-corrective model.
            labels/
                trainval.h5: Train labels
                test.h5: Test labels

Currently only models for THUMOS/MultiTHUMOS are included, but we will release
Charades models as soon as possible.

## Dumping frames

Before running on any videos, you will need to dump frames (resized to 256x256)
into a root directory which contains one subdirectory for each video. Each video
subdirectory should contain frames of the form `frame%04d.png` (e.g.
`frame0012.png`), extracted at 10 frames per second. If you would like to train
or evaluate models at different frame rates, please file an issue or contact me
and I can point you in the right direction.

You may find my
[`dump_frames`](https://github.com/achalddave/video-tools/blob/master/dump_frames.py)
and
[`resize_images`](https://github.com/achalddave/video-tools/blob/master/resize_images.py)
scripts useful for this.

## Running a pre-trained model

Store frames from your videos in one directory `frames_root`, with frames at
`frames_root/video_name/frame%04d.png` as described above.

To evaluate the predictive-corrective model, run

```lua
th scripts/evaluate_model.lua \
    --model data/multithumos/models/pc_c33-1_fc7-8.t7 \
    --frames /path/to/frames_root \
    --output_log /path/to/output.log \
    --sequence_length 8 \
    --step_size 1 \
    --batch_size 16 \
    --output_hdf5 /path/to/output_predictions.h5
```

## Training a model

### Single-frame <a name='train-single-frame'></a>

To train a single frame model, look at `config/config-vgg.yaml`. Documentation
for each config parameter is available in `main.lua`, but the only ones you
really need to change are the path to training and test frames.

```yaml
train_source_options:
    frames_root: '/path/to/multithumos/test/frames'
    labels_hdf5: 'data/multithumos/labels/test.h5'

val_source_options:
    frames_root: '/path/to/multithumos/trainval/frames'
    labels_hdf5: 'data/multithumos/labels/trainval.h5'
```

Once you have updated these, run

```bash
th main.lua config/config-vgg.yaml /path/to/output/directory
```

### Predictive Corrective

First, generate a predictive-corrective model initialized from a trained
single-frame model, as follows:

```bash
th scripts/make_predictive_corrective.lua \
--model data/multithumos/models/vgg16-trained.t7 \
--output data/multithumos/models/pc_c33-1_fc7-8-init.t7
```

Next, update `config/config-predictive-corrective.yaml` to point to your dumped
frames, as described [above](#train-single-frame). Then, run

```bash
th main.lua config/config-predictive-corrective.yaml /path/to/output/directory
```

This usually takes 2-3 days to run on 4 GPUs.

## Required packages

Note: This is an incomplete list! TODO(achald): Document all required packages.

- argparse
- classic
- cudnn
- cutorch
- luaposix
- lyaml
- nnlr
- rnn

## Extra: Generate labels hdf5 files

For convenience, we provide the labels for the datasets we use as HDF5 files.
However, it is possible to generate these yourself.
[Here](https://github.com/achalddave/thumos-scripts/blob/master/parse_temporal_annotations_to_hdf5.py)
is the script I used to generate MultiTHUMOS labels HDF5, and
[here](https://github.com/achalddave/charades-scripts/blob/master/parse_temporal_annotations_to_hdf5.py)
is a similar script for Charades. These are not very well documented, but feel
free to contact me if you run into any issues.
