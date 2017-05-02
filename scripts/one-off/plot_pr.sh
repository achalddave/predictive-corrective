#!/bin/bash

MODEL_A_PREDICTIONS="/data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_4_same_crop_finetuned_single_frame_sequencer/11-03-16-17-45-08/model_30_reinit_4_rerun_valval_predictions.h5"
MODEL_B_PREDICTIONS="/data/achald/MultiTHUMOS/models/vgg_single_frame/background_weight_20/model_30_valval_predictions.h5"

labels=$(cut -d' ' -f2 /data/all/MultiTHUMOS/class_list.txt | paste -sd ',')
python -m scripts.visualizations.plot_prec_rec \
    ${MODEL_A_PREDICTIONS} \
    ${MODEL_B_PREDICTIONS} \
    /data/achald/MultiTHUMOS/frames@10fps/labeled_video_frames/valval_without_images.lmdb \
    $labels \
    --output_dir pr_plots
