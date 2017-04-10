#!/bin/bash

# Train reinit ?
# for i in 1 2 4 8 16 ; do
#     echo $i
#     echo /data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_4_same_crop_finetuned_single_frame_sequencer/11-03-16-17-45-08/model_30_reinit_${i}_rerun_valval_evaluation.log
#     tail -2 /data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_4_same_crop_finetuned_single_frame_sequencer/11-03-16-17-45-08/model_30_reinit_${i}_rerun_valval_evaluation.log | head -1
#     # th scripts/evaluate_model.lua \
#     #     /data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_2_same_crop_finetuned_single_frame_sequencer/11-04-16-17-55-51/model_30.t7 \
#     #     /scratch/achald/valval.lmdb \
#     #     /scratch/achald/valval_without_images.lmdb \
#     #     "/data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_2_same_crop_finetuned_single_frame_sequencer/11-04-16-17-55-51/model_30_reinit_${i}_rerun_valval_evaluation.log"  \
#     #     --batch_size 4 \
#     #     --sequence_length 32 \
#     #     --reinit_rate ${i} \
#     #     --recurrent \
#     #     --output_hdf5 "/data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_2_same_crop_finetuned_single_frame_sequencer/11-04-16-17-55-51/model_30_reinit_${i}_rerun_valval_predictions.h5"
# done

# Train reinit 8
# for i in 2 4 8 16 ; do
    if [[ "$#" -ne 1 ]] ; then exit 1 ; fi
    i=$1
    echo $i
    # echo /data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_8_same_crop_finetuned_single_frame_sequencer/dropout_0.5/no_wd/04-07-17-14-54-26/model_30_reinit_${i}_rerun_valval_evaluation.log
    # tail -2 /data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_8_same_crop_finetuned_single_frame_sequencer/dropout_0.5/no_wd/04-07-17-14-54-26/model_30_reinit_${i}_rerun_valval_evaluation.log | head -1
    th scripts/evaluate_model.lua \
        /data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_2_same_crop_finetuned_single_frame_sequencer/11-04-16-17-55-51/model_30.t7 \
        /scratch/achald/valval.lmdb \
        /scratch/achald/valval_without_images.lmdb \
        "/data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_8_same_crop_finetuned_single_frame_sequencer/dropout_0.5/no_wd/04-07-17-14-54-26/model_30_reinit_${i}_rerun_valval_evaluation.log"  \
        --batch_size 4 \
        --sequence_length 32 \
        --reinit_rate ${i} \
        --recurrent \
        --output_hdf5 "/data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_8_same_crop_finetuned_single_frame_sequencer/dropout_0.5/no_wd/04-07-17-14-54-26/model_30_reinit_${i}_rerun_valval_predictions.h5"
# done
