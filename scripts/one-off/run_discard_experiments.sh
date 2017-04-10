#!/bin/bash

# for i in 0.01 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 ; do
#     echo "==="
#     echo $i
#     echo /data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_4_same_crop_finetuned_single_frame_sequencer/11-03-16-17-45-08/model_30_pcb_ignore_thresh_${i}_max_update_4_seq_52_valval_evaluation.log
#     python scripts/one-off/count_reinit.py /data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_4_same_crop_finetuned_single_frame_sequencer/11-03-16-17-45-08/model_30_pcb_ignore_thresh_${i}_max_update_4_seq_52_valval_evaluation.log Ignored
#     tail -2 /data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_4_same_crop_finetuned_single_frame_sequencer/11-03-16-17-45-08/model_30_pcb_ignore_thresh_${i}_max_update_4_seq_52_valval_evaluation.log | head -1
#     echo "==="
# done

for i in 0.01 0.03 0.075 ; do # 0.175 0.2 0.3 ; do
    echo $i
    th scripts/evaluate_model.lua \
        /data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_4_same_crop_finetuned_single_frame_sequencer/11-03-16-17-45-08/model_30_predictive_corrective_block.t7 \
        /scratch/achald/valval.lmdb \
        /scratch/achald/valval_without_images.lmdb \
        "/data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_4_same_crop_finetuned_single_frame_sequencer/11-03-16-17-45-08/model_30_pcb_ignore_thresh_${i}_max_update_8_seq_52_valval_evaluation.log"  \
        --batch_size 1 \
        --sequence_length 52 \
        --recurrent \
        --ignore_threshold " ${i}" \
        --min_reinit_rate 8 \
        --reinit_threshold 100000 \
        --output_hdf5 "/data/achald/MultiTHUMOS/models/vgg_hierarchical/with_recursive_script/sum_fc7_reinit_4_same_crop_finetuned_single_frame_sequencer/11-03-16-17-45-08/model_30_pcb_ignore_thresh_${i}_max_update_8_seq_52_valval_predictions.h5"
done
