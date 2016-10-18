Examples of commands I've used. These are not commands to replicate
experiments, but just a way to easily see how to use some of the scripts.

# Training a network

````bash
th main.lua \
    config/config-vgg-pyramid.yaml \
    /data/achald/MultiTHUMOS/models/vgg_hierarchical/residual_xavier_init_untie_last_sum_conv43_conv53_flipped_img_diffs/ \
| tee /data/achald/MultiTHUMOS/models/vgg_hierarchical/residual_xavier_init_untie_last_sum_conv43_conv53_flipped_img_diffs/training.log
````

# Evaluating a network

````bash
th scripts/evaluate_model.lua \
       /data/achald/MultiTHUMOS/models/balanced_without_bg_sampling_vgg_new/from_scratch/model_30.t7 \
       /data/achald/MultiTHUMOS/frames@10fps/labeled_video_frames/valval.lmdb/ \
       /data/achald/MultiTHUMOS/frames@10fps/labeled_video_frames/valval_without_images.lmdb \
       --sequence_length 1 \
       --step_size 1 \
       --batch_size 128 \
       --val_groups  /data/achald/MultiTHUMOS/val_split/val_val_groups.txt \
       --output_hdf5 /data/achald/MultiTHUMOS/models/balanced_without_bg_sampling_vgg_new/from_scratch/model_30_valval_predictions.h5 \
       | tee /data/achald/MultiTHUMOS/models/balanced_without_bg_sampling_vgg_new/from_scratch/model_30_valval_evaluation.log
````

# Computing activations for a video

````bash
th scripts/compute_video_activations.lua \
       --model /data/achald/MultiTHUMOS/models/balanced_without_bg_sampling_vgg_new/model_30.t7 \
       --layer_spec 'cudnn.SpatialConvolution,2' \
       --frames_lmdb /data/achald/MultiTHUMOS/frames@10fps/labeled_video_frames/valval.lmdb \
       --video_name video_validation_0000901 \
       --output_activations
       /data/achald/MultiTHUMOS/visualizations/balanced_sampling_vgg/model_30_video_validation_0000901_vis_conv1_2.t7
````

# Create activations video
````bash
python create_activations_video.py \
    --activations_t7 "/data/achald/MultiTHUMOS/visualizations/permuted_sampling_vgg_conv/model_30_${video}_vis_${layer}.t7"  \
    --video "/data/achald/THUMOS/2014/temporal_detection_videos/val/${video}.mp4" \
    --filter ${filter} \
    --frames_per_second 10 \
    --output "/data/achald/MultiTHUMOS/visualizations/permuted_sampling_vgg_conv/model_30_${video}_vis_${layer}_filter_${filter}.mp4"
````
