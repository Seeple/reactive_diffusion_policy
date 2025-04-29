#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_peel_image_gelsight_emb_dp_absolute_12fps \
    task.dataset_path=/home/wendi/Desktop/record_data/peel_v3_downsample2_zarr \
    task.name=real_peel_image_gelsight_emb_dp_absolute_12fps \
    logging.mode=online