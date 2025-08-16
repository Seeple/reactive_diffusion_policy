#!/bin/bash

# DP w. GelSight Emb. (Peeling)
#python eval_real_robot_flexiv.py \
#      --config-name train_diffusion_unet_real_image_workspace \
#      task=real_peel_image_gelsight_emb_absolute_12fps \
#      +task.env_runner.output_dir=/path/for/saving/videos \
#      +ckpt_path=/path/to/dp/checkpoint

# RDP w. Force (Peeling)
python eval_real_robot_flexiv.py \
      --config-name train_latent_diffusion_unet_real_image_workspace \
      task=real_peel_image_wrench_ldp_24fps \
      at=at_peel \
      at_load_dir=/path/to/at/checkpoint \
      +task.env_runner.output_dir=/path/for/saving/videos \
      +ckpt_path=/path/to/ldp/checkpoint