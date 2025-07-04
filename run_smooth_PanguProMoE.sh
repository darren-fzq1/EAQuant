#!/bin/bash

python main.py \
--model PanguProMoE \
--save_dir PanguProMoE_smooth \
--smooth \
--only_smooth_fp \
--fc1_scale_merge max \
--alpha 0.6 \
--otsu_ratio 0.65 \
--otsu_smooth_rate 0.7 \
--seq_length 2048


