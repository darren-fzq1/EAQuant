#!/bin/bash

python main.py \
--model OLMoE-1B-7B-0924 \
--wbits 4 \
--abits 4 \
--router_wbits 8 \
--router_abits 8 \
--smooth \
--fc1_scale_merge max \
--w_dynamic_method per_channel_tensor \
--router_w_dynamic_method per_channel_kl_top0 \
--expert_token_num_ratio 2.0 \
--epochs 0 --lwc --alpha 0.6 --lac 0.9 --swc 0.8 \
--seq_length 4096  \
--eval_ppl \
--task arc_easy,arc_challenge,hellaswag,winogrande,boolq,piqa


