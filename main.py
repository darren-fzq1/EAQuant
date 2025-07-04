import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.eaquant import eaquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories
from utils import *

from quantize.smooth import smooth_lm
import logging
torch.backends.cudnn.benchmark = True


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument('--test_dataset', type=str, default='wikitext2', help='dataset for testing')
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--router_wbits", type=int, default=8)
    parser.add_argument("--router_abits", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--otsu_ratio", type=float, default=0.65)
    parser.add_argument("--otsu_smooth_rate", type=float, default=0.7)
    parser.add_argument("--act_group_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--smooth",default=False, action="store_true")
    parser.add_argument("--only_smooth_fp",default=False, action="store_true")
    parser.add_argument("--let",default=False, action="store_true",help="activate learnable equivalent transformation")
    parser.add_argument("--lwc",default=False, action="store_true",help="activate learnable weight clipping")
    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token")
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None)

    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--fc1_scale_merge", type=str, default="max", help="expert-aware smoothing aggregation strategy")
    parser.add_argument("--router_w_dynamic_method", type=str, default="per_channel_kl_top0", help="the dynamic_method of the gate layer's weight")
    parser.add_argument("--expert_token_num_ratio", type=float, default=2.0, help="the ratio of expect expert_token_num over average expert_token_num")

    # EaQuant
    parser.add_argument("--max_rotation_step", type=int, default=256, help="max steps for rotation transformation")
    parser.add_argument("--permutation_times", type=int, default=1, help="times of permutation transformation")
    parser.add_argument("--lac", type=float, default=None, help="activation clipping ratio")
    parser.add_argument("--swc", type=float, default=None, help="weight clipping ratio, enable withou lwc")
    parser.add_argument("--block_size", type=int, default=128, help="block size for rotation matrices")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    args.use_fp_input = 1
    args.epochs = 0
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    args.quant_method = "eaquant"
    args.expert_ratio = 8/64.0   # for olmoe, select_expert_num / expert_num
    args.expert_token_num = int(args.expert_ratio * args.seq_length * args.expert_token_num_ratio)

    # init logger
    args.output_dir = os.path.join(args.output_dir, f"{args.model.split('/')[-1]}_w{args.wbits}a{args.abits}_Rw{args.router_wbits}a{args.router_abits}")   
    print(f"args.output_dir: {args.output_dir}")
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(f"args.output_dir: {args.output_dir}")
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    args.model_family = args.net.split('-')[0]
    lm = LMClass(args)
    print(f"lm.device: {args.output_dir}")
    lm.seqlen = args.seq_length
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False
    logger.info(args)

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "swc":args.swc,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "lac":args.lac,
        "act_group_size": args.act_group_size,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    
    args.router_weight_quant_params = {
        "n_bits": args.router_wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.router_w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "swc":args.swc,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.router_act_quant_params = {
        "n_bits":  args.router_abits,
        "per_channel_axes": [],
        "symmetric": False,
        "lac":args.lac,
        "act_group_size": args.act_group_size,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # quantization
    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        import time
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}_{args.seq_length}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)

        if args.smooth:
            convert_device(lm, args)
            # selected_experts = get_router_selected_experts(lm.model, dataloader, lm.model.config.num_experts_per_tok, args.nsamples, args.net)
            act_samples = get_act_samples(lm.model, dataloader, args.nsamples)
            weight_scores = get_weight_scores(lm.model)
            router_logits = get_router_logits(lm.model, dataloader, args.nsamples)
            act_scales = get_act_scales(lm.model, dataloader, args.nsamples)
            act_per_channel_scales = get_act_per_channel_scales(lm.model, dataloader, args.nsamples)
            smooth_lm(lm.model, act_scales, act_per_channel_scales, act_samples, weight_scores, router_logits, 
                    fc1_scale_merge=args.fc1_scale_merge, alpha=args.alpha, otsu_ratio=args.otsu_ratio, otsu_smooth_rate=args.otsu_smooth_rate, logger=logger)
            lm.model.cpu()

            if args.only_smooth_fp:
                logger.info(f"save smoothed model at save_dir: {args.save_dir}")
                lm.model.save_pretrained(args.save_dir)
                lm.tokenizer.save_pretrained(args.save_dir)
                exit()
            
        eaquant(
            lm,
            args,
            dataloader,
            logger=logger,
        )
        # logger.info(time.time() - tick)
    logger.info(f"args.output_dir: {args.output_dir}")
    logger.info(f"args.tasks: {args.tasks}")
    evaluate(lm, args,logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
