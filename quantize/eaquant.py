import torch
import torch.nn as nn
from models.int_olmoe_layer import QuantOlmoeDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import *
from quantize.const import CLIPMIN
import functools
from collections import defaultdict


device_map_olmoe = {'model.embed_tokens': 0,
              'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 
              'model.layers.6': 0, 'model.layers.7': 0, 
              'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0,
              'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 0, 'model.layers.15': 0, 
              'model.norm':0, "lm_head":0}


from accelerate.big_modeling import dispatch_model
def prepare_for_inference(model, tag="olmoe"):
    if tag=="olmoe":
        dispatch_model(model, device_map=device_map_olmoe)



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, (QuantLinear, nn.Linear)) and ("mlp.gate" in name or "up_proj" in name or "block_sparse_moe.gate" in name or "w3" in name) and "gate_proj" not in name}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs().clamp(min=1e-5) / weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    scale = scale.view(org_shape)
    scale = scale.mean(0).clamp(min=1e-5) # cin
    return scale


def stat_tensor(name, tensor, act_samples):
    if ("mlp.gate" in name or "up_proj" in name or "block_sparse_moe.gate" in name or "w3" in name) and "gate_proj" not in name:
        tensor = tensor.view(-1, tensor.shape[-1])
        num = tensor.shape[0]
        if name in act_samples:
            act_samples[name] += num
        else:
            act_samples[name] = num

# get input features of all linear layers for linear layer back to fp16
def cache_input_hook(m, x, y, name, feat_dict, act_samples, expert_token_num):
    x = x[0]
    x = x.detach().cpu()
    update_tag = True
    if (expert_token_num > 0 and name in act_samples and act_samples[name] >= expert_token_num):
        update_tag = False
    if update_tag:
        feat_dict[name].append(x)
        stat_tensor(name, x, act_samples)


used_device = "cuda"

import logging
def eaquant(
    lm,
    args,
    dataloader,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = used_device

    logger.info(f"dev : {dev}, lm.model.device : {lm.model.device}")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    expert_num = 64
    if "olmoe" in args.net.lower():
        is_llama = True
        is_MOE = True
        args.sp_model_name = "olmoe"
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantOlmoeDecoderLayer
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for olmoe now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    input_ids = []

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                input_ids.append(batch[0])
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "olmoe" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    else:
        raise ValueError("Only support for olmoe now")
    torch.cuda.empty_cache()
    
    quant_inps = inps
    rotate_inps = copy.copy(inps).mean(dim=0).unsqueeze(0)

    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    if args.resume:
        eaquant_parameters = torch.load(os.path.join(args.resume, f"eaquant_parameters.pth"))
    else:
        eaquant_parameters = {}

    for i in range(len(layers)):
        if "olmoe" in args.net.lower():
            dev = f'{used_device}:{device_map_olmoe[f"model.layers.{i}"]}'
        else:
            dev = used_device

        for name in ['q', 'k', 'v', 'gate', 'up', 'down', 'o']:
            exec(f"args.{name}_weight_quant_params = copy.copy(args.weight_quant_params)")
            exec(f"args.{name}_act_quant_params = copy.copy(args.act_quant_params)")
        args.q_quant_params = copy.copy(args.act_quant_params)
        args.k_quant_params = copy.copy(args.act_quant_params)

        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i]
        if "moe" in args.net.lower():
            qlayer = DecoderLayer(lm.model.config, i, layer, args)
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)

        # print(qlayer)
        qlayer = qlayer.to(dev)

        if args.quant_method == 'eaquant':
            set_init_eaquant_params_state(qlayer, True)

        if args.use_fp_input:
            rotate_inps = copy.copy(fp_inps).mean(dim=0).unsqueeze(0)

        fp_inps_back = copy.copy(fp_inps)


        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if True:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev),position_ids=position_ids.to(dev))[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev),position_ids=position_ids.to(dev))[0]
        

        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        
        if args.resume:
            # raise NotImplementedError
            qlayer.load_state_dict(eaquant_parameters[i], strict=False)
            print(eaquant_parameters[i].keys())


        qlayer.half()

        logger.info("eaquant begin")
        input_feat = defaultdict(list)
        # real smooth and quantization      
        if args.quant_method == 'eaquant':
            set_init_eaquant_params_state(qlayer, False)
            set_quant_state(qlayer, weight_quant=True, act_quant=True)
            if eaquant_parameters.get(i):
                qlayer.load_eaquant_params(eaquant_parameters[i], dev)
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        set_registered_x_none(qlayer)
                        
                        if is_MOE: # stage1: calibrate the QKV layers
                            set_init_eaquant_params_state(qlayer.mlp.experts, True) # experts: fp forward, no init_eaquant_params
                            set_quant_state(qlayer.mlp.experts, weight_quant=False, act_quant=False)
                            set_calibration_state(qlayer.mlp.experts, False)

                        ### cache input sample of experts with quantized block
                        named_linears = get_named_linears(qlayer)
                        actual_act_samples = {}
                        handles = []
                        for name in named_linears:
                            handles.append(named_linears[name].register_forward_hook(functools.partial(cache_input_hook, name=name, feat_dict=input_feat, act_samples=actual_act_samples, expert_token_num=args.expert_token_num)))
                        
                            
                        for k in range(rotate_inps.shape[0]): # stage1: calibrate the QKV layers for 1 time
                            qlayer(rotate_inps[k].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev),position_ids=position_ids.to(dev))

                        set_calibration_state(qlayer, False) # stop weight calibration for all layers
                        if args.expert_token_num > 0: # collect more calibration data for experts with quantized QKV layers
                            for k in range(fp_inps_back.shape[0]):
                                qlayer(fp_inps_back[k].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev),position_ids=position_ids.to(dev))

                        for h in handles:
                            h.remove()
                        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
                        
                        if is_MOE:
                            logger.info("========================")
                            for k, v in input_feat.items():
                                logger.info(f"k: {k}, v.shape: {v.shape}, v.dtype: {v.dtype}, v.device: {v.device}") # k: mlp.gate, v.shape: torch.Size([2048, 2048])

                        if is_MOE:
                            set_init_eaquant_params_state(qlayer.mlp.experts, False) # quant forward, do init_eaquant_params
                            set_quant_state(qlayer.mlp.experts, weight_quant=True, act_quant=True)
                            set_calibration_state(qlayer.mlp.experts, True)

                            # stage2: calibrate the activated experts with special tokens for 1 time
                            logger.info(f"calibrate the activated experts with special tokens")
                            for k in range(expert_num):
                                if f"mlp.experts.{k}.up_proj" in input_feat:
                                    logger.info(f"calibration qlayer.mlp.experts[{k}] with input_feat ...")
                                    if args.expert_token_num > 0: # 4096
                                        select_expert_token_num = min(input_feat[f"mlp.experts.{k}.up_proj"].shape[0], args.expert_token_num)
                                    else: # 0
                                        select_expert_token_num = input_feat[f"mlp.experts.{k}.up_proj"].shape[0]
                                    qlayer.mlp.experts[k](input_feat[f"mlp.experts.{k}.up_proj"][:select_expert_token_num].to(dev))

                            # stage3: recalibration the left experts with the whole tokens for 1 time
                            logger.info(f"recalibration the left experts with the whole tokens")
                            for k in range(expert_num):
                                if qlayer.mlp.experts[k].gate_proj.act_quantizer.permutation_list == []: # not calibrated yet
                                    logger.info(f"recalibration qlayer.mlp.experts[{k}] with input_feat ...")
                                    if "olmoe" in args.net.lower():
                                        qlayer.mlp.experts[k](input_feat["mlp.gate"][:args.seq_length].to(dev))

            qlayer.register_eaquant_params()
            set_init_eaquant_params_state(qlayer, True) # quant forward, init_eaquant_params done
        logger.info("eaquant done")
        del input_feat
        
        
        qlayer.half()
        set_calibration_state(qlayer, False)
        quant_inplace(qlayer)
        set_quant_state(qlayer, weight_quant=False, act_quant=True)
    
        layers[i] = qlayer.to("cpu")
        eaquant_parameters[i] = eaquant_state_dict(qlayer)
        if args.save_dir:
            torch.save(eaquant_parameters, os.path.join(args.save_dir, f"eaquant_parameters.pth"))

        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    del fp_inps_back
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache

    logger.info(f"{args.output_dir.split('/')[-1]}")


    return model

