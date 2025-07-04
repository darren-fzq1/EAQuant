import torch
import torch.nn as nn

from models.olmoe.modeling_olmoe import OlmoeDecoderLayer
from models.pangu_moe.modeling_pangu_moe import PanguProMoEDecoderLayer

from datasets import load_dataset
import functools
from tqdm import tqdm
import numpy as np
import logging
import torch.nn.functional as F

from quantize.outlier_finder import otsu

@torch.no_grad()
def calcu_outlier_mask(per_channel_max, per_channel_min, dev, ratio=0.65, smooth_rate=0.7, alpha=1.5):
    total_channel = per_channel_max.size()[0]
    per_channel_max = per_channel_max.to(dev)
    per_channel_min = per_channel_min.to(dev)

    outlier_mask = torch.ones(total_channel).to(dev)
    outlier_mask_pos = torch.ones(total_channel).to(dev)
    outlier_mask_neg = torch.ones(total_channel).to(dev)

    per_channel_min = per_channel_min.to(torch.float32)
    q1_neg = torch.quantile(per_channel_min, 1-ratio)
    q3_neg = torch.quantile(per_channel_min, ratio)
    IQR_neg = q3_neg - q1_neg
    outlier_neg = q1_neg - alpha*IQR_neg

    per_channel_max = per_channel_max.to(torch.float32)
    q1_pos = torch.quantile(per_channel_max, 1-ratio)
    q3_pos = torch.quantile(per_channel_max, ratio)
    IQR_pos = q3_pos -q1_pos
    outlier_pos = q3_pos+ alpha*IQR_pos

    if per_channel_max[per_channel_max > outlier_pos].numel()>1:
        logging.info("pos first outlier={} with outlier_pos={}".format(per_channel_max[per_channel_max > outlier_pos].numel(), outlier_pos))
        _,pos_outliers,outlier_pos=otsu(per_channel_max[per_channel_max > outlier_pos])
        logging.info("pos later outlier={} with outlier_pos={}".format(per_channel_max[per_channel_max > outlier_pos].numel(), outlier_pos))

    if per_channel_min[per_channel_min < outlier_neg].numel()>1:
        logging.info("neg first outlier={} with outlier_neg={}".format(per_channel_min[per_channel_min < outlier_neg].numel(), outlier_neg))
        _,neg_outliers,outlier_neg = otsu(per_channel_min[per_channel_min < outlier_neg],pos=False)
        logging.info("neg later outlier={} with outlier_neg={}".format(per_channel_min[per_channel_min < outlier_neg].numel(), outlier_neg))

    outlier_mask_pos[per_channel_max > outlier_pos] = 0 # mask outlier value
    outlier_mask_neg[per_channel_min < outlier_neg] = 0 # mask outlier value

    max_value = torch.max(per_channel_max[outlier_mask_pos == 1]) # get normal max value
    min_value = torch.min(per_channel_min[outlier_mask_neg == 1]) # get normal min value

    outlier_mask_pos[outlier_mask_pos == 0] = (per_channel_max.to(torch.float32)[outlier_mask_pos == 0]/max_value)**smooth_rate
    outlier_mask_neg[outlier_mask_neg == 0] = (per_channel_min.to(torch.float32)[outlier_mask_neg == 0]/min_value)**smooth_rate

    if min_value != 0:
        outlier_mask = torch.max(outlier_mask_pos,outlier_mask_neg)
    else:
        outlier_mask = outlier_mask_pos
    outlier_mask[outlier_mask==0] = 1
    logging.info("outlier channel:{}".format(torch.sum(outlier_mask!=1)))

    outlier_mask_pos = outlier_mask_pos.to("cpu")
    outlier_mask_neg = outlier_mask_neg.to("cpu")
    per_channel_max = per_channel_max.to("cpu")
    per_channel_min = per_channel_min.to("cpu")
    
    del outlier_mask_pos
    del outlier_mask_neg
    del per_channel_max
    del per_channel_min
    return outlier_mask

def merge_scale(sm_scales_list, act_samples_list, weight_scores_list=None, router_logits=None, expert_num_list=[64,8,0], fc1_scale_merge="max"):
    expert_num, select_expert_num, share_expert_num = expert_num_list
    act_samples_total = act_samples_list[0] # gate

    ### 1. act_samples
    del act_samples_list[0] # gate
    if share_expert_num > 0: 
        del act_samples_list[-1] # shared_expert, shared_experts
    for j in range(expert_num): # experts
        act_samples_list[j] = torch.tensor([act_samples_list[j]/(act_samples_total * select_expert_num)])
    act_samples_ratio = torch.stack(act_samples_list, dim=0)

    ### 2. weight_scores
    del weight_scores_list[0] # gate
    if share_expert_num > 0:
        del weight_scores_list[-1] # shared_expert, shared_experts
    weight_scores_list = [torch.tensor([x]) for x in weight_scores_list]
    if "sum2_0" in fc1_scale_merge:
        weight_scores_list = torch.stack(weight_scores_list, dim=0).sqrt()
        weight_scores_ratio = F.softmax(weight_scores_list, dim=0, dtype=torch.float)
    elif "sum2_1" in fc1_scale_merge:
        weight_scores_list = torch.stack(weight_scores_list, dim=0)
        weight_scores_ratio = F.softmax(weight_scores_list, dim=0, dtype=torch.float)
    elif "sum2_2" in fc1_scale_merge:
        weight_scores_list = torch.stack(weight_scores_list, dim=0).sqrt()
        weight_scores_ratio = weight_scores_list / weight_scores_list.sum()
    elif "sum2_3" in fc1_scale_merge:
        weight_scores_list = torch.stack(weight_scores_list, dim=0)
        weight_scores_ratio = weight_scores_list / weight_scores_list.sum()
    else:
        weight_scores_ratio = torch.stack(weight_scores_list, dim=0) * 0.0

    sm_scales_fc1 = torch.stack(sm_scales_list, dim=0)
    if "_gate" in fc1_scale_merge:
        sm_scales_gate = sm_scales_list[0]
    else:
        sm_scales_gate = sm_scales_list[0] * 0.0
    del sm_scales_list[0] # gate
    if share_expert_num > 0:
        sm_scales_share_expert = sm_scales_list[-1]
        del sm_scales_list[-1] # shared_expert, shared_experts
        sm_scales_gate = torch.max(sm_scales_gate, sm_scales_share_expert)
    sm_scales_expert = torch.stack(sm_scales_list, dim=0)
    fc1_smooth_ratio1 = act_samples_ratio.view(-1,1).to(sm_scales_expert.device).to(sm_scales_expert.dtype)    # expert frequency
    fc1_smooth_ratio2 = weight_scores_ratio.view(-1,1).to(sm_scales_expert.device).to(sm_scales_expert.dtype) # weight distribution
    fc1_smooth_ratio3 = router_logits.view(-1,1).to(sm_scales_expert.device).to(sm_scales_expert.dtype)       # router distribution

    if "max" in fc1_scale_merge:
        merge_fc1_smooth_scale = torch.max(torch.max(sm_scales_expert, dim=0).values, sm_scales_gate)
    elif "sum1" in fc1_scale_merge:
        merge_fc1_smooth_scale = torch.max(torch.sum(sm_scales_expert*fc1_smooth_ratio1, dim=0), sm_scales_gate)
    elif "sum2" in fc1_scale_merge:
        merge_fc1_smooth_scale = torch.max(torch.sum(sm_scales_expert*fc1_smooth_ratio2, dim=0), sm_scales_gate)
    elif "sum3" in fc1_scale_merge:
        merge_fc1_smooth_scale = torch.max(torch.sum(sm_scales_expert*fc1_smooth_ratio3, dim=0), sm_scales_gate)

    elif "all" in fc1_scale_merge:
        merge_fc1_smooth_scale0 = torch.max(torch.max(sm_scales_expert, dim=0).values, sm_scales_gate)
        merge_fc1_smooth_scale1 = torch.max(torch.sum(sm_scales_expert*fc1_smooth_ratio1, dim=0), sm_scales_gate)
        merge_fc1_smooth_scale2 = torch.max(torch.sum(sm_scales_expert*fc1_smooth_ratio2, dim=0), sm_scales_gate)
        merge_fc1_smooth_scale3 = torch.max(torch.sum(sm_scales_expert*fc1_smooth_ratio3, dim=0), sm_scales_gate)
        merge_fc1_smooth_scale = torch.max(torch.max(torch.max(merge_fc1_smooth_scale0, merge_fc1_smooth_scale1), merge_fc1_smooth_scale2), merge_fc1_smooth_scale3)

    return merge_fc1_smooth_scale


@torch.no_grad()
def get_scale(fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype).clamp(min=1e-5)
    weight_scales = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)).clamp(min=1e-5)
    return scales


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    scales = scales.to(device).to(dtype)
    
    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    
    device, dtype = fc2.weight.device, fc2.weight.dtype
    scales = scales.to(device).to(dtype)
    
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def smooth_lm(model, scales, act_per_channel_scales, act_samples, weight_scores, router_logits, fc1_scale_merge="max", alpha=0.6, otsu_ratio=0.8, otsu_smooth_rate=0.7, logger=None):

    for name, module in model.named_modules():
        if isinstance(module, (OlmoeDecoderLayer)):
            expert_num = 64
            select_expert_num = 8
            share_expert_num = 0
            dev = module.self_attn.q_proj.weight.device
            logger.info(f"[smooth_lm] name: {name}")
            
            logger.info("smooth qkv")
            attn_ln = module.input_layernorm
            qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
            sp_name = name + '.self_attn.q_proj'
            logger.info(f"[smooth_lm] sp_name: {sp_name}")
            logger.info("get_scale")
            sm_scales = get_scale(qkv, scales[sp_name], alpha)
            logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
            smooth_ln_fcs(attn_ln, qkv, sm_scales)

            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                logging.info("smooth vo")
                prev_op = module.self_attn.v_proj
                layers = [module.self_attn.o_proj]
                sp_name = name + '.self_attn.o_proj'
                logger.info(f"[smooth_lm] sp_name: {sp_name}")
                logger.info("get_scale")
                sm_scales = get_scale(layers[0], scales[sp_name], alpha)
                logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
                scale_fc_fc(prev_op, layers[0], sm_scales)
            

            logging.info(f"smooth mlp.gate, up_proj/gate_proj in mlp.experts")
            ffn_ln = module.post_attention_layernorm
            fc1 = [module.mlp.gate] + [module.mlp.experts[i].gate_proj for i in range(expert_num)] + [module.mlp.experts[i].up_proj for i in range(expert_num)]
            fc1_split = [module.mlp.gate] + [[module.mlp.experts[i].gate_proj, module.mlp.experts[i].up_proj] for i in range(expert_num)]


            logger.info("calcu_outlier_mask")
            sp_name_list = []
            sp_name_list += [name + '.mlp.gate']
            for i in range(expert_num):
                sp_name_list += [name + f'.mlp.experts.{i}.up_proj']
            logging.info(f"len(sp_name_list): {len(sp_name_list)}")
            logging.info(f"len(act_samples.keys()): {len(act_samples.keys())}")
            logging.info(f"len(weight_scores.keys()): {len(weight_scores.keys())}")

            sm_scales_list = []
            act_samples_list = []
            weight_scores_list = []
            for i in range(len(sp_name_list)):
                sp_name = sp_name_list[i]
                logger.info(f"[smooth_lm] sp_name: {sp_name}")
                sm_scales = get_scale(fc1_split[i], scales[sp_name], alpha)
                logger.info("original scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
                sm_scales_list.append(sm_scales)
                act_samples_list.append(act_samples[sp_name])
                weight_scores_list.append(weight_scores[sp_name])
            logging.info(f"len(sm_scales_list): {len(sm_scales_list)}")
            logging.info(f"len(act_samples_list): {len(act_samples_list)}")
            logging.info(f"len(weight_scores_list): {len(weight_scores_list)}")
            sm_scales = merge_scale(sm_scales_list, act_samples_list, weight_scores_list, router_logits[name + '.mlp.gate'], expert_num_list=[expert_num,select_expert_num,share_expert_num], fc1_scale_merge=fc1_scale_merge)
            logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
            smooth_ln_fcs(ffn_ln, fc1, sm_scales)

            for i in range(expert_num):
                logging.info(f"smooth down_proj in mlp.experts.{i}")
                prev_op = module.mlp.experts[i].up_proj
                layers = [module.mlp.experts[i].down_proj]
                sp_name = name + f'.mlp.experts.{i}.down_proj'
                if sp_name in scales.keys():
                    logger.info(f"[smooth_lm] sp_name: {sp_name}")
                    logger.info("calcu_outlier_mask")
                    sm_scales = calcu_outlier_mask(act_per_channel_scales[0][sp_name], act_per_channel_scales[1][sp_name], dev, ratio=otsu_ratio, smooth_rate=otsu_smooth_rate)
                    logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
                    scale_fc_fc(prev_op, layers[0], sm_scales)

        elif isinstance(module, (PanguProMoEDecoderLayer)):
            expert_num = 64
            select_expert_num = 8
            share_expert_num = 1
            dev = module.self_attn.q_proj.weight.device
            logger.info(f"[smooth_lm] name: {name}")
            
            logger.info("smooth qkv")
            attn_ln = module.input_layernorm
            qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
            sp_name = name + '.self_attn.q_proj'
            logger.info(f"[smooth_lm] sp_name: {sp_name}")
            logger.info("get_scale")
            sm_scales = get_scale(qkv, scales[sp_name], alpha)
            logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
            smooth_ln_fcs(attn_ln, qkv, sm_scales)

            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                logging.info("smooth vo")
                prev_op = module.self_attn.v_proj
                layers = [module.self_attn.o_proj]
                sp_name = name + '.self_attn.o_proj'
                logger.info(f"[smooth_lm] sp_name: {sp_name}")
                logger.info("get_scale")
                sm_scales = get_scale(layers[0], scales[sp_name], alpha)
                logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
                scale_fc_fc(prev_op, layers[0], sm_scales)
            

            logging.info(f"smooth mlp.gate, up_proj/gate_proj in mlp.experts")
            ffn_ln = module.post_attention_layernorm
            fc1 = [module.mlp.gate] + [module.mlp.experts[i].gate_proj for i in range(expert_num)] + [module.mlp.experts[i].up_proj for i in range(expert_num)] + [module.mlp.shared_expert.gate_proj, module.mlp.shared_expert.up_proj]
            fc1_split = [module.mlp.gate] + [[module.mlp.experts[i].gate_proj, module.mlp.experts[i].up_proj] for i in range(expert_num)] + [[module.mlp.shared_expert.gate_proj, module.mlp.shared_expert.up_proj]]


            logger.info("calcu_outlier_mask")
            sp_name_list = []
            sp_name_list += [name + '.mlp.gate']
            for i in range(expert_num):
                sp_name_list += [name + f'.mlp.experts.{i}.up_proj']
            sp_name_list += [name + f'.mlp.shared_expert.up_proj']
            logging.info(f"len(sp_name_list): {len(sp_name_list)}")
            logging.info(f"len(act_samples.keys()): {len(act_samples.keys())}")
            logging.info(f"len(weight_scores.keys()): {len(weight_scores.keys())}")

            sm_scales_list = []
            act_samples_list = []
            weight_scores_list = []
            for i in range(len(sp_name_list)):
                sp_name = sp_name_list[i]
                logger.info(f"[smooth_lm] sp_name: {sp_name}")
                sm_scales = get_scale(fc1_split[i], scales[sp_name], alpha)
                logger.info("original scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
                sm_scales_list.append(sm_scales)
                act_samples_list.append(act_samples[sp_name])
                weight_scores_list.append(weight_scores[sp_name])
            logging.info(f"len(sm_scales_list): {len(sm_scales_list)}")
            logging.info(f"len(act_samples_list): {len(act_samples_list)}")
            logging.info(f"len(weight_scores_list): {len(weight_scores_list)}")
            sm_scales = merge_scale(sm_scales_list, act_samples_list, weight_scores_list, router_logits[name + '.mlp.gate'], expert_num_list=[expert_num,select_expert_num,share_expert_num], fc1_scale_merge=fc1_scale_merge)
            logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
            smooth_ln_fcs(ffn_ln, fc1, sm_scales)

            for i in range(expert_num):
                logging.info(f"smooth down_proj in mlp.experts.{i}")
                prev_op = module.mlp.experts[i].up_proj
                layers = [module.mlp.experts[i].down_proj]
                sp_name = name + f'.mlp.experts.{i}.down_proj'
                if sp_name in scales.keys():
                    logger.info(f"[smooth_lm] sp_name: {sp_name}")
                    logger.info("calcu_outlier_mask")
                    sm_scales = calcu_outlier_mask(act_per_channel_scales[0][sp_name], act_per_channel_scales[1][sp_name], dev, ratio=otsu_ratio, smooth_rate=otsu_smooth_rate)
                    logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
                    scale_fc_fc(prev_op, layers[0], sm_scales)

            prev_op = module.mlp.shared_expert.up_proj
            layers = [module.mlp.shared_expert.down_proj]
            sp_name = name + f'.mlp.shared_expert.down_proj'
            logger.info(f"[smooth_lm] sp_name: {sp_name}")
            logger.info("calcu_outlier_mask")
            sm_scales = calcu_outlier_mask(act_per_channel_scales[0][sp_name], act_per_channel_scales[1][sp_name], dev, ratio=otsu_ratio, smooth_rate=otsu_smooth_rate)
            logger.info("scale={},max={},min={},scale.shape: {}".format(sm_scales,sm_scales.max(), sm_scales.min(), sm_scales.shape))
            scale_fc_fc(prev_op, layers[0], sm_scales)
