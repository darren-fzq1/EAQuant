from collections import OrderedDict
from quantize.int_linear import QuantLinear
import torch
import torch.nn as nn
from quantize.int_matmul import QuantMatMul
from quantize.quantizer import UniformAffineQuantizer
from models.transformation import *
import pickle
from quantize.const import CLIPMIN

def smooth_parameters(model, use_shift=True):
    params = []
    for n, m in model.named_parameters():
        if n.find('smooth') > -1:
            params.append(m)
    return iter(params)

def let_parameters(model, use_shift=True):
    params = []
    # template = "smooth" if use_shift else "smooth_scale"
    template = "post_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)  


def get_eaquant_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1:
            params.append(m)
    return iter(params)  

def get_post_parameters(model):
    params = []
    template = "post_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1:
            params.append(m)
    return iter(params)

def set_requires_grad(it, requires_grad):
    for param in it:
        param.requires_grad = requires_grad

def eaquant_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_buffers():
        if name.find('init_eaquant_params') > -1 or name.find('R') > -1 or name.find('permutation_list') > -1 or name.find('scales') > -1 or name.find('zeros') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

def post_rotate_quant_temporary(model, args):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "post_scale" in name:
                    module.data = truncate_number(module)
        post_fcs_temporary([model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj], model.qkv_post_scale)
        post_fcs_temporary([model.mlp.up_proj,model.mlp.gate_proj], model.fc1_post_scale)
        post_fcs_temporary(model.mlp.down_proj, model.down_post_scale)
        post_fcs_temporary(model.self_attn.o_proj, model.out_post_scale)



@torch.no_grad()
def post_quant_inplace(model, args):
    if args.let:
        for name, module in model.named_parameters():
            if "post_scale" in name:
                module.data = truncate_number(module)
            if isinstance(module, QuantLinear):
                if module.act_quantizer.let_s is not None:
                    module.act_quantizer.let_s.requires_grad = False
                if module.weight_quantizer.let_s is not None:
                    module.weight_quantizer.let_s.requires_grad = False
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

def set_registered_x_none(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.registered_x = None
            module.act_quantizer.registered_x = None


@torch.no_grad()
def set_init_eaquant_params_state(model, mode):
    if isinstance(mode, bool):
        mode = torch.tensor(mode)
    for name, module in model.named_modules():
        if hasattr(module, "init_eaquant_params"):
            module.init_eaquant_params = mode


@torch.no_grad()
def set_calibration_state(model, mode):
    if isinstance(mode, bool):
        mode = torch.tensor(mode)
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.do_calibration = mode



@torch.no_grad()
def quant_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight, return_no_quant=False)

@torch.no_grad()
def quant_soft_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight, return_no_quant=True)

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)
