import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from typing import List, Optional, Tuple, Union
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.ea_norm import EaOlmoeRMSNorm
from collections import OrderedDict

import logging
import math
import sys

from models.olmoe.configuration_olmoe import OlmoeConfig
from models.olmoe.modeling_olmoe import OlmoeRotaryEmbedding, apply_rotary_pos_emb, repeat_kv


from models.transformation import *



class QuantOlmoeMLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,        
        args=None,
        module_name="mlp"
    ):
        super().__init__()
        
        self.gate_proj = QuantLinear(org_module.gate_proj,
                                           args.gate_weight_quant_params,
                                           args.gate_act_quant_params, f"{module_name}.gate_proj")
        self.down_proj = QuantLinear(org_module.down_proj,
                                           args.down_weight_quant_params,
                                           args.down_act_quant_params, f"{module_name}.down_proj")
        self.up_proj = QuantLinear(org_module.up_proj,
                                           args.up_weight_quant_params,
                                           args.up_act_quant_params, f"{module_name}.up_proj")
        self.act_fn = ACT2FN[hidden_act]
        self.init_eaquant_params = torch.tensor(0) if args.gate_weight_quant_params['quant_method'] == 'eaquant' else torch.tensor(1)

    def forward(self, x):
        if not self.init_eaquant_params:
            self.init_eaquant_params = torch.tensor(1)
            act = self.act_fn(self.gate_proj(x))
            self.up_proj.copy_quantizers_eaquant_params(self.gate_proj)
            mul = act * self.up_proj(x)
            return self.down_proj(mul)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QuantOlmoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, 
                 org_module: nn.Module,
                 config: OlmoeConfig,
                 layer_idx: Optional[int] = None,
                 args=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
            
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.k_proj = QuantLinear(
            org_module.k_proj,
            args.k_weight_quant_params,
            args.k_act_quant_params, "k_proj"
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
            args.v_weight_quant_params,
            args.v_act_quant_params, "v_proj"
        )
        self.q_proj = QuantLinear(
            org_module.q_proj,
            args.q_weight_quant_params,
            args.q_act_quant_params, "q_proj"
        )
        self.o_proj = QuantLinear(
            org_module.o_proj, args.o_weight_quant_params, args.o_act_quant_params, "o_proj"
        )
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, "qkt_matmul", matmul_func=torch.matmul, rotate=None
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, "pv_matmul", matmul_func=torch.matmul, rotate=None
        )

        self.use_weight_quant = False
        self.use_act_quant = False
        self.init_eaquant_params = torch.tensor(0) if args.gate_weight_quant_params['quant_method'] == 'eaquant' else torch.tensor(1)

        self.q_norm = EaOlmoeRMSNorm(org_module.q_norm,eps=org_module.q_norm.variance_epsilon)
        self.k_norm = EaOlmoeRMSNorm(org_module.k_norm,eps=org_module.k_norm.variance_epsilon)
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if not self.init_eaquant_params:
            self.k_proj.copy_quantizers_eaquant_params(self.q_proj)
        key_states = self.k_norm(self.k_proj(hidden_states)).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if not self.init_eaquant_params:
            self.v_proj.copy_quantizers_eaquant_params(self.q_proj)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)


        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        self.init_eaquant_params = torch.tensor(1)

        return attn_output, attn_weights, past_key_value

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)
                
class QuantOlmoeSparseMoeBlock(nn.Module):
    def __init__(
        self, 
        org_module: nn.Module,
        config,
        args=None):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = QuantLinear(org_module.gate, args.router_weight_quant_params, args.router_act_quant_params, "router")

        self.experts = nn.ModuleList([QuantOlmoeMLP(org_module=org_module.experts[i], 
                                                    hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act,
                                                    args=args, module_name=f"experts.{i}") for i in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0) # torch.Size([64, 8, 800])

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # print("[QuantizedOlmoeSparseMoeBlock] idx, top_x, len(top_x): ", idx, top_x, len(top_x))
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            if len(top_x)>0:
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
                
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class QuantOlmoeDecoderLayer(nn.Module):
    def __init__(self, 
                 config: OlmoeConfig,
                 layer_idx: int,
                 ori_layer,
                 args):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = QuantOlmoeAttention(org_module=ori_layer.self_attn,
            config=config,
            layer_idx=layer_idx,
            args=args)
        
        self.mlp = QuantOlmoeSparseMoeBlock(
            org_module=ori_layer.mlp,
            config=config,
            args=args)
        self.input_layernorm = EaOlmoeRMSNorm(ori_layer.input_layernorm,eps=ori_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = EaOlmoeRMSNorm(ori_layer.post_attention_layernorm,eps=ori_layer.post_attention_layernorm.variance_epsilon)
        self.rotary_emb = OlmoeRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        if position_embeddings is None:
            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs
    

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)
      

    def clear_temp_variable(self):
       for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

            
    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)  

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)  


    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()


    def register_eaquant_params(self):        
        for name, module in self.named_modules():
            if isinstance(module, QuantOlmoeMLP) or isinstance(module, QuantOlmoeAttention):
                delattr(module, 'init_eaquant_params')
                module.register_buffer('init_eaquant_params', torch.tensor(1))
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_eaquant_params(scale_zp = True)
                module.act_quantizer.register_eaquant_params(scale_zp = False)
    
    def load_eaquant_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('R') > -1 or k.find('permutation_list') > -1 or k.find('init_eaquant_params') > -1 or k.find('scales') > -1 or k.find('zeros') > -1:
                if "shared_expert" not in k:
                    k = k.replace("experts.", "experts[").replace(".gate_proj", "].gate_proj").replace(".up_proj", "].up_proj").replace(".down_proj", "].down_proj")
                exec(f'self.{k} = v.to(device)')
    
    def load_smooth_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('smooth') > -1:
                # exec(f'self.{k} = v')
                self.register_parameter(k, torch.nn.Parameter(v.to(device), requires_grad=False))
    
    def load_post_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('post') > -1:
                # exec(f'self.{k} = v')
                rg = False if k.find('down') > -1 else True
                self.register_parameter(k, torch.nn.Parameter(v.to(device), requires_grad=rg))

    def load_lwc_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('bound_factor') > -1:
                v = torch.nn.Parameter(v.to(device))
                if "shared_expert" not in k:
                    k = k.replace("experts.", "experts[").replace(".gate_proj", "].gate_proj").replace(".up_proj", "].up_proj").replace(".down_proj", "].down_proj")
                exec(f'self.{k} = v.to(device)')
