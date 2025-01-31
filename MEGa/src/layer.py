# Modified from meteora. https://github.com/NJUDeepEngine/meteora/blob/main/MoELoRA/layer.py

# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append('../src')

import os
import random
import math
import warnings
from contextlib import nullcontext
from typing import Any, List, Optional, Union, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from transformers.pytorch_utils import Conv1D


from peft.tuners.lora.layer import Conv2d, Embedding
from peft.tuners.lora.config import LoraConfig
from peft.utils.other import transpose

from tuners_utils import BaseTunerLayer, check_adapters_to_merge

# Those are fast implementation of MoELinear.
# from .layer_ops import moelinear_fwd_inner_bmm_torch, moelinear_fwd_inner_bmm_triton



# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------




class GatedLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_index = {}
        self.rmoe = 0 # why 224?
        # for key in self.r.keys:
        #     self.rmoe += self.r[key]
        in_features = self.get_base_layer().in_features
        out_features = self.get_base_layer().out_features

        self.loras = 0
        self.top_k = 2
        self.T = 1
        # gating
        # self.moe_gate = nn.Linear(in_features, 28, bias=False)

        self.lora_dropout = nn.ModuleDict({}) # todo: dict to list?
        # In PEFT LORA, lora_A and B are module dicts. Each key is an adapter name.
        # Here, meteora merged all adapters into a single lora_A and lora_B.
        # This initialization is just for placeholder.
        self.lora_A = None #nn.Linear(in_features, self.rmoe, bias=False)
        self.lora_B = None #nn.Linear(self.rmoe, out_features, bias=False)
        # For Embedding layer
        # Not used in meteora
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features # Should this be before the lora_A and B? It looks it only support dense layers.
        
    # Meat
    def add_lora(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        # no support for dropout for now.
        # no support for bias for now.
        device = self.get_base_layer().weight.device
        old_rmoe = self.rmoe
        self.rmoe += r
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.loras += 1
        self.lora_idx2name[self.loras-1] = adapter_name
        self.lora_name2idx[adapter_name] = self.loras-1
        base_dtype = self.get_base_layer().weight.dtype
        if self.lora_A is None:
            self.lora_A = nn.Linear(self.in_features, self.rmoe, bias=False).to(device, dtype=base_dtype)
            self.lora_B = nn.Linear(self.rmoe, self.out_features, bias=False).to(device, dtype=base_dtype)
            nn.init.normal_(self.lora_A.weight, std=1 / r)
            nn.init.zeros_(self.lora_B.weight)
        else:
            new_lora_A = nn.Linear(self.in_features, self.rmoe, bias=False).to(device, dtype=base_dtype)
            new_lora_B = nn.Linear(self.rmoe, self.out_features, bias=False).to(device, dtype=base_dtype)
            nn.init.normal_(new_lora_A.weight, std=1 / r)
            nn.init.zeros_(new_lora_B.weight)
            new_lora_A.weight.data[:old_rmoe, :] = self.lora_A.weight.detach()
            new_lora_B.weight.data[:, :old_rmoe] = self.lora_B.weight.detach()
            
            # Release memory for the old lora layers
            del self.lora_A
            del self.lora_B
            torch.cuda.empty_cache()
            
            self.lora_A = new_lora_A
            self.lora_B = new_lora_B
        self.lora_index[adapter_name] = len(self.lora_index.keys())
        self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        return
    
    # Don't use this.
    # This is for adding a new lora (adapter). This alone doesn't add any new parameters.
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        # r = r
        # lora_alpha = 16
        lora_dropout = 0.1
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.rmoe = 0

        self.loras = len(self.r.keys())
        n_loras = len(self.r.keys())

        # gating


        for key in self.r.keys():
            self.rmoe += self.r[key]
        in_features = self.get_base_layer().in_features
        out_features = self.get_base_layer().out_features
        # self.moe_gate = nn.Linear(in_features, n_loras+1, bias=False)
        # self.moe_gate = nn.Linear(in_features, n_loras, bias=False)
        # self.lora_A = nn.Linear(in_features, self.rmoe, bias=False)
        # self.lora_B = nn.Linear(self.rmoe, out_features, bias=False)
        # print(adapter_name, self.r.keys(), self.lora_A, self.lora_B)
        # import copy
        # lora_A_new.weight = copy.deepcopy(self.lora_A.weight)
        # lora_B_new.weight = copy.deepcopy(self.lora_B.weight)

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            if adapter_name not in self.lora_index:
                lora_index = len(self.lora_index.keys())
                self.lora_index[adapter_name] = lora_index
                # self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
                # self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
            if use_rslora:
                self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
            else:
                self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break
        self.set_adapter(self.active_adapters)

    # Don't use this.
    # This was originally used for winowhy dataset. But we need to do similar for all updates.
    def update_layer_post(self):
        in_features = self.get_base_layer().in_features
        out_features = self.get_base_layer().out_features
        n_loras = len(self.r.keys())
        # self.moe_gate = nn.Linear(in_features, n_loras, bias=False)
        self.lora_A = nn.Linear(in_features, self.rmoe, bias=False)
        self.lora_B = nn.Linear(self.rmoe, out_features, bias=False)
         # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break
    


    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class GatedLinear(nn.Module, GatedLoraLayer):
    # MoELora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        GatedLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.lora_idx2name, self.lora_name2idx = {}, {}
        self.add_lora(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)
        # self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

        # NOTE: add extra initialization for solving the speed problem for forward pass especially during inference
        self._extra_initialize()
        self._check_extra_initialize()

    def _extra_initialize(self):
        """do some extra initialization for MoELinear"""
        # the flag for whether the forward pass is prepared at once
        # it will be automatically set to True after the first forward
        self.fwd_prepared = False
        
        # to avoid unnecessary searching when mapping from idx to name
        # self.lora_idx2name, self.lora_name2idx = {}, {}

        # this is views for self.lora_A and self.lora_B to treat each lora adapter seperatedly
        # i.e. if self.lora_A.weight.shape = (L*r, h), then self.lora_A_weights.shape = (L, r, h), where self.lora_A.weight[r*i:r*(i+1), :] == self.lora_A_weights[i]
        # note that: this requires all lora adapters to have the same rank r, so we use another flag to check this condition
        self.lora_A_weights, self.lora_B_weights, self.lora_A_mask = None, None, None
        self.all_ranks_are_identical, self.unique_rank = None, None
        self.scalings, self.unique_lora_dropout = None, None

        # NOTE: accelerate forward pass (speed up for 5x, i.e. only 2x slower than single lora now)
        # compared with the one using original mixtral MoE style (10x slower than single lora)
        self.use_accelerated_fwd = os.environ.get('MOELINEAR_USE_ACCELERATE_FWD', '1') == '1'
        
        # NOTE: when use accelerate forward, set the op implementation, 
        # choosing from {'torch', 'triton'}, default 'torch' for now
        self.accelerate_fwd_backend = os.environ.get('MOELINEAR_ACCELERATE_FWD_BACKEND', 'torch')
        
        self.accelerate_fwd_backend_torch_version = os.environ.get('MOELINEAR_ACCELERATE_FWD_BACKEND_TORCH_VERSION', 'v1')
        self.accelerate_fwd_backend_triton_version = os.environ.get('MOELINEAR_ACCELERATE_FWD_BACKEND_TRITON_VERSION', 'v4')
        self.accelerate_fwd_backend_triton_group_size = 16
        
        self.accelerate_fwd_inner_func = None # this unc is set by self.accelerate_fwd_backend in self._prepare_forward automatically

        # add this to apply normal lora
        # when there's only one moe adapter with index `gt_lora_idx`
        # (now only for debugging)
        self.single_moe_lora = os.environ.get('MOELINEAR_SINGLE_MOE_LORA', None) is not None
        self.gt_lora_idx = int(os.environ.get('MOELINEAR_SINGLE_MOE_LORA', '0'))

        ############    NOTE: choose `self.fwd_inner_loop_mode` from:
        # - 'all': 
        #   - to loop over each lora adapters and apply mm to selected sub-group tokens in each iteration
        # - 'parallel': 
        #   - to parallelly loop over each lora adapters and apply mm to selected sub-group tokens in each group of corresponding cuda.stream
        #   - now only for: 
        #       1. `self.training == False` if setting the flag: `self.fwd_inner_loop_pmode4train == False`, otherwise only for inference
        #   - FIXME: this mode is not even as quick as `self.fwd_inner_loop_mode == 'all'`
        # - 'batch': to apply selected lora adapters using bmm without looping over each lora adapter
        #   - now only for:
        #       1. `self.use_accelerated_fwd == True`
        #       2. the rank r is identical to all lora adapters, i.e. `self.unique_rank` (which is default in our framework settings right now)
        #       3. `self.training == False` if setting the flag: `self.fwd_inner_loop_bmode4train == False`, otherwise only for inference
        self.fwd_inner_loop_mode = os.environ.get('MOELINEAR_FWD_INNER_LOOP_MODE', 'batch')
        
        if self.fwd_inner_loop_mode == 'parallel':
            self.fwd_inner_loop_pmode4train = False # FIXME: can this mode work well in training? if so, toggle this flag to True
            self.fwd_inner_loop_psize = 8 # the maximum number of parallel streams
        elif self.fwd_inner_loop_mode == 'batch':
            self.fwd_inner_loop_bmode4train = False # FIXME: can this mode work well in training? if so, toggle this flag to True
        
        self.parallel_fwd_inner_loop, self.fwd_inner_loop_stream_contexts = None, None # a flag with a list of stream contexts, auto create in `self._prepare_forward`
        
    def _check_extra_initialize(self):
        assert self.fwd_inner_loop_mode in ["all", "parallel", "batch",]
        assert self.accelerate_fwd_backend in ["torch", "triton",]
        assert self.accelerate_fwd_backend_torch_version in ["v1", "v2"]
        assert self.accelerate_fwd_backend_triton_version in ["v1", "v2", "v3", "v4"]

    # Leave this untested for now.
    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)
                
    # Leave this untested for now.
    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)
                
    # Leave this untested for now.
    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def _prepare_forward(self):
        """prepare some variables before forward at once"""

        ## prepare the one2one mapping between lora_idx and lora_name
        if self.lora_name2idx is None:
            self.lora_name2idx = self.lora_index
        # if self.lora_idx2name is None:
        #     self.lora_idx2name = {v:k for k, v in self.lora_name2idx.items()}

        ## prepare others beforing applying moe loras
        if self.use_accelerated_fwd and self.fwd_inner_loop_mode == "batch" \
            and (self.fwd_inner_loop_bmode4train or not self.training) and self.is_all_ranks_identical():
            if self.scalings is None:
                self.scalings = torch.zeros(self.loras, device=self.lora_A.weight.device, dtype=torch.float32) # shape = (l,)
                for i in range(self.loras): self.scalings[i] = self.scaling[self.lora_idx2name[i]]
            if self.unique_lora_dropout is None: # BUG: now just take the first dropout to use, since all p are the same for this framework now
                self.unique_lora_dropout = self.lora_dropout[self.lora_idx2name[0]]
            if self.lora_A_weights is None:
                self.lora_A_weights = self.lora_A.weight.view(self.loras, self.unique_rank, -1).transpose(1,2) # (l, h, r)
                if self.accelerate_fwd_backend == "triton":
                    h, m, r = self.lora_A_weights.shape[1], self.accelerate_fwd_backend_triton_group_size, self.unique_rank
                    self.lora_A_weights_split = torch.cat(self.lora_A_weights.split(h//m, dim=1), dim=2).contiguous() # shape from (l, h, r) to (l, h//m, r*m)
                    if self.accelerate_fwd_backend_triton_version == "v3":
                        self.lora_A_mask = torch.zeros(m, r*m, dtype=torch.bool, device=self.lora_A_weights.device) # shape: (m, r*m)
                        for i in range(m): self.lora_A_mask[i, i*r:(i+1)*r] = True
                    elif self.accelerate_fwd_backend_triton_version == "v4":
                        self.lora_A_weights_split *= self.scalings[: , None, None] # prescaling on lora_A
                        lora_A_mask1 = torch.zeros(m, r*m, dtype=self.lora_A_weights.dtype, device=self.lora_A_weights.device) # shape: (m, r*m)
                        for i in range(m): lora_A_mask1[i, i*r:(i+1)*r] = 1.
                        lora_A_mask2 = torch.cat([torch.eye(r, dtype=self.lora_A_weights.dtype, device=self.lora_A_weights.device) for _ in range(m)], dim=0).contiguous() # shape: (r*m, r)
                        self.lora_A_mask = (lora_A_mask1, lora_A_mask2) # mask olA by `olA * lA_mask1 @ lA_mask2` (shape: (m, r*m) => (m, r)) to be used in `olA @ lB`
            if self.lora_B_weights is None:
                self.lora_B_weights = self.lora_B.weight.view(-1, self.loras, self.unique_rank).permute(1,2,0).contiguous() # (l, r, hout)
                if self.accelerate_fwd_backend == "triton":
                    hout, n, r = self.lora_B_weights.shape[-1], self.accelerate_fwd_backend_triton_group_size, self.unique_rank
                    self.lora_B_weights_split = torch.cat(self.lora_B_weights.split(hout//n, dim=2), dim=1).contiguous()
                    if self.accelerate_fwd_backend_triton_version == "v3":
                        self.lora_B_mask = torch.zeros(n, r*n, dtype=self.lora_B_weights.dtype, device=self.lora_B_weights.device)
                        for i in range(n): self.lora_B_mask[i, i*r:(i+1)*r] = True
                    elif self.accelerate_fwd_backend_triton_version == "v4":
                        self.lora_B_weights_split *= self.scalings[:, None, None] # prescaling on lora_B
                        lora_B_mask1 = torch.zeros(r*n, n, dtype=self.lora_B_weights.dtype, device=self.lora_B_weights.device) # shape: (r*n, n)
                        for i in range(n): lora_B_mask1[i*r:(i+1)*r, i] = 1.
                        lora_B_mask2 = torch.cat([torch.eye(r, dtype=self.lora_B_weights.dtype, device=self.lora_B_weights.device) for _ in range(n)], dim=1).contiguous() # shape: (r, r*n)
                        self.lora_B_mask = (lora_B_mask1, lora_B_mask2) # mask olB 
            
        self.parallel_fwd_inner_loop = self.use_accelerated_fwd and self.fwd_inner_loop_mode == 'parallel' and (self.fwd_inner_loop_pmode4train or not self.training)
        if self.parallel_fwd_inner_loop:
            stream_pools = [torch.cuda.Stream() for _ in range(min(self.fwd_inner_loop_psize, self.loras))]
            self.fwd_inner_loop_stream_contexts = [torch.cuda.stream(stream_pools[i % self.fwd_inner_loop_psize]) for i in range(self.loras)]
        else:
            null_context = nullcontext()
            self.fwd_inner_loop_stream_contexts = [null_context for _ in range(self.loras)]

        
        self.fwd_prepared = True
        
    def forward(self, x: torch.Tensor, 
                gate_mode: Optional[int] = -1, # -1: use Theta_0 (User), -2: softmax gated (Assistant during inference), >=0: use the i-th gate only (Assistant during training)
                lora_weights = None, 
                *args: Any, **kwargs: Any) -> torch.Tensor:
        ## preprare output placeholder
        prev_dtype = x.dtype
        moe_logits, moe_weights, selected_loras = None, None, None
        
        ## prepare forward
        if not self.fwd_prepared: self._prepare_forward()
        
        ############################################################
        # Xu implementation
        if gate_mode == -2 and lora_weights is None:
            raise ValueError("lora_weights is None when gate_mode == -2 (softmax mode)")
        if gate_mode > self.loras-1:
            raise ValueError(f"gate_mode {gate_mode} is out of range, should be in [0, {self.loras-1}]")
        
        # residule path
        result = self.base_layer(x, *args, **kwargs)

        if self.loras == 0 or gate_mode == -1:
            return result.to(prev_dtype)
        
        # batch_size, sequence_length, hidden_dim = x.shape
        # x = x.view(-1, hidden_dim)
        # prepare lora weights
        if gate_mode == -2: # softmax gated
            expanded_lora_weights = torch.cat([w * s * torch.ones(r_i, device=x.device, dtype=x.dtype) for w, r_i, s in zip(lora_weights, self.r.values(), self.scaling.values())])
        # elif gate_mode == -1: # no lora
        #     expanded_lora_weights = torch.zeros(self.rmoe, device=x.device, dtype=x.dtype) # rmoe is the sum of all lora ranks
        elif gate_mode >= 0: # use the i-th gate only
            expanded_lora_weights = torch.zeros(self.rmoe, device=x.device, dtype=x.dtype)
            adapter_name = self.lora_idx2name[gate_mode]
            start = sum(list(self.r.values())[:gate_mode])
            end = start + self.r[adapter_name]
            expanded_lora_weights[start:end] = self.scaling[adapter_name]#1

        # Compute intermediate A transformation
        x = self.lora_A(x)
        # Apply expanded lora weights
        x = x * expanded_lora_weights.unsqueeze(0).unsqueeze(0)
        # Compute intermediate B transformation
        x = self.lora_B(x)
        
        result += x
        
        return result.to(prev_dtype)
        
        ############################################################
        
      

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

    def is_all_ranks_identical(self) -> bool:
        if self.all_ranks_are_identical is None:
            rs = set(self.r.values())
            self.all_ranks_are_identical = len(rs) == 1
            self.unique_rank = rs.pop()

        return self.all_ranks_are_identical


def dispatch_default_moe(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

   
    # Only support linear layer for now.
    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = GatedLinear(target, adapter_name, **kwargs)


    return new_module