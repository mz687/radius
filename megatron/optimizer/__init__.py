# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD
from apex.optimizers import FusedLAMB as LAMB

from megatron import get_args
from megatron.model import LayerNorm
from megatron.model import GPTModel

from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer

import torch

def _get_params_for_weight_decay_optimization(modules):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    args = get_args()

    no_weight_decay_params_idx = []
    weight_decay_params_idx = []

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    idx = 0
    for module in modules:
        for module_ in module.modules():
            if isinstance(module_, LayerNorm):
                no_weight_decay_params['params'].extend(
                    [p for p in list(module_._parameters.values())
                     if p is not None])
                
                len_beforehand=len(no_weight_decay_params_idx)
                no_weight_decay_params_idx.extend([idx+i for i, p in enumerate(module_._parameters.values()) if p is not None])
                idx += len(no_weight_decay_params_idx) - len_beforehand
            else:
                weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                     if p is not None and n != 'bias'])
                no_weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                     if p is not None and n == 'bias'])
                
                no_weight_len_beforehand = len(no_weight_decay_params_idx)
                weight_len_beforehand = len(weight_decay_params_idx)
                weight_decay_params_idx.extend([idx+i for i,(n,p) in enumerate(module_._parameters.items()) if p is not None and n != 'bias'])
                no_weight_decay_params_idx.extend([idx+i for i,(n,p) in enumerate(module_._parameters.items()) if p is not None and n == 'bias'])
                
                idx += len(weight_decay_params_idx) - weight_len_beforehand
                idx += len(no_weight_decay_params_idx) - no_weight_len_beforehand 

    args.optimizer_layer_order = weight_decay_params_idx + no_weight_decay_params_idx  

    return weight_decay_params, no_weight_decay_params


def get_megatron_optimizer(model):
    args = get_args()

    # Base optimizer.
    param_groups = _get_params_for_weight_decay_optimization(model)

    if args.grad_comp and "Correction" in args.grad_comp_type:
        # # we move Adam operations to the place before params-all-reduce
        # All the operations are done inside reducer.reduce 
        optimizer = SGD(param_groups,
                        lr=args.lr,
                        weight_decay=0,
                        momentum=0)
        
        args.optimizer_var = optimizer # creates a global reference for optimizer 
        
    else:
        if args.optimizer == 'adam':
            optimizer = Adam(param_groups,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(args.adam_beta1, args.adam_beta2),
                            eps=args.adam_eps)
        elif args.optimizer == 'sgd':
            optimizer = SGD(param_groups,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            momentum=args.sgd_momentum)
        elif args.optimizer == 'lamb':
            optimizer = LAMB(param_groups,
                            lr=args.lr,
                            betas=(args.lamb_beta1, args.lamb_beta2),
                            eps=args.lamb_eps,
                            weight_decay=args.weight_decay)
        else:
            raise Exception('{} optimizer is not supported.'.format(
                args.optimizer))
    
    if torch.distributed.get_rank() == 0:
        print(f"memory usage after creating optimizer ({args.optimizer}): {torch.cuda.memory_allocated(0)/1024**2}MiB" )


    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    if args.DDP_impl == 'local':
        params_have_main_grad = True

    if args.fp16 or args.bf16:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None
        # Constant loss scale.
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)
        # Dynamic loss scale.
        else:
            if args.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=args.initial_loss_scale,
                    min_scale=args.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=args.loss_scale_window,
                    hysteresis=args.hysteresis)

        # Megatron optimizer.
        return Float16OptimizerWithFloat16Params(optimizer,
                                                 args.clip_grad,
                                                 args.log_num_zeros_in_grad,
                                                 params_have_main_grad,
                                                 args.use_contiguous_buffers_in_local_ddp,
                                                 args.bf16,
                                                 grad_scaler)

    # FP32.
    return FP32Optimizer(optimizer, args.clip_grad,
                         args.log_num_zeros_in_grad,
                         params_have_main_grad,
                         args.use_contiguous_buffers_in_local_ddp)
