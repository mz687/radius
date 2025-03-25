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
from collections import OrderedDict

from abc import ABC
from abc import abstractmethod

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist

from megatron import get_args
from megatron import mpu
from .module import MegatronModule

from megatron.mpu import PowerSGDReducer

from megatron.mpu import RandomizedSVDReducer

from megatron.mpu import StableTopKReducerWithRangeBucketsEFCorrectionResIsGrad
from megatron.mpu import StableTopKReducerWithRangeBucketsEFCorrectionResIsGradPerLayer
from megatron.mpu import TopKReducerEFCorrectionResIsGradWarmup

class MemoryBuffer:

    def __init__(self, numel, dtype):
        self.numel = numel
        self.dtype = dtype
        self.data = torch.zeros(self.numel,
                                dtype=self.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)


    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()


    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor
    
    def get_flat_to_end(self, start_index):
        """Return a flat tensor starting at `start_index`."""
        end_index = self.numel
        assert start_index < self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(-1)
        return buffer_tensor        



class DistributedDataParallelBase(MegatronModule, ABC):
    """Abstract class for DDP."""

    def __init__(self, module):
        super(DistributedDataParallelBase, self).__init__()
        # Keep a pointer to the model.
        self.module = module


    @abstractmethod
    def allreduce_gradients(self):
        pass


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix,
                                                          keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)



class DistributedDataParallel(DistributedDataParallelBase):
    """DDP with contiguous buffers options to store and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)
    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
    """

    def __init__(self, module,
                 accumulate_allreduce_grads_in_fp32,
                 use_contiguous_buffers):

        super(DistributedDataParallel, self).__init__(module)

        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers

        """
        [TODO] We need arguments for parsing grad comp arguments
        """
        self.args = get_args()        
        if self.args.grad_comp:
            """
            [TODO] We need to make error feedback memory for Grad Compression
            """
            # Error-feedback memory
            # self.memories = [torch.zeros_like(param) for param in self.module.parameters()]
            # This initialization is in below part
            # self.memories = {}
            # [TODO] This EF feature should be implemented in reducer class
            #        so that we can use reducer and EF feature freely (!!! important to implement !!!)
            """
            [TODO] We need another buffer for gradient compression
            """
            # Do we really need reducer buffer? Yes...
            # self.reducer_buffer = [torch.zeros_like(param) for param in self.module.parameters()]
            # This initialization is in below part
            self.reducer_buffer = {}
            
            if self.args.grad_comp_type == 'PowerSGD':
                """
                [TODO] We need custom reducer for Grad Compression
                """
                self.reducer = PowerSGDReducer(random_seed=self.args.comp_seed, device=torch.cuda.current_device(),\
                                                group=mpu.get_data_parallel_group() , group_num=mpu.get_data_parallel_world_size(),\
                                                rank=self.args.grad_comp_rank, start_iter=self.args.train_iters*self.args.grad_comp_warm_up, \
                                                use_error_feedback=self.args.use_error_feedback, fp16=self.args.fp16)
                # [TODO] argument for start_iter should be implemented later !!!
            elif self.args.grad_comp_type == 'RandomizedSVD':
                self.reducer = RandomizedSVDReducer(random_seed=self.args.comp_seed, device=torch.cuda.current_device(),\
                                                    group=mpu.get_data_parallel_group(), group_num=mpu.get_data_parallel_world_size(),\
                                                    rank=self.args.grad_comp_rank, start_iter=self.args.train_iters*self.args.grad_comp_warm_up, \
                                                    use_error_feedback=self.args.use_error_feedback, fp16=self.args.fp16)
            # [TODO] Add new compression algorithms. 
            #        Check compressors in compression.py to see all the supported comp algo.
            # [DONE] Remember to add new CMD argument to arguments.py
            elif self.args.grad_comp_type == 'StableTopKReducerWithRangeBucketsEFCorrectionResIsGrad':
                self.reducer = StableTopKReducerWithRangeBucketsEFCorrectionResIsGrad(random_seed=self.args.comp_seed,
                                                                                    device=torch.cuda.current_device(),
                                                                                    group=mpu.get_data_parallel_group(),
                                                                                    group_num=mpu.get_data_parallel_world_size(),
                                                                                    beta1=self.args.adam_beta1,
                                                                                    beta2=self.args.adam_beta2,
                                                                                    epsilon=self.args.adam_eps,
                                                                                    density=self.args.density, 
                                                                                    stable_topk_interval=self.args.stable_topk_interval, 
                                                                                    stable_topk_threshold=self.args.stable_topk_threshold, 
                                                                                    stable_topk_range=self.args.stable_topk_range,
                                                                                    stable_topk_warmup_method=self.args.stable_topk_warmup_method)
            elif self.args.grad_comp_type == 'TopKReducerEFCorrectionResIsGradWarmup':
                self.reducer = TopKReducerEFCorrectionResIsGradWarmup(random_seed=self.args.comp_seed,
                                                                device=torch.cuda.current_device(),
                                                                group=mpu.get_data_parallel_group(),
                                                                group_num=mpu.get_data_parallel_world_size(),
                                                                beta1=self.args.adam_beta1,
                                                                beta2=self.args.adam_beta2,
                                                                epsilon=self.args.adam_eps,
                                                                density=self.args.density, 
                                                                stable_topk_threshold=self.args.stable_topk_threshold, 
                                                                stable_topk_warmup_method=self.args.stable_topk_warmup_method,
                                                                stable_topk_density_warmup_steps=self.args.stable_topk_density_warmup_steps)
            elif self.args.grad_comp_type == 'StableTopKReducerWithRangeBucketsEFCorrectionResIsGradPerLayer':
                self.reducer = StableTopKReducerWithRangeBucketsEFCorrectionResIsGradPerLayer(random_seed=self.args.comp_seed,
                                                                                    device=torch.cuda.current_device(),
                                                                                    group=mpu.get_data_parallel_group(),
                                                                                    group_num=mpu.get_data_parallel_world_size(),
                                                                                    density=self.args.density, 
                                                                                    stable_topk_interval=self.args.stable_topk_interval, 
                                                                                    stable_topk_threshold=self.args.stable_topk_threshold, 
                                                                                    stable_topk_warmup_method=self.args.stable_topk_warmup_method)
            else:
                """
                [TODO] We need custom reducer for Grad Compression
                """
                self.reducer = PowerSGDReducer(random_seed=self.args.comp_seed, device=torch.cuda.current_device(),\
                                                group=mpu.get_data_parallel_group() , group_num=mpu.get_data_parallel_world_size(),\
                                                rank=self.args.grad_comp_rank, start_iter=self.args.train_iters*self.args.grad_comp_warm_up, \
                                                use_error_feedback=self.args.use_error_feedback, fp16=self.args.fp16)


        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        # For module that contains multiple layers,
        # we need to perform topk on each layer independently.
        self._layer_start_end_idx = None
        self._layer_idx = None
        # Used for loading optimizer_state
        self._layer_idx_to_start_end_idx = None

        if self.use_contiguous_buffers:
            self._grad_buffers = {}
            self._layer_start_end_idx = {}
            self._layer_idx = {}
            self._layer_idx_to_start_end_idx={}

            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            """
            [TODO] Error FeedBack Memory and Buffer ????
            """
            # First calculate total number of elements per type.
            # Normally, Just one type is used ... len(type_num_elements) == 1
            type_num_elements = {}
            self.non_emb_start_idx = 0
            if self.args.emb_comm_opt: # add word embeddings at first !
                for i, (name, param) in enumerate(self.module.named_parameters()):
                    if 'word_embeddings' in name:
                        if param.requires_grad:
                                dtype = _get_buffer_type(param)
                                
                                # Store the start and end index of a layer
                                if dtype not in self._layer_start_end_idx:
                                    self._layer_start_end_idx[dtype] = []
                                self._layer_start_end_idx[dtype].append((type_num_elements.get(dtype, 0), 
                                                                         type_num_elements.get(dtype, 0)+param.data.nelement()))
                                
                                if dtype not in self._layer_idx:
                                    self._layer_idx[dtype] = {}
                                self._layer_idx[dtype][i] = name
                                
                                type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                                        + param.data.nelement()
                                self.non_emb_start_idx = type_num_elements[dtype]                                 
                    else:
                        continue
            for i, (name, param) in enumerate(self.module.named_parameters()):
                
                if self.args.emb_comm_opt and 'word_embeddings' in name:
                    continue    
                if param.requires_grad:
                        dtype = _get_buffer_type(param)
                        
                        # Store the start and end index of a layer
                        if dtype not in self._layer_start_end_idx:
                            self._layer_start_end_idx[dtype] = []
                        self._layer_start_end_idx[dtype].append((type_num_elements.get(dtype, 0), 
                                                                    type_num_elements.get(dtype, 0)+param.data.nelement()))
                         
                        if dtype not in self._layer_idx:
                            self._layer_idx[dtype] = {}
                        self._layer_idx[dtype][i] = name
                                
                        
                        type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                                + param.data.nelement()

            # Allocate the buffer.
            for dtype, num_elements in type_num_elements.items():
                self._grad_buffers[dtype] = MemoryBuffer(num_elements, dtype)
                # Allocate EF memory with right MemoryBuffer format
                if self.args.grad_comp :
                    # self.memories[dtype] = MemoryBuffer(num_elements, dtype)
                    self.reducer_buffer[dtype] = MemoryBuffer(num_elements, dtype)

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for i, (name, param) in enumerate(self.module.named_parameters()):
                if self.args.emb_comm_opt and 'word_embeddings' in name:
                    continue 
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= param.data.nelement()

                    # For loading optimizer_state, must have the correct mapping 
                    if dtype not in self._layer_idx_to_start_end_idx:
                        self._layer_idx_to_start_end_idx[dtype] = {}
                    self._layer_idx_to_start_end_idx[dtype][i] = (type_num_elements[dtype],
                                                                    type_num_elements[dtype]+param.data.nelement())
                       

                    # So param.main_grad is view of grad_buffers
                    param.main_grad = self._grad_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype])
            # embedding grads are saved in early memories (reverse order)
            if self.args.emb_comm_opt: # word embedding opt need main_grad too !
                for i, (name, param) in enumerate(self.module.named_parameters()):
                    if 'word_embeddings' in name: 
                        if param.requires_grad:
                            dtype = _get_buffer_type(param)
                            type_num_elements[dtype] -= param.data.nelement()
                            
                            # For loading optimizer_state, must have the correct mapping 
                            if dtype not in self._layer_idx_to_start_end_idx:
                                self._layer_idx_to_start_end_idx[dtype] = {}
                            self._layer_idx_to_start_end_idx[dtype][i] = (type_num_elements[dtype],
                                                                          type_num_elements[dtype]+param.data.nelement())
                   

                            # So param.main_grad is view of grad_buffers
                            param.main_grad = self._grad_buffers[dtype].get(
                                param.data.shape, type_num_elements[dtype])
                    else:
                        continue

            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for name, param in self.module.named_parameters():
                # if self.args.emb_comm_opt and 'word_embeddings' in name:
                #     continue 
                if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param))
                    self.grad_accs.append(grad_acc)


    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad.data is not None:
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                param.grad = None
        return param_hook


    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        assert self._grad_buffers is not None, 'buffers are not initialized.'
        for _, buffer_ in self._grad_buffers.items():
            buffer_.zero()


    """
    [TODO] Adapt Gradient Compression on Data Parallel...
        [Finished]   1. PowerSGD: let's use powersgd first
        [Unfinished] 2. ScaleCom (no idea why optimus-cc did not implement this)
        [TODO]:      3. stable-oktopk
    """
    def allreduce_gradients(self, train_step):
        # if torch.distributed.get_rank() == 0:
        #     size = 0
        #     for _, buffer_ in self._grad_buffers.items():
        #         size += buffer_.numel() * buffer_.element_size()
        #     print("self._grad_buffers in bytes: ", size )
        """Reduce gradients across data parallel ranks."""
        # If we have buffers (i.e. use_contiguous_buffers=1), simply reduce the data in the buffer.
        if self._grad_buffers is not None:
            # Divide into two cases:
            # 1. stable-topkA and other compression algorithms
            # 2. optimus-cc's implementation of PowerSGD
            if self.args.grad_comp:
                # For stable-topkA and other compression algorithms
                if "TopK" in self.args.grad_comp_type:
                    need_callback = self.reducer.reduce(train_step,
                                                        self.module, 
                                                        self._layer_idx,
                                                        self._layer_idx_to_start_end_idx,
                                                        self._grad_buffers, 
                                                        self.reducer_buffer, 
                                                        self.non_emb_start_idx)
                    if need_callback:
                        for (_, buffer_), (_, reduced_) in zip(self._grad_buffers.items(), self.reducer_buffer.items()):
                            buffer_.data[:] = reduced_.data[:]
                else:
                    """
                    [TODO] For grad comp, EF-SGD is default mechanism.
                    """
                    # For Error-Feedback SGD
                    
                    # [debugging] print shapes
                    # print(len(self._grad_buffers.items()))
                    # print(len(self.memories))

                    # For Memory Buffer (contiguous) Type -> pass self.module.parameters()
                    # for caculate size(shape) of each layer
                    
                    # This feature was merged into Reducer
                    # caculate EF error for each 'DataType' !! 
                    # for (_, buffer_), (_, e_) in zip(self._grad_buffers.items(), self.memories.items()):
                    #     buffer_.data[:] += e_.data[:]

                    """
                    [TODO] Compressed All-Reduce by Reducer Algorithm
                    """
                    # Arguments: 
                    # self.module
                    # self._grad_buffers.items()
                    # self.memories.items() 
                    need_callback = self.reducer.reduce(self.module, \
                                                        self._grad_buffers, 
                                                        self.reducer_buffer, 
                                                        self.non_emb_start_idx)

                    # all-reduce results are in self.reducer_buffer
                    if need_callback:
                        for (_, buffer_), (_, reduced_) in zip(self._grad_buffers.items(), self.reducer_buffer.items()):
                            # buffer_.data.copy_(reduced_.data) # [TODO] Watch out this line... I think I made some mistake :(
                            buffer_.data[:] = reduced_.data[:]

            else:
                # Normal process of all reduce
                for _, buffer_ in self._grad_buffers.items():
                    if self.args.emb_comm_opt:
                        non_emb_buffer_ = buffer_.get_flat_to_end(self.non_emb_start_idx)
                        non_emb_buffer_ /= mpu.get_data_parallel_world_size()
                        torch.distributed.all_reduce(
                            non_emb_buffer_, group=mpu.get_data_parallel_group())
                    else:
                        buffer_.data /= mpu.get_data_parallel_world_size()
                        torch.distributed.all_reduce(
                            buffer_.data, group=mpu.get_data_parallel_group())

                # if dist.get_rank() == 0:
                #     print(' > DP all-reduce: ', n_elements, ' elements')
        else:
            # Otherwise, bucketize and all-reduce
            buckets = {}
            # Pack the buckets.
            for param in self.module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
                    param.main_grad = param.grad

            # For each bucket, all-reduce and copy all-reduced grads.
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                coalesced /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(
                    coalesced, group=mpu.get_data_parallel_group())
                for buf, synced in zip(grads, _unflatten_dense_tensors(
                        coalesced, grads)):
                    buf.copy_(synced)

    
    def gradient_clip(self):
        if self._grad_buffers is not None and self.args.grad_comp and "TopK" in self.args.grad_comp_type:
            if self.args.move_grad_clip_to_reducer:
                self.args.grad_norm = self.reducer.gradient_clip(self.module)
    
    
    def gradient_correction(self):
        if self.args.grad_comp and "TopK" in self.args.grad_comp_type and 'Correction' in self.args.grad_comp_type:
            self.reducer.gradient_correction(self._layer_idx,
                                            self._layer_idx_to_start_end_idx,
                                            self._grad_buffers)
    
    def resample_topk_indices(self, train_step):
        if self._grad_buffers is not None and self.args.grad_comp and "TopK" in self.args.grad_comp_type:
            if self.args.grad_comp_type == 'TopKReducerEFCorrectionResIsGradWarmup' or self.args.grad_comp_type == 'StableTopKReducerWithRangeBucketsEFCorrectionResIsGradSelectByGradDivW':
                self.reducer.resample_topk_indices(self.module,
                                                train_step,
                                                self._layer_idx,
                                                self._layer_idx_to_start_end_idx,
                                                self.non_emb_start_idx,
                                                self._grad_buffers)
            else:
                self.reducer.resample_topk_indices(train_step,
                                                self._layer_idx,
                                                self._layer_idx_to_start_end_idx,
                                                self.non_emb_start_idx,
                                                self._grad_buffers)
    
def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()