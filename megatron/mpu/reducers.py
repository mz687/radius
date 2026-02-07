from asyncio.proactor_events import _ProactorSocketTransport
import numpy as np
import torch
import torch.distributed as dist
import os
import torch.distributed
import torch.multiprocessing as mp
import multiprocessing
import time
import itertools
from typing import List
import scipy
import sys
import math

from megatron import get_args
from megatron import get_timers
import megatron.mpu.tensor_buffer as tb
from megatron import mpu
from apex.multi_tensor_apply import multi_tensor_applier

# For support fp16
from torch.cuda.amp import autocast

try:
    from torch._six import inf
except:
    from torch import inf
import amp_C

from abc import abstractmethod

BITS_PER_BYTE = 8

# This detects any nan generated and will throw exception!
# torch.autograd.set_detect_anomaly(True)

def current_time_in_ms():
    return int(round(time.time() * 1000))


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
    
    def get_flat_from_start_to_end(self, start_index, end_index):
        """Return a flat tensor starting at `start_index` and ending at `end_index`."""
        assert start_index < self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(-1)
        return buffer_tensor    

class Reducer:
    def __init__(self, random_seed, device, group, group_num):
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024 # TODO: Why need to set this? What for?
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        if torch.distributed.is_available():
            self.n_workers = group_num
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device
        self.group = group

    def reduce(self, grad_in, grad_out, memory_out):
        """Return communicated bits"""
        raise NotImplementedError()

"""
[FP32 Version only...]
[TODO] Compare this Randomized SVD with PowerSGD Reducer
[TODO] Given codes are not using memory buffer! => change it to use that
"""
class RandomizedSVDReducer(Reducer):
    def __init__(self, random_seed, device, group, group_num, rank=8, m=32, start_iter=10,\
                use_error_feedback=False, fp16=False):
        super().__init__(random_seed, device, group, group_num)
        # compression rank
        self.rank = rank
        # matrix m
        self.m = m
        # matrix n
        self.n = None # not initialized
        # warm-up period for 10% training iteration
        self.start_iter = int(start_iter)
        # track current iteration
        self.current_iter = 0
        # EF SGD enabled?
        self.use_error_feedback = use_error_feedback
        # support fp16?
        self.fp16 = fp16
        # EF memory
        self.memories = None
        if dist.get_rank() == 0:
            self._init_printer()

    def _init_printer(self):
        print('===== Randomized SVD Reducer =====')
        print(' >> rank: ', self.rank)
        print(' >> m: ', self.m)
        print(' >> start_iter: ', self.start_iter)
        print(' >> EF on: ', self.use_error_feedback)
        print(' >> FP16: ', self.fp16)
        print('============================')

    # [TODO] should be changed into MemoryBuffer format
    # arguments: parameters(module.parameters()) / grad_in(grad buffers) / grad_out(reducer buffer) / memory_out(EF memory)
    # EF memeory is merged into this class
    def reduce(self, module, grad_in_buffers, grad_out_buffers):

        if self.current_iter < self.start_iter:
            for _, buffer_ in grad_in_buffers.items():
                buffer_.data /= self.n_workers
                all_reduce(buffer_.data, group=self.group)
            if self.current_iter % 100 == 0 and dist.get_rank() == 0:
                print(' >> Still in Warm-up Period... ', self.current_iter)
            self.current_iter += 1
        else:
            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return param.dtype

            # Collect Views of all tensors for each layer
            grad_in = []
            grad_out = []
            memory_out = []

            type_num_elements = {}
            for param in module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param.grad if param.grad is not None else param.main_grad)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                                + param.data.nelement()

            # We'll use error feedback but uninitialized
            if self.use_error_feedback and self.memories == None:
                self.memories = {}
                # if dist.get_rank() == 0:
                #     print(' >> EF Memory initialized')
                for dtype, num_elements in type_num_elements.items():
                    self.memories[dtype] = MemoryBuffer(num_elements, dtype)
                    # if dist.get_rank() == 0:
                    #     print(' >> Dtype: ', dtype, ' / # elements: ', num_elements)                                                + param.data.nelement()

            # # add EF error for each 'DataType'
            if self.use_error_feedback:
                # if dist.get_rank() == 0 and self.current_iter == 0:
                #     print(' >> EF update into input buffer')
                # copied_elements = 0
                for (_, buffer_), (_, e_) in zip(grad_in_buffers.items(), self.memories.items()):
                    # buffer_.data.nan_to_num(nan=1e-8)
                    # e_.data.nan_to_num_(nan=1e-4)
                    # print(e_.data[:100])
                    buffer_.data += e_.data
                    # copied_elements += e_.data.nelement()
                # if dist.get_rank() == 0 and self.current_iter == 0:
                #     print(' >> EF updated total ', copied_elements, ' elements')

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for param in module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param.grad if param.grad is not None else param.main_grad)
                    type_num_elements[dtype] -= param.data.nelement()
                    # So param.main_grad is view of grad_buffers
                    grad_in.append(grad_in_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype]))
                    grad_out.append(grad_out_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype]))
                    if self.use_error_feedback:
                        memory_out.append(self.memories[dtype].get(
                            param.data.shape, type_num_elements[dtype]))

            # [For Rank 1] It's out of Algorithms!!!!
            # rank1 tensors will be reduced un-compressed
            # and rank > 1 tensors should be compressed and reduced
            rank1_tensors = [
                (tensor, out, mem)
                for tensor, out, mem in zip(grad_in, grad_out, memory_out)
                if tensor.ndimension() <= 1
            ]

            # Error Handling... There's a case that rank1 tensors are not exist.
            # Most cases of NLP tasks have more dimension than rank1
            if len(rank1_tensors) == 0:
                process_rank1_tensors = False
            else:
                process_rank1_tensors = True

            high_rank_tensors = [
                (tensor, out, mem)
                for tensor, out, mem in zip(grad_in, grad_out, memory_out)
                if tensor.ndimension() > 1
            ]

            # [TODO] Change order of all reduce rank1 tensors
            # [For Rank 1] Start Communicating Rank 1 Tensors
            # Maybe due to there's no rank1 tensors?
            if process_rank1_tensors:
                rank1_tensor_list = tb.TensorBuffer([tensor for (tensor, _, _) in rank1_tensors], group=self.group)
                rank1_handler = rank1_tensor_list.all_reduce(async_op=True)

            for tensor, out, mem in high_rank_tensors:
                # convert grad(M) into 2d tensor
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                u, s, v = torch.svd_lowrank(matrix, q=rank, niter=1, M=None)
                n_data_parallel_workers = self.n_workers
                if n_data_parallel_workers > 1:
                    u /= n_data_parallel_workers
                    v /= n_data_parallel_workers
                    s /= n_data_parallel_workers
                    h1 = torch.distributed.all_reduce(u, group=self.group, async_op=True)
                    h2 = torch.distributed.all_reduce(v, group=self.group, async_op=True)
                    h3 = torch.distributed.all_reduce(s, group=self.group, async_op=True)
                    h1.wait()
                    h2.wait()
                    h3.wait()
                out.data[:] = torch.einsum("in, n, jn -> ij", u, s, v)
                if self.use_error_feedback:
                    mem.data[:] = tensor - out
            
            if process_rank1_tensors:
                rank1_handler.wait()
                rank1_tensor_list.buffer /= self.n_workers
                rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])     
        
        if self.current_iter < self.start_iter:
            # track current iteration
            self.current_iter += 1
            return False
        else:
            # track current iteration
            self.current_iter += 1
            return True

"""
[Finished] Change PowerSGD Reducer for MemoryBuffer Type (contiguous)
[TODO] Error-Feedback should be merged into this class (finished - debugging)
[TODO] warm-up period should be implemented into this class (finished - debugging)
"""
class PowerSGDReducer(Reducer):
    def __init__(self, random_seed, device, group, group_num, n_power_iterations=0, reuse_query=True,\
                 rank=4, start_iter=10, use_error_feedback=False, fp16=False):
        super().__init__(random_seed, device, group, group_num)
        # check if power iteration == 0 or not
        assert n_power_iterations == 0
        # compression_rank
        self.rank = rank
        # matrix P and Q
        self.p_memory = None
        self.q_memory = None
        # reuse_query => warm-start (in PowerSGD paper)
        # in most cases it is essential to be True.
        self.reuse_query = reuse_query
        # warm-up period for 10% training iteration (!!! important !!!)
        self.start_iter = int(start_iter)
        # track current iteration
        self.current_iter = 0
        # EF SGD enabled?
        self.use_error_feedback = use_error_feedback
        # support fp16?
        self.fp16 = fp16
        self.memories = None
        if dist.get_rank() == 0:
            self._init_printer()
        self.args = get_args()

    def _init_printer(self):
        print('===== PowerSGD Reducer =====')
        print(' >> rank: ', self.rank)
        print(' >> warm_start: ', self.reuse_query)
        print(' >> start_iter: ', self.start_iter)
        print(' >> EF on: ', self.use_error_feedback)
        print('============================')

    def _set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize needs to be done
        # But almost not needed... randn make almost perfect
        orthogonalize(vector)

    # [TODO] should be changed into MemoryBuffer format
    # arguments: parameters(module.parameters()) / grad_in(grad buffers) / grad_out(reducer buffer) / memory_out(EF memory)
    # EF memeory is merged into this class
    def reduce(self, module, grad_in_buffers, grad_out_buffers, grad_start_idx):
        """
        grad_in, grad_out, memory_out : dictionary of params grads
        return total communicated
        """
        if self.current_iter < self.start_iter:
            for _, buffer_ in grad_in_buffers.items():
                if self.args.emb_comm_opt:
                    non_emb_buffer_ = buffer_.get_flat_to_end(grad_start_idx)
                    non_emb_buffer_ /= self.n_workers
                    all_reduce(non_emb_buffer_, group=self.group)
                else:
                    buffer_.data /= self.n_workers
                    all_reduce(buffer_.data, group=self.group)
            if self.current_iter % 100 == 0 and dist.get_rank() == 0:
                print(' >> Still in Warm-up Period... ', self.current_iter)
        else:
            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return param.dtype

            # Collect Views of all tensors for each layer
            grad_in = []
            grad_out = []
            memory_out = []

            type_num_elements = {}
            for param in module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param.grad if param.grad is not None else param.main_grad)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                                + param.data.nelement()

            # We'll use error feedback but uninitialized
            # if self.use_error_feedback and self.memories == None:
            if self.memories == None:
                self.memories = {}
                # if dist.get_rank() == 0:
                #     print(' >> EF Memory initialized')
                for dtype, num_elements in type_num_elements.items():
                    self.memories[dtype] = MemoryBuffer(num_elements, dtype)
                    # if dist.get_rank() == 0:
                    #     print(' >> Dtype: ', dtype, ' / # elements: ', num_elements)

            # # add EF error for each 'DataType'
            if self.use_error_feedback:
                # if dist.get_rank() == 0 and self.current_iter == 0:
                #     print(' >> EF update into input buffer')
                # copied_elements = 0
                for (_, buffer_), (_, e_) in zip(grad_in_buffers.items(), self.memories.items()):
                    # buffer_.data.nan_to_num(nan=1e-8)
                    # e_.data.nan_to_num_(nan=1e-4)
                    # print(e_.data[:100])
                    # buffer_.data += e_.data
                    if self.args.emb_comm_opt:
                        add_error_feedback(buffer_.get_flat_to_end(grad_start_idx), \
                                            e_.get_flat_to_end(grad_start_idx))
                    else:
                        add_error_feedback(buffer_.data, e_.data)
                    # copied_elements += e_.data.nelement()
                # if dist.get_rank() == 0 and self.current_iter == 0:
                #     print(' >> EF updated total ', copied_elements, ' elements')


            # if dist.get_rank() == 0 and self.current_iter == 0:
            #     print(' >> Tensor pointer gathering...')

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for name, param in module.named_parameters():
                if self.args.emb_comm_opt and 'word_embeddings' in name:
                        # embeddings are in early in memories... so it's ok to do like this
                        continue
                if param.requires_grad:
                    dtype = _get_buffer_type(param.grad if param.grad is not None else param.main_grad)
                    type_num_elements[dtype] -= param.data.nelement()
                    # So param.main_grad is view of grad_buffers
                    grad_in.append(grad_in_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype]))
                    grad_out.append(grad_out_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype]))
                    # if self.use_error_feedback:
                    #     memory_out.append(self.memories[dtype].get(
                    #         param.data.shape, type_num_elements[dtype]))
                    # [debugging] error on non error feedback case
                    memory_out.append(self.memories[dtype].get(
                        param.data.shape, type_num_elements[dtype]))
                

            # [For Rank 1] It's out of Algorithms!!!!
            # rank1 tensors will be reduced un-compressed
            # and rank > 1 tensors should be compressed and reduced
            rank1_tensors = [
                (tensor, out, mem)
                for tensor, out, mem in zip(grad_in, grad_out, memory_out)
                if tensor.ndimension() <= 1
            ]
            # Error Handling... There's a case that rank1 tensors are not exist.
            # Most cases of NLP tasks have more dimension than rank1
            if len(rank1_tensors) == 0:
                process_rank1_tensors = False
            else:
                process_rank1_tensors = True

            high_rank_tensors = [
                (tensor, out, mem)
                for tensor, out, mem in zip(grad_in, grad_out, memory_out)
                if tensor.ndimension() > 1
            ]

            # build rank-k approx of every tensor
            # Approx equation
            # M = PQ^T
            # allocate consequtive mem for P's and Q's

            mem_uninitialized = self.p_memory is None

            # Step 1. Calc Matrix Size and Allocate Memory
            p_total_size = 0
            q_total_size = 0
            for tensor, _, _ in high_rank_tensors:
                # convert grad(M) into 2d tensor
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m, self.rank)
                p_total_size += n*rank
                q_total_size += m*rank
            # [Important] Initialization on Device !!!
            if self.p_memory == None: # not initialized
                self.p_memory = torch.empty(p_total_size, device=self.device, dtype=torch.float)
                self.q_memory = torch.empty(q_total_size, device=self.device, dtype=torch.float)
            # for easier implementation, gather pointers
            p_ptrs = []
            q_ptrs = []
            p_idx = 0
            q_idx = 0
            for tensor, _, _ in high_rank_tensors:
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                rank = min(n, m , self.rank)
                # torch.tensor.view returns pointer
                p_ptrs.append(self.p_memory[p_idx : p_idx + n*rank].view(n, rank))
                q_ptrs.append(self.q_memory[q_idx : q_idx + m*rank].view(m, rank))
                p_idx += n * rank
                q_idx += m * rank

            # Step 2. Prepare Q if not initailized
            for (tensor, _, _), q, p in zip(high_rank_tensors, q_ptrs, p_ptrs):
                matrix = tensor.view(tensor.shape[0], -1)
                n, m = matrix.shape
                if self.reuse_query and not mem_uninitialized:
                    # if u wanna reuse and already init
                    # use prev_Q
                    # do not need orthogonalize if properly _set_random...ed!
                    # megatron-lm need nan to zero
                    orthogonalize(q)
                    # q.nan_to_num_(nan=1e-4)
                    pass
                else:
                    self._set_random(q)
            
            """
            PowerSGD
            Algorithm 1: Rank-r PowerSGD Compression
            
            All Compression/Decompression is done in Reducer
            """

            # Step 3. (Algo 1: line 3) P <- MQ (Compute P)
            for (tensor, _, _), q, p in zip(high_rank_tensors, q_ptrs, p_ptrs):
                matrix = tensor.view(tensor.shape[0], -1)
                if self.fp16:
                    torch.matmul(matrix.float(), q, out=p)
                else:
                    torch.matmul(matrix, q, out=p)
                # p.nan_to_num_(nan=1e-4)
            
            # if dist.get_rank() == 0 and self.current_iter == 0:
            # print(self.q_memory.data[:100])
            # print(self.p_memory.data[:1000])

            # Step 4. (Algo 1: line 4) ALL_REDUCE_MEAN(P)
            all_reduce(self.p_memory, group=self.group)

            # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
            #     print(' > Compressed P Matrix: ', n_bits(self.p_memory), 'bits')

            # it's different from original PowerSGD code...
            # if there's another degradation in accurcy
            # uncomment this line for accuracy regain
            # self.p_memory.data[:] /= self.n_workers

            # [For Rank 1] Start Communicating Rank 1 Tensors
            # Maybe due to there's no rank1 tensors?
            if process_rank1_tensors:
                rank1_tensor_list = tb.TensorBuffer([tensor for (tensor, _, _) in rank1_tensors], group=self.group)
                rank1_handler = rank1_tensor_list.all_reduce(async_op=True)

            # Step 5. (Algo 1: line 5) P_hat <- ORTHOGONALIZE(P)
            for p in p_ptrs:
                orthogonalize(p)
                # p.nan_to_num_(nan=1e-4)

            # Step 6. (Algo 1: line 6) Q <- M_T P_hat
            for p, q, (tensor, _, _) in zip(p_ptrs, q_ptrs, high_rank_tensors):
                matrix = tensor.view(tensor.shape[0], -1)
                if self.fp16:
                    torch.matmul(matrix.t().float(), p, out=q)
                else:
                    torch.matmul(matrix.t(), p, out=q)
                # q.nan_to_num_(nan=1e-4)
            
            # Step 7. (Algo 1: line 7) ALL_REDUCE_MEAN(Q)
            all_reduce(self.q_memory, group=self.group)
            # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
            #     print(' > Compressed Q Matrix: ', n_bits(self.q_memory), 'bits')
            self.q_memory.data /= self.n_workers

            """
            PowerSGD
            Algorithm 2: Distributed Error-feedback SGD with Momentum
            Only Local Error is return by Reducer!
            Main Algorithm is implemented in Main Process
            """
            out_n_bits = 0
            # Step 8. (Algo 1: line 11) Decompress
            for p, q, (tensor, out, mem) in zip(p_ptrs, q_ptrs, high_rank_tensors):
                # einsum representation
                # out.data[:] = torch.einsum("nr, mr -> nm", (p, q)).view(*tensor.shape)
                if self.fp16:
                    with autocast():
                        out.data[:] = torch.mm(p, q.t())
                        # torch.matmul(p, q.t(), out=out.data[:])
                else:
                    torch.matmul(p, q.t(), out=out.data[:])
                out_n_bits += n_bits(out.data[:])
                # Step 9. (Algo 2: line 9) Memorize Local Errors
                if self.use_error_feedback:
                    update_error_feedback(mem.data[:], tensor, out)
                    # mem.data[:] = tensor - out
            # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
            #     print(' > Original Matrix: ', out_n_bits, 'bits')
            #     if (n_bits(self.p_memory)+n_bits(self.q_memory)) != 0:
            #         print(' > Compression Ratio: ', \
            #                 out_n_bits/(n_bits(self.p_memory)+n_bits(self.q_memory)))

            # [For Rank 1] Wait for Reducing
            if process_rank1_tensors:
                rank1_handler.wait()
                rank1_tensor_list.buffer /= self.n_workers
                rank1_tensor_list.unpack([out for (_, out, _) in rank1_tensors])     
        
        if self.current_iter < self.start_iter:
            # track current iteration
            self.current_iter += 1
            return False
        else:
            # track current iteration
            self.current_iter += 1
            return True


"""
[TODO] Embedding Gradient All-Reduce
[TODO] Should be changed into proper format and arguemt type
"""
class EmbPowerSGDReducer(Reducer):
    # Do we really need group num for embedding powersgd reducer??
    # [TODO] Figure out the logic and remove/keep group_num argument
    # Embedding Reducer do not need group_num !
    # Just sum them
    def __init__(self, random_seed, device, group, group_num, n_power_iterations=0, reuse_query=True,\
                 rank=4, start_iter=10, use_error_feedback=False, fp16=False):
        assert group_num == 1 # group_num must be 1 in Embedding All-Reduce
        super().__init__(random_seed, device, group, group_num)
        # check if power iteration == 0 or not
        assert n_power_iterations == 0
        # compression_rank
        self.rank = rank
        # matrix P and Q
        self.p_memory = None
        self.q_memory = None
        # reuse_query => warm-start (in PowerSGD paper)
        # in most cases it is essential to be True.
        self.reuse_query = reuse_query
        # warm-up period for 10% training iteration (!!! important !!!)
        self.start_iter = int(start_iter)
        # track current iteration
        self.current_iter = 0
        # EF SGD enabled?
        self.use_error_feedback = use_error_feedback
        # support fp16?
        self.fp16 = fp16
        self.memories = None
        if dist.get_rank() == 0:
            self._init_printer()

    def _init_printer(self):
        print('===== Embedding PowerSGD Reducer =====')
        print(' >> rank: ', self.rank)
        print(' >> warm_start: ', self.reuse_query)
        print(' >> start_iter: ', self.start_iter)
        print(' >> EF on: ', self.use_error_feedback)
        print('============================')

    def _set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize needs to be done
        # But almost not needed... randn make almost perfect
        orthogonalize(vector)

    # [TODO] should be changed into MemoryBuffer format
    # arguments: parameters(module.parameters()) / grad_in(grad buffers) / grad_out(reducer buffer) / memory_out(EF memory)
    # EF memeory is merged into this class
    def reduce(self, grad_in):
        """
        grad_in: embedding gradient compression only need grad_in
        """
        if self.current_iter < self.start_iter:
            # no averaging for embedding gradient
            all_reduce(grad_in, group=self.group)
            if self.current_iter % 100 == 0 and dist.get_rank() == 0:
                print(' >> Still in Warm-up Period... ', self.current_iter)
        else:
            """
            [Embedding PowerSGD]
            : Grad_in tensor is already 2dim (simple type)
            """
            # We'll use error feedback but unintialized
            if self.use_error_feedback and self.memories == None:
                self.memories = torch.zeros_like(grad_in)
                if dist.get_rank() == 0:
                    print(' >> EF Memory initialized')
                    print(' >> Dtype: ', self.memories.dtype, ' / # elements: ', self.memories.nelement())
            
            # add Error Feedback
            if self.use_error_feedback:
                grad_in += self.memories

            # build rank-k approx of every tensor (in this case 2d grad matrix)
            # approx eq =>
            # M = PQ^T
            # allocate consequtive mem for P's and Q's

            mem_uninitialized = self.p_memory is None

            # Step 1. Calc Matrix Size and Allocate Memory
            p_total_size = 0
            q_total_size = 0
            n, m = grad_in.shape
            rank = min(n, m, self.rank)
            p_total_size += n * rank
            q_total_size += m * rank

            # initialize on device
            if self.p_memory == None:
                self.p_memory = torch.empty(p_total_size, device=self.device, dtype=torch.float)
                self.q_memory = torch.empty(q_total_size, device=self.device, dtype=torch.float)

            # for easy implementation, gather pointers
            p_ptr = self.p_memory.view(n, rank)
            q_ptr = self.q_memory.view(m, rank)

            # Step 2. prepare q if not initialized
            if self.reuse_query and not mem_uninitialized:
                # if u wanna reuse and already init
                # use prev_Q
                # do not need orthogonalize if properly _set_randomed
                orthogonalize(q_ptr)
            else:
                self._set_random(q_ptr)

            """
            PowerSGD
            Algorithm 1: Rank-r PowerSGD Compression
            
            All Compression/Decompression is done in Reducer
            """

            # Step 3. (Algo 1: line 3) P <- MQ (Compute P)
            if self.fp16:
                torch.matmul(grad_in.float(), q_ptr, out=p_ptr)
            else:
                torch.matmul(grad_in, q_ptr, out=p_ptr)
            
            # Step 4. (Algo 1: line 4) ALL_REDUCE_MEAN(P)
            all_reduce(self.p_memory, group=self.group)

            # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
            #     print(' > Compressed P Matrix: ', n_bits(self.p_memory), 'bits')

            # it's different from original PowerSGD code...
            # if there's another degradation in accurcy
            # uncomment this line for accuracy regain
            # self.p_memory.data[:] /= self.n_workers

            # Step 5. (Algo 1: line 5) P_hat <- ORTHOGONALIZE(P)
            orthogonalize(p_ptr)

            # Step 6. (Algo 1: line 6) Q <- M_T P_hat
            if self.fp16:
                torch.matmul(grad_in.t().float(), p_ptr, out=q_ptr)
            else:
                torch.matmul(grad_in.t(), p_ptr, out=q_ptr)
            
            # Step 7. (Algo 1: line 7) ALL_REDUCE_MEAN(Q)
            all_reduce(self.q_memory, group=self.group)
            # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
            #     print(' > Compressed Q Matrix: ', n_bits(self.q_memory), 'bits')

            # no need for averaging !
            # self.q_memory.data /= self.n_workers

            """
            PowerSGD
            Algorithm 2: Distributed Error-feedback SGD with Momentum
            Only Local Error is return by Reducer!
            Main Algorithm is implemented in Main Process
            """
            # out_n_bits = 0
            # Step 8. (Algo 1: line 11) Decompress
            # make temp grad space
            grad_out = torch.zeros_like(grad_in)
            if self.fp16:
                with autocast():
                    grad_out.data[:] = torch.mm(p_ptr, q_ptr.t())
                    # torch.matmul(p, q.t(), out=out.data[:])
            else:
                torch.matmul(p_ptr, q_ptr.t(), out=grad_out.data[:])
            # out_n_bits += n_bits(grad_out.data[:])

            # Step 9. (Algo 2: line 9) Memorize Local Errors
            if self.use_error_feedback:
                self.memories.data[:] = grad_in - grad_out

            # copy to grad_in
            grad_in.data[:] = grad_out.data[:]

            # remove temp grad space
            del grad_out
            # torch.cuda.empty_cache()

            # if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
            #     print(' > Original Matrix: ', out_n_bits, 'bits')
            #     if (n_bits(self.p_memory)+n_bits(self.q_memory)) != 0:
            #         print(' > Compression Ratio: ', \
            #                 out_n_bits/(n_bits(self.p_memory)+n_bits(self.q_memory)))
        
        if self.current_iter < self.start_iter:
            # track current iteration
            self.current_iter += 1
        else:
            # track current iteration
            self.current_iter += 1

# PowerSGD Reducer for Model Parallel
class MPPowerSGDReducer(Reducer):
    def __init__(self, random_seed, device, group, group_num, n_power_iterations=0, reuse_query=True,\
                 rank=4, start_iter=10, use_error_feedback=True, fp16=False):
        super().__init__(random_seed, device, group, group_num)
        # check if power iteration == 0 or not
        assert n_power_iterations == 0
        # compression_rank
        self.rank = rank
        # matrix P and Q
        self.p_memory = None
        self.q_memory = None
        # reuse_query => warm-start (in PowerSGD paper)
        # in most cases it is essential to be True.
        self.reuse_query = reuse_query
        # warm-up period for 10% training iteration (!!! important !!!)
        self.start_iter = int(start_iter)
        # track current iteration
        self.current_iter = 0
        # EF SGD enabled?
        self.use_error_feedback = use_error_feedback
        # support fp16?
        self.fp16 = fp16
        self.memories = None
        # if dist.get_rank() == 0:
        #     self._init_printer()

    def _init_printer(self):
        print('===== MP PowerSGD Reducer =====')
        print(' >> rank: ', self.rank)
        print(' >> warm_start: ', self.reuse_query)
        print(' >> start_iter: ', self.start_iter)
        print(' >> EF on: ', self.use_error_feedback)
        print('===============================')

    def _set_random(self, vector):
        torch.manual_seed(self.rng.randint(1_000_000_000))
        vector.data[:] = torch.randn(*vector.shape, device=self.device)
        # orthogonalize needs to be done
        # But almost not needed... randn make almost perfect
        orthogonalize(vector)

    # [TODO] should be changed into MemoryBuffer format
    # arguments: parameters(module.parameters()) / grad_in(grad buffers) / grad_out(reducer buffer) / memory_out(EF memory)
    # EF memeory is merged into this class
    def reduce(self, grad_in):
        """
        grad_in, grad_out, memory_out : dictionary of params grads
        return total communicated
        """

        if self.current_iter < self.start_iter:
            # no averaging for model parallel
            all_reduce(grad_in, group=self.group)
            if self.current_iter % 100 == 0 and dist.get_rank() == 0:
                print(' >> Still in Warm-up Period... ', self.current_iter)
        else:
            """
            [MP PowerSGD]
            : Grad_in tensor is already 2dim (no multi layer type!)
            """
            # We'll use error feedback but uninitialized
            if self.use_error_feedback and self.memories == None:
                self.memories = torch.zeros_like(grad_in)
                if dist.get_rank() == 0:
                    print(' >> EF Memory initialized')
                    print(' >> Dtype: ', self.memories.dtype, ' / # elements: ', self.memories.nelement())

            # add EF
            if self.use_error_feedback:
                if dist.get_rank() == 0 and self.current_iter == 0:
                    print(' >> EF update into input buffer')
                grad_in += self.memories
                if dist.get_rank() == 0 and self.current_iter == 0:
                    print(' >> EF updated total ', self.memories.nelement(), ' elements')        

            # build rank-k approx of every tensor
            # Approx equation
            # M = PQ^T
            # allocate consequtive mem for P's and Q's

            mem_uninitialized = self.p_memory is None

            # Step 1. Calc Matrix Size and Allocate Memory
            p_total_size = 0
            q_total_size = 0
            n, m = grad_in.shape # already 2d matrix format
            rank = min(n, m, self.rank)
            p_total_size += n*rank
            q_total_size += m*rank

            # [Important] Initialization on Device !!!
            if self.p_memory == None: # not initialized
                self.p_memory = torch.empty(p_total_size, device=self.device, dtype=torch.float)
                self.q_memory = torch.empty(q_total_size, device=self.device, dtype=torch.float)

            # for easier implementation, gather pointers
            p_ptr = self.p_memory.view(n, rank)
            q_ptr = self.q_memory.view(m, rank)


            # Step 2. Prepare Q if not initailized
            if self.reuse_query and not mem_uninitialized:
                # if u wanna reuse and already init
                # use prev_Q
                # do not need orthogonalize if properly _set_random...ed!
                orthogonalize(q_ptr)
            else:
                self._set_random(q_ptr)
            
            """
            PowerSGD
            Algorithm 1: Rank-r PowerSGD Compression
            
            All Compression/Decompression is done in Reducer
            """

            # Step 3. (Algo 1: line 3) P <- MQ (Compute P)
            if self.fp16:
                torch.matmul(grad_in.float(), q_ptr, out=p_ptr)
            else:
                torch.matmul(grad_in, q_ptr, out=p_ptr)
            
            # Step 4. (Algo 1: line 4) ALL_REDUCE_MEAN(P)
            all_reduce(self.p_memory, group=self.group)

            if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
                print(' > Compressed P Matrix: ', n_bits(self.p_memory), 'bits')

            # it's different from original PowerSGD code...
            # if there's another degradation in accurcy
            # uncomment this line for accuracy regain
            # self.p_memory.data[:] /= self.n_workers

            # Step 5. (Algo 1: line 5) P_hat <- ORTHOGONALIZE(P)
            orthogonalize(p_ptr)

            # Step 6. (Algo 1: line 6) Q <- M_T P_hat
            if self.fp16:
                torch.matmul(grad_in.t().float(), p_ptr, out=q_ptr)
            else:
                torch.matmul(grad_in.t(), p_ptr, out=q_ptr)
            
            # Step 7. (Algo 1: line 7) ALL_REDUCE_MEAN(Q)
            handle = all_reduce(self.q_memory, group=self.group)
            if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
                print(' > Compressed Q Matrix: ', n_bits(self.q_memory), 'bits')

            # no need for averaging !
            # self.q_memory.data /= self.n_workers

            """
            PowerSGD
            Algorithm 2: Distributed Error-feedback SGD with Momentum
            Only Local Error is return by Reducer!
            Main Algorithm is implemented in Main Process
            """
            out_n_bits = 0
            # Step 8. (Algo 1: line 11) Decompress
            # make temp grad space
            grad_out = torch.zeros_like(grad_in)
            if self.fp16:
                with autocast():
                    grad_out.data[:] = torch.mm(p_ptr, q_ptr.t())
                    # torch.matmul(p, q.t(), out=out.data[:])
            else:
                torch.matmul(p_ptr, q_ptr.t(), out=grad_out.data[:])
            out_n_bits += n_bits(grad_out.data[:])

            # Step 9. (Algo 2: line 9) Memorize Local Errors
            if self.use_error_feedback:
                self.memories.data[:] = grad_in - grad_out

            # copy to grad_in
            grad_in.data.copy_(grad_out)

            # remove temp grad space
            del grad_out
            torch.cuda.empty_cache()

            if self.current_iter % 1000 == 0 and dist.get_rank() == 0:
                print(' > Original Matrix: ', out_n_bits, 'bits')
                if (n_bits(self.p_memory)+n_bits(self.q_memory)) != 0:
                    print(' > Compression Ratio: ', \
                            out_n_bits/(n_bits(self.p_memory)+n_bits(self.q_memory)))

        if self.current_iter < self.start_iter:
            # track current iteration
            self.current_iter += 1
            return False
        else:
            # track current iteration
            self.current_iter += 1
            return True


class StableTopKReducerWithRangeBucketsEFCorrectionResIsGrad(Reducer):
    '''
    **This is the version w/ error feedback added.**
    **Simply applying EF will lead to divergence.**
    Now will try to use momentum to regulate it.
    1. All_reduce on gradients.
    2. Select topk indices
    3. Make topk indices as centers, then select the gradients on the left and right by 500-range.
    4. Gradient correction.
    '''
    def __init__(self,
                 random_seed,
                 device,
                 group,
                 group_num,
                 beta1,
                 beta2,
                 epsilon,
                 density,
                 stable_topk_interval,
                 stable_topk_threshold,
                 stable_topk_range,
                 stable_topk_warmup_method="Dense"):
        
        # params for comm setup
        super().__init__(random_seed, device, group, group_num)
        
        # compression density
        self.density = density
        # how often will resampling happen
        self.stable_topk_interval = stable_topk_interval
        # how long will the warmup stage last
        self.stable_topk_threshold = stable_topk_threshold
        # which warmup method will be used ([TODO] used to support other methods, such as gtopk)
        assert stable_topk_warmup_method == "Dense", "Currently only support Dense Allreduce as the warmup method."
        self.stable_topk_warmup_method = stable_topk_warmup_method
        # How many buckets will the topk gradients will be divided into
        self.stable_topk_range = stable_topk_range
        
        ## for storing layer dim info
        # self.grad_shapes = {}
        # self.grad_sizes = {}
        
        self.args = get_args()
        
        # # mask for extracting the residuals
        # self.zero_conditions = {}
        
        # # Residuals (i.e. non-topk values)
        # self.residuals = {}
        
        # # dict for transmitting data in sparse_allreduce
        # self.storage = {}
        
        # Store buckets that are merged acorss all workers in the same group
        self.buckets_indices = {}
        self.tensor_buckets_start_end_indices = {}
        self.num_element_per_chunk={}
        self.selected_num_element_per_chunk={}
        
        # A List[int] that contains the numel per layer
        self.num_element_per_layer = {}
        
        # Store intermediate values in adam (i.e. m_t and v_t)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.tensor_corrected = {}

        # range formed with topk as centers and self.stable_topk_range as radius
        self.intervals = {}
        self.indices = {}
        self.do_resample = True

        # analyze the importance of gradients
        self.prev_grad_topk_indices = {}
        
        
        # For amp_C fused_adam function
        if multi_tensor_applier.available:
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_adam = amp_C.multi_tensor_adam
        
        # print configurations into log
        if dist.get_rank() == 0:
            self._init_printer()
          
            
    def _init_printer(self):
        print('===== StableTopKReducerWithRangeBucketsEFCorrectionResIsGrad Reducer =====')
        print(' >> density: ', self.density)
        print(' >> resampling interval: ', self.stable_topk_interval)
        print(' >> number of warmup iterations: ', self.stable_topk_threshold)
        print(' >> warmup method: ', self.stable_topk_warmup_method)
        print(' >> range radius: ', self.stable_topk_range)
        print('==========================================================================')
        
        
    def _dense_allreduce(self, grad_in_buffers, grad_start_idx):
        for dtype, buffer_ in grad_in_buffers.items():
            # fp16 causes data overflow, which are nan/inf in the buffer_
            if self.args.emb_comm_opt:
                non_emb_buffer_ = buffer_.get_flat_to_end(grad_start_idx)
                non_emb_buffer_ /= self.n_workers
                all_reduce(non_emb_buffer_, op=dist.ReduceOp.SUM, group=self.group)
            else:
                buffer_.data /= self.n_workers
                all_reduce(buffer_.data, op=dist.ReduceOp.SUM, group=self.group)
    
    
    @abstractmethod
    def param_is_not_shared(param):
        return not hasattr(param, 'shared') or not param.shared


    @torch.no_grad()
    def clip_grad_norm_fp32(self, parameters, max_norm, do_clip=True, norm_type=2):
        """Clips gradient norm of an iterable of parameters whose gradients
        are in fp32.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        # Filter parameters based on:
        #   - grad should not be none
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        grads = []
        grads_for_norm = []
        for param in parameters:
            grad_not_none = param.main_grad is not None
            is_not_shared = StableTopKReducerWithValBucketsEFCorrection.param_is_not_shared(param)
            is_not_tp_duplicate = mpu.param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none:
                grad = param.main_grad.detach()
            if grad_not_none:
                # Make sure the params are in fp32
                assert param.main_grad.type() == 'torch.cuda.FloatTensor'
                grads.append(grad)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)
        
        # Norm parameters.
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        total_norm = 0.0

        # Calculate norm.
        if norm_type == inf:
            total_norm = max(grad.abs().max() for grad in grads_for_norm)
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
            # # Take max across all model-parallel GPUs.
            all_reduce(total_norm_cuda,
                        op=torch.distributed.ReduceOp.MAX,
                        group=mpu.get_model_parallel_group())
            total_norm = total_norm_cuda[0].item()

        else:
            if norm_type == 2.0:
                if len(grads_for_norm) > 0:
                    dummy_overflow_buf = torch.cuda.IntTensor([0])
                    # Use apex's multi-tensor applier for efficiency reasons.
                    # Multi-tensor applier takes a function and a list of list
                    # and performs the operation on that list all in one kernel.
                    grad_norm, _ = multi_tensor_applier(
                        amp_C.multi_tensor_l2norm,
                        dummy_overflow_buf,
                        [grads_for_norm],
                        False # no per-parameter norm
                    )
                    # Since we will be summing across data parallel groups,
                    # we need the pow(norm-type).
                    total_norm = grad_norm ** norm_type
                else:
                    total_norm = torch.tensor([0.], dtype=torch.float32, device=self.device)
                
            else:
                for grad in grads_for_norm:
                    grad_norm = torch.norm(grad, norm_type)
                    total_norm += grad_norm ** norm_type

            # # Sum across all model-parallel GPUs.
            all_reduce(total_norm,
                        op=torch.distributed.ReduceOp.SUM,
                        group=mpu.get_model_parallel_group())
            total_norm = total_norm.item() ** (1.0 / norm_type)

        # Scale.
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        if clip_coeff < 1.0 and do_clip:
            dummy_overflow_buf = torch.cuda.IntTensor([0])
            multi_tensor_applier(amp_C.multi_tensor_scale,
                                dummy_overflow_buf,
                                [grads, grads],
                                clip_coeff)
        return total_norm


    # def gradient_clip(self, dtype, start_end_idx_per_layer, tensor):
    #     # for start_end_idx_cur_layer in start_end_idx_per_layer[dtype].values():    
    #     #     start_idx, end_idx = start_end_idx_cur_layer
    #     #     torch.nn.utils.clip_grad_norm_(tensor[start_idx:end_idx], self.args.clip_grad)  
    #     if dtype not in self.num_element_per_layer:
    #         self.num_element_per_layer[dtype] = []
    #         for start_end_idx_cur_layer in start_end_idx_per_layer[dtype].values():   
    #             start_idx, end_idx = start_end_idx_cur_layer
    #             self.num_element_per_layer[dtype].append(end_idx - start_idx)
    #     tensor_splitted_by_layer = torch.split(tensor, self.num_element_per_layer[dtype])
    #     # grad_norm = self.clip_grad_norm_fp32(tensor_splitted_by_layer, self.args.clip_grad*(self.n_workers)**(-1/2))
    #     grad_norm = self.clip_grad_norm_fp32(tensor_splitted_by_layer, self.args.clip_grad)
    #     tensor_clipped = torch.cat(tensor_splitted_by_layer)
    #     return tensor_clipped
    
    def get_parameters(self, module):
        params = []
        for name, param in module.named_parameters():
            if self.args.emb_comm_opt and 'word_embeddings' in name:
                continue
            if param.requires_grad:
                params.append(param)
        return params
                                            
    
    def gradient_clip(self, module):
        params = self.get_parameters(module)
        return self.clip_grad_norm_fp32(params, self.args.clip_grad)
 
  
    # def gradient_correction(self, dtype, layer_idx_, start_end_idx_per_layer, grad_start_idx, tensor):
    #     '''
    #     Perform gradient correction (like momentum correction in DGC)
    #     '''
    #     assert "optimizer_var" in self.args and "optimizer_layer_order" in self.args

    #     keys = {i:t for i, t in enumerate(self.args.optimizer_var.state.keys())}

    #     # layer_idx_set = [set() for _ in range(len(self.args.optimizer_var.param_groups))]
    #     group_dicts = [{t:i for i,t in enumerate(group['params'])} for group in self.args.optimizer_var.param_groups]
        
        
    #     for i, layer_idx in enumerate(self.args.optimizer_layer_order):
    #         if layer_idx not in start_end_idx_per_layer[dtype]: ## should skip for embedding layer if emb_comm_opt is set
    #             continue
    #         model_params = keys[i]
    #         beta1, beta2, bias_correction = None, None, None
    #         eps, weight_decay, step = None, None, None
            
    #         for group_idx, group in enumerate(self.args.optimizer_var.param_groups):
    #             if model_params in group_dicts[group_idx]:
    #                 # Every layer should be updated only once

    #                 # assert group_dicts[group_idx][model_params] not in layer_idx_set[group_idx]
    #                 # layer_idx_set[group_idx].add(group_dicts[group_idx][model_params])
                    
    #                 beta1, beta2 = group['betas']
    #                 bias_correction = 1 if group['bias_correction'] else 0
    #                 eps = group['eps']
    #                 weight_decay = group['weight_decay_reducer']
    #                 if 'step' in group:
    #                     group['step'] += 1
    #                 else:
    #                     group['step'] = 1
    #                 step = group['step']
    #                 break
    #         assert beta1 is not None and \
    #             beta2 is not None and \
    #             step is not None and \
    #             bias_correction is not None and \
    #             eps is not None and \
    #             weight_decay is not None 
                
    #         start_end_idx = start_end_idx_per_layer[dtype][layer_idx]
    #         start_idx, end_idx = start_end_idx
            
    #         if self.args.emb_comm_opt:
    #             start_idx = start_idx - grad_start_idx
    #             end_idx = end_idx - grad_start_idx
            
    #         assert model_params.numel() == end_idx - start_idx and end_idx <= tensor.numel()
    #         # if dist.get_rank() == 0:
    #         #     print("model_params.numel(): ", model_params.numel())
    #         #     print("end_idx - start_idx: ", end_idx - start_idx,'\n' )
    #         #     print("model_params.shape: ",model_params.shape)
    #         #     print("tensor[start_idx:end_idx].numel()",tensor[start_idx:end_idx].numel())
    #         #     print("tensor.numel(): ", tensor.numel())
    #         #     print("start_idx: ",start_idx)
    #         #     print("end_idx: ", end_idx)
    #         # dist.barrier()
    #         original_shape = model_params.shape
            
    #         grads_cur_layer = tensor[start_idx:end_idx].reshape(*original_shape)
            
    #         exp_avg = self.args.optimizer_var.state[keys[i]]['exp_avg']
    #         exp_avg_sq = self.args.optimizer_var.state[keys[i]]['exp_avg_sq']

    #         # # [NOTE] take out non-topk momentum 
    #         # if reducer == 'org' and topk_pos_sum != 0:
    #         #     # mask = self.zero_conditions[dtype][start_idx:end_idx].reshape(*original_shape)
    #         #     # exp_avg_nontopk = exp_avg * mask
    #         #     # exp_avg_sq_nontopk = exp_avg_sq * mask
    #         #     # exp_avg.data.mul_(1-mask)
    #         #     # exp_avg_sq.data.mul_(1-mask)
                
                
    #         #     mask = 1-self.zero_conditions[dtype][start_idx:end_idx].reshape(*original_shape)
    #         #     mask += (1-mask)/beta1**self.stable_topk_interval
    #         #     exp_avg.mul_(mask)
                
    #         #     mask = 1-self.zero_conditions[dtype][start_idx:end_idx].reshape(*original_shape)
    #         #     mask += (1-mask)/beta2**self.stable_topk_interval
    #         #     exp_avg_sq.mul_(mask)
                
    #         #     if torch.distributed.get_rank() == 0:
    #         #         print("Conpensated exp_avg and exp_avg_sq", flush=True)
            
    #         model_params_clone = keys[i].clone()

    #         g_32 = [grads_cur_layer]
    #         p_32 = [model_params_clone]
    #         m_32 = [exp_avg]
    #         v_32 = [exp_avg_sq]
            
    #         multi_tensor_applier(self.multi_tensor_adam,
    #                             self._dummy_overflow_buf,
    #                             [g_32, p_32, m_32, v_32],
    #                             1.0, # avoid using lr here, this causes grad norm to be large
    #                             beta1,
    #                             beta2,
    #                             eps,
    #                             step,
    #                             True, #adam_w_mode is default to be True
    #                             bias_correction,
    #                             weight_decay)
            
    #          # w = w - lr * grad - lr * weight_decay * w, so need to recover it
    #         tensor[start_idx:end_idx] = (model_params-p_32[0]).flatten()

    #         # # [NOTE] Add back non-topk momentum
    #         # if reducer != 'org':
    #         #     exp_avg.data.add_(exp_avg_nontopk)
    #         #     exp_avg_sq.data.add_(exp_avg_sq_nontopk)    
    
    @torch.no_grad()
    def _gradient_correction(self, dtype, layer_idx_, start_end_idx_per_layer, tensor):
        '''
        Perform gradient correction (like momentum correction in DGC)
        '''
        assert "optimizer_var" in self.args and "optimizer_layer_order" in self.args

        keys = {i:t for i, t in enumerate(self.args.optimizer_var.state.keys())}

        group_dicts = [{t:i for i,t in enumerate(group['params'])} for group in self.args.optimizer_var.param_groups]
        
        for i, layer_idx in enumerate(self.args.optimizer_layer_order):
            if self.args.emb_comm_opt and 'word_embeddings' in layer_idx_[dtype][layer_idx]:
                continue
            model_params = keys[i]
            beta1, beta2, bias_correction = None, None, None
            eps, weight_decay, step = None, None, None
            
            for group_idx, group in enumerate(self.args.optimizer_var.param_groups):
                if model_params in group_dicts[group_idx]:
                    # Every layer should be updated only once

                    # assert group_dicts[group_idx][model_params] not in layer_idx_set[group_idx]
                    # layer_idx_set[group_idx].add(group_dicts[group_idx][model_params])
                    
                    beta1, beta2 = group['betas']
                    bias_correction = 1 if group['bias_correction'] else 0
                    eps = group['eps']
                    weight_decay = group['weight_decay_reducer']
                    if 'step' in group:
                        group['step'] += 1
                    else:
                        group['step'] = 1
                    step = group['step']
                    break
            assert beta1 is not None and \
                beta2 is not None and \
                step is not None and \
                bias_correction is not None and \
                eps is not None and \
                weight_decay is not None 
                
            start_end_idx = start_end_idx_per_layer[dtype][layer_idx]
            start_idx, end_idx = start_end_idx

            
            assert model_params.numel() == end_idx - start_idx and end_idx <= tensor.numel()
 
            original_shape = model_params.shape
            
            grads_cur_layer = tensor[start_idx:end_idx].reshape(*original_shape)
            
            exp_avg = self.args.optimizer_var.state[keys[i]]['exp_avg']
            exp_avg_sq = self.args.optimizer_var.state[keys[i]]['exp_avg_sq']
            
            # remember to avoid loading momentum_buffer when resume from checkpoint
            if 'momentum_buffer' not in self.args.optimizer_var.state[keys[i]]:
                self.args.optimizer_var.state[keys[i]]['momentum_buffer'] = exp_avg

            model_params_clone = keys[i].clone()

            g_32 = [grads_cur_layer]
            p_32 = [model_params_clone]
            m_32 = [exp_avg]
            v_32 = [exp_avg_sq]
            
            multi_tensor_applier(self.multi_tensor_adam,
                                self._dummy_overflow_buf,
                                [g_32, p_32, m_32, v_32],
                                1.0, # avoid using lr here, this causes grad norm to be large
                                beta1,
                                beta2,
                                eps,
                                step,
                                True, #adam_w_mode is default to be True
                                bias_correction,
                                weight_decay)
            
             # w = w - lr * grad - lr * weight_decay * w, so need to recover it
            tensor[start_idx:end_idx] = (model_params-p_32[0]).flatten()
    
    
    @torch.no_grad()    
    def gradient_correction(self, layer_idx_, start_end_idx_per_layer, grad_in_buffers):
        for dtype, buffer_ in grad_in_buffers.items():
            all_grads = buffer_.data 
            self._gradient_correction(dtype, layer_idx_, start_end_idx_per_layer, all_grads)
                
    
    
    def resample_topk_indices(self,
                              train_step,
                              layer_idx_,
                              start_end_idx_per_layer,
                              grad_start_idx,
                              grad_in_buffers):
        with torch.no_grad():
            if train_step >= self.stable_topk_threshold:
                for dtype, buffer_ in grad_in_buffers.items():
                    if train_step % self.stable_topk_interval == 0 or self.do_resample:
                        self.do_resample = False
                        if 0 < self.density < 1:
                            if self.args.emb_comm_opt:
                                all_grads = buffer_.get_flat_to_end(grad_start_idx)
                            else:
                                all_grads = buffer_.data
                            
                            if self.stable_topk_range > 0:
                                # intervals = self._resample_topk_intervals(dtype, layer_idx_, start_end_idx_per_layer, grad_start_idx, all_grads)
                                # self.intervals[dtype] = self.merge_intervals(torch.cat(intervals).tolist())
                                # self.indices[dtype] = self.generate_indices_from_intervals(self.intervals[dtype], dtype)
                                self._resample_topk_intervals(dtype, layer_idx_, start_end_idx_per_layer, grad_start_idx, all_grads)
                            else:
                                # self.indices[dtype] = self._resample_topk_indices(dtype, layer_idx_, start_end_idx_per_layer, grad_start_idx, all_grads)
                                self._resample_topk_indices(dtype, layer_idx_, start_end_idx_per_layer, grad_start_idx, all_grads)
                            
                            # try:
                            #     assert dtype in self.zero_conditions
                            # except:
                            #     if dist.get_rank() == 0:
                            #         print("self.zero_conditions: ", self.zero_conditions)
                            #     if self.zero_conditions is None:
                            #         self.zero_conditions = torch.ones_like(tensor.view(-1), dtype=tensor.dtype, device=self.device)
                
                            
                            # # self.zero_conditions[dtype].index_fill_(0,self.indices[dtype],0)
                            # if torch.distributed.get_rank() == 0 and dtype in self.indices:
                            #     print(f"Real compression ratio: {torch.sum(self.indices[dtype])/all_grads.numel():.4f}", flush=True)
                            # dist.barrier()
                            
    
    @torch.no_grad()                
    def merge_intervals(self, intervals):
        intervals.sort(key=lambda x: x[0])
        sorted_intervals = intervals
        merged_intervals = [sorted_intervals.pop(0)]
        for sorted_interval in sorted_intervals:
            prev_end = merged_intervals[-1][1]
            cur_start = sorted_interval[0]
            cur_end = sorted_interval[1]
            # 3 cases: 
            # prev_end < cur_start, 
            if prev_end < cur_start:
                merged_intervals.append(sorted_interval)
            # prev_end == cur_start, 
            elif prev_end == cur_start:
                merged_intervals[-1][1] = cur_end
            # prev_end > cur_start, 
            else:
                # two sub cases:
                # prev_end <= cur_end,
                # prev_end > cur_end
                if prev_end <= cur_end:
                    merged_intervals[-1][1] = cur_end
        return merged_intervals


    @torch.no_grad()
    def _resample_topk_intervals(self, dtype, layer_idx_, start_end_idx_per_layer, grad_start_idx, tensor):
        # intervals = []
        for i, layer_start_end_idx in start_end_idx_per_layer[dtype].items():
            if self.args.emb_comm_opt and 'word_embeddings' in layer_idx_[dtype][i]:
                continue
            
            layer_start_idx, layer_end_idx = layer_start_end_idx
            if self.args.emb_comm_opt:
                layer_start_idx = layer_start_idx - grad_start_idx
                layer_end_idx = layer_end_idx - grad_start_idx
            
            if 'bias' in layer_idx_[dtype][i]:
                # intervals.append(torch.tensor([[layer_start_idx, layer_end_idx]], device=self.device))
                self.indices[dtype][layer_start_idx:layer_end_idx] = True
            else:
                numel_cur_layer = layer_end_idx - layer_start_idx
                ratio = self.density
                k=max(int(numel_cur_layer*ratio), 1)
                _, topk_indices = torch.topk(tensor[layer_start_idx:layer_end_idx].abs(), k=k)
                topk_indices, _ = torch.sort(topk_indices)

                left_boundary = (topk_indices - self.stable_topk_range + layer_start_idx).reshape(-1,1)
                left_boundary[left_boundary<layer_start_idx] = layer_start_idx
                right_boundary = (topk_indices + self.stable_topk_range + layer_start_idx + 1).reshape(-1,1)
                right_boundary[right_boundary>layer_end_idx] = layer_end_idx
                # intervals.append(torch.cat([left_boundary, right_boundary],dim=1))
                # self.indices[dtype][torch.cat([left_boundary, right_boundary],dim=1)] = True
                
                # for left, right in zip(left_boundary, right_boundary):
                #     self.indices[dtype][left:right] = True

                intervals = torch.cat([left_boundary, right_boundary],dim=1).tolist()
                intervals = self.merge_intervals(intervals)
                for interval in intervals:
                    self.indices[dtype][interval[0]:interval[1]] = True
                
        # return intervals
   
    
    @torch.no_grad()
    def _resample_topk_indices(self, dtype, layer_idx_, start_end_idx_per_layer, grad_start_idx, tensor):
        # indices = []
        for i, layer_start_end_idx in start_end_idx_per_layer[dtype].items():
            if self.args.emb_comm_opt and 'word_embeddings' in layer_idx_[dtype][i]:
                continue
            
            layer_start_idx, layer_end_idx = layer_start_end_idx
            
            if self.args.emb_comm_opt:
                layer_start_idx = layer_start_idx - grad_start_idx
                layer_end_idx = layer_end_idx - grad_start_idx
            
            numel_cur_layer = layer_end_idx - layer_start_idx
            if 'bias' in layer_idx_[dtype][i]:
                # indices.append(torch.arange(layer_start_idx, layer_end_idx, device=self.device))
                self.indices[dtype][layer_start_idx:layer_end_idx] = True
            else:
                ratio = self.density
                k=max(int(numel_cur_layer*ratio), 1)
                _, topk_indices = torch.topk(tensor[layer_start_idx:layer_end_idx].abs(), k=k)
                
                topk_indices.add_(layer_start_idx)
                topk_indices,_ = torch.sort(topk_indices)
                # indices.append(topk_indices)
                
                # nonzeros=self.indices[dtype].nonzero().sum()
                # self.indices[dtype][nonzeros:nonzeros+k] = topk_indices
                
                self.indices[dtype][topk_indices] = True

        # indices = torch.cat(indices).reshape(-1)
        # return indices

    @torch.no_grad()
    def check_intersection_rate(self, dtype, indices, update_prev_indices=False):
        cur_indices= set(indices.cpu().numpy())
        if dtype in self.prev_grad_topk_indices:
            intersection = self.prev_grad_topk_indices[dtype].intersection(cur_indices)
            intersection_rate = len(intersection)/len(cur_indices)
            if dist.get_rank() == 0:
                print(f"Intersection rate: {intersection_rate}")
        dist.barrier()
        
        if update_prev_indices:
            self.prev_grad_topk_indices[dtype] = cur_indices

    @torch.no_grad()
    def generate_indices_from_intervals(self,
                                        intervals,
                                        dtype):
        indices = []
        for interval in intervals:
            start, end = interval
            indices.append(torch.arange(start,end,device=self.device))
        return torch.cat(indices)
 
    @torch.no_grad()
    def get_compressed_tensor_len(self,
                                  dtype,
                                  start_end_idx_per_layer,
                                  layer_idx_):
        numel = 0
        for i, layer_start_end_idx in start_end_idx_per_layer[dtype].items():
            start_idx, end_idx = layer_start_end_idx
            numel_cur_layer = end_idx - start_idx
            if 'bias' not in layer_idx_[dtype][i]:
                numel += max(int(self.density * numel_cur_layer), 1)
            else:
                numel += numel_cur_layer
        return numel
    
    def compress_org(self, 
                     module,
                     dtype,
                     layer_idx_,
                     start_end_idx_per_layer,
                     grad_start_idx,
                     tensor):
        with torch.no_grad(): 
            
            # # Residuals here store the gradients, not momentums, nor the corrected momentums
            if 'residuals' not in self.args.optimizer_var.state:
                self.args.optimizer_var.state['residuals'] = torch.zeros_like(tensor, 
                                                                              dtype=dtype, 
                                                                              device=self.device, 
                                                                              requires_grad=False)
            elif self.args.optimizer_var.state['residuals'].get_device() != self.device:
                self.args.optimizer_var.state['residuals'] = self.args.optimizer_var.state['residuals'].to(self.device)
                

            # # As a mask for residual selection
            # if dtype not in self.zero_conditions:
            #     self.zero_conditions[dtype] = torch.ones_like(tensor,dtype=dtype, device=self.device)
            # else:
            #     self.zero_conditions[dtype].fill_(1.0)
            
            # if self.stable_topk_range == 0:
            #     self.indices[dtype] = torch.zeros(int(tensor.numel() * self.density), device=self.device, dtype=torch.int64)
            if dtype not in self.indices:
                self.indices[dtype] = torch.zeros_like(tensor, 
                                                       dtype=torch.bool, 
                                                       device=self.device, 
                                                       requires_grad=False)
            else:
                self.indices[dtype].fill_(False)
            self.do_resample = True
            
            tensor.add_(self.args.optimizer_var.state['residuals'])
            tensor.div_(self.n_workers)
            
            # Perform all_reduce to get the averaged gradients from all workers
            all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
            
            self.args.optimizer_var.state['residuals'].fill_(0.0)
            
            # if self.args.move_grad_clip_to_reducer:   
            #     self.args.grad_norm = self.gradient_clip(module)
            
            # # # tensor = self.gradient_correction(dtype, layer_idx_, start_end_idx_per_layer, tensor, reducer='stable')
            # self.gradient_correction(dtype, layer_idx_, start_end_idx_per_layer, grad_start_idx, tensor)
            
            # if 0 <= self.density < 1:
                
            #     if self.stable_topk_range > 0:
            #         intervals = self._resample_topk_intervals(dtype, layer_idx_, start_end_idx_per_layer, grad_start_idx, tensor)

            #         self.intervals[dtype] = self.merge_intervals(torch.cat(intervals).tolist())

            #         # Use intervals to generate the individual indices
            #         self.indices[dtype] = self.generate_indices_from_intervals(self.intervals[dtype], dtype)
            #     else:
            #         self.indices[dtype] = self._resample_topk_indices(dtype, layer_idx_, start_end_idx_per_layer, grad_start_idx, tensor)
                
            #     # # initialize the zero_conditions as masks for selecting residuals
            #     # # i.e. residuals=tensor*zero_conditions
            #     self.zero_conditions[dtype].fill_(1.0)
                
            #     # initialize the residuals for later accumulation
            #     self.args.optimizer_var.state['residuals'].fill_(0.0)
                
            #     self.zero_conditions[dtype].index_fill_(0,self.indices[dtype],0)

            #     if torch.distributed.get_rank() == 0 and dtype in self.indices:
            #         print(f"Real compression ratio: {len(self.indices[dtype])/tensor.numel():.4f}", flush=True)
            #     dist.barrier()
            
        
    def compress_stable(self, 
                        dtype, 
                        module,
                        layer_idx_,
                        start_end_idx_per_layer,
                        grad_start_idx,
                        tensor):
        with torch.no_grad():
            # # Make sure compress_org has been called before calling compress_stable
            # # assert dtype in self.buckets_indices and \
            # #         dtype in self.tensor_buckets_start_end_indices
            
            # assert 'residuals' in self.args.optimizer_var.state and \
            #         dtype in self.zero_conditions and \
            #         dtype in self.indices 
            assert 'residuals' in self.args.optimizer_var.state and dtype in self.indices 
            
            if self.density == 1:
                selected_tensor = tensor
            else:
                # selected_tensor = torch.index_select(tensor,0,self.indices[dtype])
                # selected_tensor = torch.masked_select(tensor, self.indices[dtype])
                selected_tensor = tensor[self.indices[dtype]]

            selected_tensor.data.div_(self.n_workers)
            
            
            # Allreduce on selected/compressed gradients
            all_reduce(selected_tensor, op=dist.ReduceOp.SUM, group=self.group)
            
            # # indices = self.resample_topk_indices(dtype, start_end_idx_per_layer, tensor)
            # # self.check_intersection_rate(dtype, indices)

            if 0 <= self.density < 1:
                # tensor[self.indices[dtype]] = selected_tensor
                tensor[self.indices[dtype]] = selected_tensor
                # self.args.optimizer_var.state['residuals'].add_(tensor * self.zero_conditions[dtype])
                # tensor.mul_(1 - self.zero_conditions[dtype])
                self.args.optimizer_var.state['residuals'].add_(tensor * ~self.indices[dtype])
                tensor.mul_(self.indices[dtype])
                
                
            # if self.args.move_grad_clip_to_reducer:
            #     # tensor = self.gradient_clip(dtype, start_end_idx_per_layer, tensor)
            #     self.args.grad_norm = self.gradient_clip(module)
            
            # # # Perform direction correction here
            # # tensor = self.gradient_correction(dtype, layer_idx_, start_end_idx_per_layer, tensor, reducer='stable')
            # self.gradient_correction(dtype, layer_idx_, start_end_idx_per_layer, grad_start_idx, tensor)
            
    
    def reconstruct_grads(self, 
                          dtype, 
                          allreduce_results, 
                          grad_in_buffers, 
                          grad_out_buffers, 
                          grad_start_idx):
        '''
        Divide the combined gradients into each layer's gradient buffer.
        Update on the gradients will be performed after self.reduce has returned
        '''
        # Normally, only one type of data in gradients buffers
        if self.args.emb_comm_opt:
            emb_grads = grad_in_buffers[dtype].data[:grad_start_idx]
            grad_out_buffers[dtype].data = torch.cat(emb_grads, allreduce_results)
        else:
            grad_out_buffers[dtype].data = allreduce_results

    
    def reduce(self, 
               train_iter,
               module, 
               layer_idx_,
               start_end_idx_per_layer,
               grad_in_buffers, 
               grad_out_buffers, 
               grad_start_idx):
        """
        grad_in_buffers:  A list of continuous memory which contains all the grads from diff layers
                          No need to worry about mapping issues.
                          They have been taken care when param.main_grads are constructed.
                          **[NOTE] grad_in_buffer is the reverse of the layer order**
                          **I.e. grad_in_buffers = L_N, L_{N-1}, ..., L_2, L_1**
        grad_out_buffers: Similar to grad_in_buffers, but will not use it until warmup stage finishes.
                          **[NOTE] grad_out_buffer = L_1, L_2, ..., L_{N-1}, L_N**
        grad_start_idx:   Starting index of the gradient, excluding embedding layer grads.
        return: Whether need to do copy back from grad_in to grad_out or not.
        """
        if train_iter < self.stable_topk_threshold:
            # Collect topk grad distributions
            self._dense_allreduce(grad_in_buffers, grad_start_idx)
            # report whether in warmup stage or not every 100 iterations
            if train_iter % 100 == 0 and dist.get_rank() == 0:
                print(' >> Still in Warm-up Period... ', train_iter)
        else:    
            # Here we assume there are more than one type of data in grad_in_buffers.
            # Even though this is usually unlikely, we need to keep the integrity of the code.
            # [Finished] for dtype, buffer_ in grad_in_buffers.items():
            # all_grads = self.combine_all_grads(grad_in_buffers)
            
            for dtype, buffer_ in grad_in_buffers.items():
                if self.args.emb_comm_opt:
                    all_grads = buffer_.get_flat_to_end(grad_start_idx)
                    # if dist.get_rank() == 0:
                    #     print(f"all_grads/grad_in_buffers: {100*all_grads.numel()/buffer_.data.numel()}%")
                else:
                    all_grads = buffer_.data
                # Recompute every self.stable_topk_interval intervals or we have no indecies 
                if train_iter % self.stable_topk_interval == 0 or dtype not in self.indices:
                    # compress_org performs operation directly on all_grads
                    # so processed_tensor and all_grads point to the same underlying tensor
                    self.compress_org(module,
                                        dtype, 
                                        layer_idx_,
                                        start_end_idx_per_layer,
                                        grad_start_idx,
                                        all_grads)
                else:
                    # perform update locally
                    self.compress_stable(dtype,
                                        module,
                                        layer_idx_,
                                        start_end_idx_per_layer,
                                        grad_start_idx,
                                        all_grads)
            
            # self.reconstruct_grads(dtype,
            #                        processed_tensor,
            #                        grad_in_buffers,
            #                        grad_out_buffers,
            #                        grad_start_idx)
        
        # if warmup ends, need to use reduced_buffer to update grad_buffer
        # no need to do so during the warmup stage, becuase grad_in_buffers points to models' grad tensor
        # return train_iter >= self.stable_topk_threshold
        return False
        

class StableTopKReducerWithRangeBucketsEFCorrectionResIsGradPerLayer(Reducer):
    '''
    **This is the version w/ error feedback added.**
    **Simply applying EF will lead to divergence.**
    **Note that due to memory performance, this reducer does not support `stable_topk_range` parameter.**
    **Having a deterministic density helps prevent memory alloc and de-alloc, and avoids the need to torch.concat, which is super heavy.**
    Now will try to use momentum to regulate it.
    1. All_reduce on gradients.
    2. Select topk indices
    3. Make topk indices as centers, then select the gradients on the left and right by 500-range.
    4. Gradient correction.
    '''
    def __init__(self,
                 random_seed,
                 device,
                 group,
                 group_num,
                 density,
                 stable_topk_interval,
                 stable_topk_threshold,
                 stable_topk_warmup_method="Dense"):
        
        # params for comm setup
        super().__init__(random_seed, device, group, group_num)
        
        # compression density
        self.density = density
        # how often will resampling happen
        self.stable_topk_interval = stable_topk_interval
        # how long will the warmup stage last
        self.stable_topk_threshold = stable_topk_threshold
        # which warmup method will be used ([TODO] used to support other methods, such as gtopk)
        assert stable_topk_warmup_method == "Dense", "Currently only support Dense Allreduce as the warmup method."
        self.stable_topk_warmup_method = stable_topk_warmup_method
        
        ## for storing layer dim info
        # self.grad_shapes = {}
        # self.grad_sizes = {}
        
        self.args = get_args()
        
        # # Residuals (i.e. non-topk values)
        self.residuals = {}
        
        self.compressed_grad = {}
        
        
        # Store buckets that are merged acorss all workers in the same group
        self.buckets_indices = {}
        self.tensor_buckets_start_end_indices = {}
        self.num_element_per_chunk={}
        self.selected_num_element_per_chunk={}
        
        # A List[int] that contains the numel per layer
        self.num_element_per_layer = {}
        
        # range formed with topk as centers and self.stable_topk_range as radius
        self.topk_indices = {}
        self.nontopk_indices = {}
        self.do_resample = True

        # analyze the importance of gradients
        self.prev_grad_topk_indices = {}
        
        # For amp_C fused_adam function
        if multi_tensor_applier.available:
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_adam = amp_C.multi_tensor_adam
        
        # print configurations into log
        if dist.get_rank() == 0:
            self._init_printer()
          
            
    def _init_printer(self):
        print('===== StableTopKReducerWithRangeBucketsEFCorrectionResIsGradPerLayer Reducer =====')
        print(' >> density: ', self.density)
        print(' >> resampling interval: ', self.stable_topk_interval)
        print(' >> number of warmup iterations: ', self.stable_topk_threshold)
        print(' >> warmup method: ', self.stable_topk_warmup_method)
        print('==========================================================================')
        
        
    def _dense_allreduce(self, grad_in_buffers, grad_start_idx):
        for dtype, buffer_ in grad_in_buffers.items():
            # fp16 causes data overflow, which are nan/inf in the buffer_
            if self.args.emb_comm_opt:
                non_emb_buffer_ = buffer_.get_flat_to_end(grad_start_idx)
                non_emb_buffer_ /= self.n_workers
                all_reduce(non_emb_buffer_, op=dist.ReduceOp.SUM, group=self.group)
            else:
                buffer_.data /= self.n_workers
                all_reduce(buffer_.data, op=dist.ReduceOp.SUM, group=self.group)
    
    
    @abstractmethod
    def param_is_not_shared(param):
        return not hasattr(param, 'shared') or not param.shared


    @torch.no_grad()
    def clip_grad_norm_fp32(self, parameters, max_norm, do_clip=True, norm_type=2):
        """Clips gradient norm of an iterable of parameters whose gradients
        are in fp32.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        # Filter parameters based on:
        #   - grad should not be none
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        grads = []
        grads_for_norm = []
        for param in parameters:
            grad_not_none = param.main_grad is not None
            is_not_shared = StableTopKReducerWithValBucketsEFCorrection.param_is_not_shared(param)
            is_not_tp_duplicate = mpu.param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none:
                grad = param.main_grad.detach()
            if grad_not_none:
                # Make sure the params are in fp32
                assert param.main_grad.type() == 'torch.cuda.FloatTensor'
                grads.append(grad)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)
        
        # Norm parameters.
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        total_norm = 0.0

        # Calculate norm.
        if norm_type == inf:
            total_norm = max(grad.abs().max() for grad in grads_for_norm)
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
            # # Take max across all model-parallel GPUs.
            all_reduce(total_norm_cuda,
                        op=torch.distributed.ReduceOp.MAX,
                        group=mpu.get_model_parallel_group())
            total_norm = total_norm_cuda[0].item()

        else:
            if norm_type == 2.0:
                if len(grads_for_norm) > 0:
                    dummy_overflow_buf = torch.cuda.IntTensor([0])
                    # Use apex's multi-tensor applier for efficiency reasons.
                    # Multi-tensor applier takes a function and a list of list
                    # and performs the operation on that list all in one kernel.
                    grad_norm, _ = multi_tensor_applier(
                        amp_C.multi_tensor_l2norm,
                        dummy_overflow_buf,
                        [grads_for_norm],
                        False # no per-parameter norm
                    )
                    # Since we will be summing across data parallel groups,
                    # we need the pow(norm-type).
                    total_norm = grad_norm ** norm_type
                else:
                    total_norm = torch.tensor([0.], dtype=torch.float32, device=self.device)
                
            else:
                for grad in grads_for_norm:
                    grad_norm = torch.norm(grad, norm_type)
                    total_norm += grad_norm ** norm_type

            # # Sum across all model-parallel GPUs.
            all_reduce(total_norm,
                        op=torch.distributed.ReduceOp.SUM,
                        group=mpu.get_model_parallel_group())
            total_norm = total_norm.item() ** (1.0 / norm_type)

        # Scale.
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        if clip_coeff < 1.0 and do_clip:
            dummy_overflow_buf = torch.cuda.IntTensor([0])
            multi_tensor_applier(amp_C.multi_tensor_scale,
                                dummy_overflow_buf,
                                [grads, grads],
                                clip_coeff)
        return total_norm

    
    @torch.no_grad()
    def clip_grad_norm_fp32_by_idx(self, parameters, start_end_idx_per_layer, 
                                   max_norm, indices, do_clip=True, norm_type=2):
        """Clips gradient norm of an iterable of parameters whose gradients
        are in fp32.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        # Filter parameters based on:
        #   - grad should not be none
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        grads = []
        grads_for_norm = []
        for i, param in enumerate(parameters):
            grad_not_none = param.main_grad is not None
            is_not_shared = StableTopKReducerWithValBucketsEFCorrection.param_is_not_shared(param)
            is_not_tp_duplicate = mpu.param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none:
                grad = param.main_grad.detach()
                dtype = grad.dtype
                start_idx, end_idx = start_end_idx_per_layer[dtype][i]

                grad = grad.view(-1)[indices[dtype].data[start_idx:end_idx] == True]

            if grad_not_none:
                # Make sure the params are in fp32
                assert param.main_grad.type() == 'torch.cuda.FloatTensor'
                grads.append(grad)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)
        
        # Norm parameters.
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        total_norm = 0.0

        # Calculate norm.
        if norm_type == inf:
            total_norm = max(grad.abs().max() for grad in grads_for_norm)
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
            # # Take max across all model-parallel GPUs.
            all_reduce(total_norm_cuda,
                        op=torch.distributed.ReduceOp.MAX,
                        group=mpu.get_model_parallel_group())
            total_norm = total_norm_cuda[0].item()

        else:
            if norm_type == 2.0:
                if len(grads_for_norm) > 0:
                    dummy_overflow_buf = torch.cuda.IntTensor([0])
                    # Use apex's multi-tensor applier for efficiency reasons.
                    # Multi-tensor applier takes a function and a list of list
                    # and performs the operation on that list all in one kernel.
                    grad_norm, _ = multi_tensor_applier(
                        amp_C.multi_tensor_l2norm,
                        dummy_overflow_buf,
                        [grads_for_norm],
                        False # no per-parameter norm
                    )
                    # Since we will be summing across data parallel groups,
                    # we need the pow(norm-type).
                    total_norm = grad_norm ** norm_type
                else:
                    total_norm = torch.tensor([0.], dtype=torch.float32, device=self.device)
                
            else:
                for grad in grads_for_norm:
                    grad_norm = torch.norm(grad, norm_type)
                    total_norm += grad_norm ** norm_type

            # # Sum across all model-parallel GPUs.
            all_reduce(total_norm,
                        op=torch.distributed.ReduceOp.SUM,
                        group=mpu.get_model_parallel_group())
            square_sum = total_norm.item()
            total_norm = square_sum ** (1.0 / norm_type)

        # Scale.
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        if clip_coeff < 1.0 and do_clip:
            dummy_overflow_buf = torch.cuda.IntTensor([0])
            multi_tensor_applier(amp_C.multi_tensor_scale,
                                dummy_overflow_buf,
                                [grads, grads],
                                clip_coeff)

        return square_sum, total_norm



    def get_parameters(self, module):
        params = []
        for name, param in module.named_parameters():
            if self.args.emb_comm_opt and 'word_embeddings' in name:
                continue
            if param.requires_grad:
                params.append(param)
        return params
                                            
    
    def gradient_clip(self, module):
        params = self.get_parameters(module)
        return self.clip_grad_norm_fp32(params, self.args.clip_grad)
        # if self.do_resample:
        #     norm_type = 2
        #     topk_grad_sq_sum, _ = self.clip_grad_norm_fp32_by_idx(params, start_end_idx_per_layer, 
        #                                                         self.args.clip_grad, self.topk_indices, norm_type=norm_type)
        #     nontopk_grad_sq_sum, _ = self.clip_grad_norm_fp32_by_idx(params, start_end_idx_per_layer, 
        #                                                             self.args.clip_grad, self.nontopk_indices, norm_type=norm_type)
        #     if torch.distributed.get_rank() == 0:
        #         print("topk_grad_sq_sum: {}, nontopk_grad_sq_sum: {}".format(topk_grad_sq_sum, nontopk_grad_sq_sum), flush=True)
        #     return (topk_grad_sq_sum + nontopk_grad_sq_sum) ** (1.0 / norm_type)
        # else:
        #     return self.clip_grad_norm_fp32(params, self.args.clip_grad)
        

    @torch.no_grad()
    def _gradient_correction(self, dtype, layer_idx_, start_end_idx_per_layer, tensor):
        '''
        Perform gradient correction (like momentum correction in DGC)
        '''
        assert "optimizer_var" in self.args and "optimizer_layer_order" in self.args

        keys = {i:t for i, t in enumerate(self.args.optimizer_var.state.keys())}

        group_dicts = [{t:i for i,t in enumerate(group['params'])} for group in self.args.optimizer_var.param_groups]
        
        for i, layer_idx in enumerate(self.args.optimizer_layer_order):
            if self.args.emb_comm_opt and 'word_embeddings' in layer_idx_[dtype][layer_idx]:
                continue
            model_params = keys[i]
            beta1, beta2, bias_correction = None, None, None
            eps, weight_decay, step = None, None, None
            
            for group_idx, group in enumerate(self.args.optimizer_var.param_groups):
                if model_params in group_dicts[group_idx]:
                    # Every layer should be updated only once

                    # assert group_dicts[group_idx][model_params] not in layer_idx_set[group_idx]
                    # layer_idx_set[group_idx].add(group_dicts[group_idx][model_params])
                    
                    beta1, beta2 = group['betas']
                    bias_correction = 1 if group['bias_correction'] else 0
                    eps = group['eps']
                    weight_decay = group['weight_decay_reducer']
                    if 'step' in group:
                        group['step'] += 1
                    else:
                        group['step'] = 1
                    step = group['step']
                    break
            assert beta1 is not None and \
                beta2 is not None and \
                step is not None and \
                bias_correction is not None and \
                eps is not None and \
                weight_decay is not None 
                
            start_end_idx = start_end_idx_per_layer[dtype][layer_idx]
            start_idx, end_idx = start_end_idx

            
            assert model_params.numel() == end_idx - start_idx and end_idx <= tensor.numel()
 
            original_shape = model_params.shape
            
            grads_cur_layer = tensor[start_idx:end_idx].view(*original_shape)
            
            exp_avg = self.args.optimizer_var.state[keys[i]]['exp_avg']
            exp_avg_sq = self.args.optimizer_var.state[keys[i]]['exp_avg_sq']

            # [NOTE]
            # Create a reference for momentum_buffer to avoid memory waste
            # since momentum=0 is fixed in this case, momentum_buffer is never used.
            if 'momentum_buffer' not in self.args.optimizer_var.state[keys[i]]:
                self.args.optimizer_var.state[keys[i]]['momentum_buffer'] = exp_avg
            
            model_params_clone = keys[i].clone()

            g_32 = [grads_cur_layer]
            p_32 = [model_params_clone]
            m_32 = [exp_avg]
            v_32 = [exp_avg_sq]
            
            multi_tensor_applier(self.multi_tensor_adam,
                                self._dummy_overflow_buf,
                                [g_32, p_32, m_32, v_32],
                                1.0, # avoid using lr here, this causes grad norm to be large
                                beta1,
                                beta2,
                                eps,
                                step,
                                True, #adam_w_mode is default to be True
                                bias_correction,
                                weight_decay)
            
             # w = w - lr * grad - lr * weight_decay * w, so need to recover it
            tensor[start_idx:end_idx] = (model_params-p_32[0]).flatten()
    
    
    @torch.no_grad()    
    def gradient_correction(self, layer_idx_, start_end_idx_per_layer, grad_in_buffers):
        for dtype, buffer_ in grad_in_buffers.items():
            all_grads = buffer_.data 
            self._gradient_correction(dtype, layer_idx_, start_end_idx_per_layer, all_grads)
                
    
    
    def resample_topk_indices(self,
                              train_step,
                              layer_idx_,
                              start_end_idx_per_layer,
                              grad_start_idx,
                              grad_in_buffers):
        with torch.no_grad():
            if train_step >= self.stable_topk_threshold:
                for dtype, buffer_ in grad_in_buffers.items():
                    if train_step % self.stable_topk_interval == 0 or self.do_resample:
                        self.do_resample = False
                        if 0 < self.density < 1:
                            if self.args.emb_comm_opt:
                                all_grads = buffer_.get_flat_to_end(grad_start_idx)
                            else:
                                all_grads = buffer_.data
                            
                            # [NOTE] Only supports deterministic density to avoid dynamic memory allocation
                            self.topk_indices[dtype].data.fill_(False)
                            self.nontopk_indices[dtype].data.fill_(True)
                            self._resample_topk_indices(dtype, layer_idx_, start_end_idx_per_layer, grad_start_idx, all_grads)
                            
    @torch.no_grad()
    def _resample_topk_indices(self, dtype, layer_idx_, start_end_idx_per_layer, grad_start_idx, tensor):
        for i, layer_start_end_idx in start_end_idx_per_layer[dtype].items():
            if self.args.emb_comm_opt and 'word_embeddings' in layer_idx_[dtype][i]:
                continue
            
            layer_start_idx, layer_end_idx = layer_start_end_idx
            
            if self.args.emb_comm_opt:
                layer_start_idx = layer_start_idx - grad_start_idx
                layer_end_idx = layer_end_idx - grad_start_idx
            
            numel_cur_layer = layer_end_idx - layer_start_idx
            cur_layer_topk_idx = self.topk_indices[dtype].get_flat_from_start_to_end(layer_start_idx, layer_end_idx)
            cur_layer_nontopk_idx = self.nontopk_indices[dtype].get_flat_from_start_to_end(layer_start_idx, layer_end_idx)
            
            if 'bias' in layer_idx_[dtype][i]:
                # indices.append(torch.arange(layer_start_idx, layer_end_idx, device=self.device))
                cur_layer_topk_idx[:] = True
                cur_layer_nontopk_idx[:] = False
            else:
                ratio = self.density
                k=max(int(numel_cur_layer*ratio), 1)
                _, topk_indices = torch.topk(tensor[layer_start_idx:layer_end_idx].abs(), k=k)
                
                cur_layer_topk_idx[topk_indices] = True
                cur_layer_nontopk_idx[topk_indices] = False

        # indices = torch.cat(indices).reshape(-1)
        # return indices

    @torch.no_grad()
    def check_intersection_rate(self, dtype, indices, update_prev_indices=False):
        cur_indices= set(indices.cpu().numpy())
        if dtype in self.prev_grad_topk_indices:
            intersection = self.prev_grad_topk_indices[dtype].intersection(cur_indices)
            intersection_rate = len(intersection)/len(cur_indices)
            if dist.get_rank() == 0:
                print(f"Intersection rate: {intersection_rate}")
        dist.barrier()
        
        if update_prev_indices:
            self.prev_grad_topk_indices[dtype] = cur_indices

    
    def compress_org(self, 
                     module,
                     dtype,
                     layer_idx_,
                     start_end_idx_per_layer,
                     grad_start_idx,
                     tensor):
        with torch.no_grad(): 
            
            # # Residuals here store the gradients, not momentums, nor the corrected momentums
            # if dtype not in self.residuals:
            #     self.residuals[dtype] = torch.zeros_like(tensor, dtype=dtype, device=self.device)
            
            if 'residuals' not in self.args.optimizer_var.state:
                self.args.optimizer_var.state['residuals'] = torch.zeros_like(tensor, dtype=dtype, device=self.device, requires_grad=False)
            elif self.args.optimizer_var.state['residuals'].get_device() != self.device:
                self.args.optimizer_var.state['residuals'] = self.args.optimizer_var.state['residuals'].to(self.device)
            
            if dtype not in self.topk_indices:
                self.topk_indices[dtype] = MemoryBuffer(tensor.numel(), dtype=bool)
                self.topk_indices[dtype].data.fill_(False)

            if dtype not in self.nontopk_indices:
                self.nontopk_indices[dtype] = MemoryBuffer(tensor.numel(), dtype=bool)
                self.nontopk_indices[dtype].data.fill_(True)

            self.do_resample = True

            tensor.add_(self.args.optimizer_var.state['residuals'])
            tensor.div_(self.n_workers)
            
            # Perform all_reduce to get the averaged gradients from all workers
            all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
            
            self.args.optimizer_var.state['residuals'].fill_(0.0)
            
            
    @torch.no_grad()
    def get_compressed_tensor_len(self,
                                  dtype,
                                  start_end_idx_per_layer,
                                  layer_idx_):
        numel = 0
        for i, layer_start_end_idx in start_end_idx_per_layer[dtype].items():
            start_idx, end_idx = layer_start_end_idx
            numel_cur_layer = end_idx - start_idx
            if 'bias' not in layer_idx_[dtype][i]:
                numel += max(int(self.density * numel_cur_layer), 1)
            else:
                numel += numel_cur_layer
        return numel
        

    def compress_stable(self, 
                        dtype, 
                        module,
                        layer_idx_,
                        start_end_idx_per_layer,
                        grad_start_idx,
                        tensor):
        with torch.no_grad():
            # # Make sure compress_org has been called before calling compress_stable
            # # assert dtype in self.buckets_indices and \
            # #         dtype in self.tensor_buckets_start_end_indices
            
            assert 'residuals' in self.args.optimizer_var.state and \
                    dtype in self.topk_indices and \
                    dtype in self.nontopk_indices 
            
            if dtype not in self.compressed_grad:
                compressed_size = self.get_compressed_tensor_len(dtype, start_end_idx_per_layer, layer_idx_)
                self.compressed_grad[dtype] = MemoryBuffer(compressed_size, dtype=dtype)
            
            if self.density == 1:
                self.compressed_grad[dtype].data[:] = tensor
            else:
                prev_numel = 0
                for i, layer_start_end_idx in start_end_idx_per_layer[dtype].items():
                    layer_start_idx, layer_end_idx = layer_start_end_idx
                    if 'bias' not in layer_idx_[dtype][i]:
                        cur_layer_numel_compressed = max(int(self.density * (layer_end_idx - layer_start_idx)), 1)
                    else:
                        cur_layer_numel_compressed = layer_end_idx - layer_start_idx
                    cur_layer_topk_indices = self.topk_indices[dtype].get_flat_from_start_to_end(layer_start_idx, 
                                                                                                 layer_end_idx)
                    compressed_tensor_slice = self.compressed_grad[dtype].get_flat_from_start_to_end(prev_numel, 
                                                                                                     prev_numel + cur_layer_numel_compressed)
                    compressed_tensor_slice[:] = tensor[layer_start_idx:layer_end_idx][cur_layer_topk_indices == True]
                    prev_numel += cur_layer_numel_compressed
            self.compressed_grad[dtype].data.div_(self.n_workers)
            
            
            # Allreduce on selected/compressed gradients
            all_reduce(self.compressed_grad[dtype].data, op=dist.ReduceOp.SUM, group=self.group)
            
            if 0 <= self.density < 1:
                prev_numel = 0
                for i, layer_start_end_idx in start_end_idx_per_layer[dtype].items():
                    layer_start_idx, layer_end_idx = layer_start_end_idx
                    if 'bias' not in layer_idx_[dtype][i]:
                        cur_layer_numel_compressed = max(int(self.density * (layer_end_idx - layer_start_idx)), 1)
                    else:
                        cur_layer_numel_compressed = layer_end_idx - layer_start_idx
                    cur_layer_topk_indices = self.topk_indices[dtype].get_flat_from_start_to_end(layer_start_idx, 
                                                                                            layer_end_idx)
                    compressed_tensor_slice = self.compressed_grad[dtype].get_flat_from_start_to_end(prev_numel, 
                                                                                                     prev_numel + cur_layer_numel_compressed)
                    tensor[layer_start_idx:layer_end_idx][cur_layer_topk_indices == True] = compressed_tensor_slice
                    prev_numel += cur_layer_numel_compressed

                self.args.optimizer_var.state['residuals'].add_(tensor)
                self.args.optimizer_var.state['residuals'].mul_(self.nontopk_indices[dtype].data)
                tensor.mul_(self.topk_indices[dtype].data)

    
    def reconstruct_grads(self, 
                          dtype, 
                          allreduce_results, 
                          grad_in_buffers, 
                          grad_out_buffers, 
                          grad_start_idx):
        '''
        Divide the combined gradients into each layer's gradient buffer.
        Update on the gradients will be performed after self.reduce has returned
        '''
        # Normally, only one type of data in gradients buffers
        if self.args.emb_comm_opt:
            emb_grads = grad_in_buffers[dtype].data[:grad_start_idx]
            grad_out_buffers[dtype].data = torch.cat(emb_grads, allreduce_results)
        else:
            grad_out_buffers[dtype].data = allreduce_results

    
    def reduce(self, 
               train_iter,
               module, 
               layer_idx_,
               start_end_idx_per_layer,
               grad_in_buffers, 
               grad_out_buffers, 
               grad_start_idx):
        """
        grad_in_buffers:  A list of continuous memory which contains all the grads from diff layers
                          No need to worry about mapping issues.
                          They have been taken care when param.main_grads are constructed.
                          **[NOTE] grad_in_buffer is the reverse of the layer order**
                          **I.e. grad_in_buffers = L_N, L_{N-1}, ..., L_2, L_1**
        grad_out_buffers: Similar to grad_in_buffers, but will not use it until warmup stage finishes.
                          **[NOTE] grad_out_buffer = L_1, L_2, ..., L_{N-1}, L_N**
        grad_start_idx:   Starting index of the gradient, excluding embedding layer grads.
        return: Whether need to do copy back from grad_in to grad_out or not.
        """
        if train_iter < self.stable_topk_threshold:
            # Collect topk grad distributions
            self._dense_allreduce(grad_in_buffers, grad_start_idx)
            # report whether in warmup stage or not every 100 iterations
            if train_iter % 100 == 0 and dist.get_rank() == 0:
                print(' >> Still in Warm-up Period... ', train_iter)
        else:    
            # Here we assume there are more than one type of data in grad_in_buffers.
            # Even though this is usually unlikely, we need to keep the integrity of the code.
            # [Finished] for dtype, buffer_ in grad_in_buffers.items():
            # all_grads = self.combine_all_grads(grad_in_buffers)
            
            for dtype, buffer_ in grad_in_buffers.items():
                if self.args.emb_comm_opt:
                    all_grads = buffer_.get_flat_to_end(grad_start_idx)
                    # if dist.get_rank() == 0:
                    #     print(f"all_grads/grad_in_buffers: {100*all_grads.numel()/buffer_.data.numel()}%")
                else:
                    all_grads = buffer_.data
                # Recompute every self.stable_topk_interval intervals or we have no indecies 
                if train_iter % self.stable_topk_interval == 0 or dtype not in self.topk_indices or dtype not in self.nontopk_indices:
                    # compress_org performs operation directly on all_grads
                    # so processed_tensor and all_grads point to the same underlying tensor
                    self.compress_org(module,
                                        dtype, 
                                        layer_idx_,
                                        start_end_idx_per_layer,
                                        grad_start_idx,
                                        all_grads)
                else:
                    # perform update locally
                    self.compress_stable(dtype,
                                        module,
                                        layer_idx_,
                                        start_end_idx_per_layer,
                                        grad_start_idx,
                                        all_grads)
            
            # self.reconstruct_grads(dtype,
            #                        processed_tensor,
            #                        grad_in_buffers,
            #                        grad_out_buffers,
            #                        grad_start_idx)
        
        # if warmup ends, need to use reduced_buffer to update grad_buffer
        # no need to do so during the warmup stage, becuase grad_in_buffers points to models' grad tensor
        # return train_iter >= self.stable_topk_threshold
        return False
        

class EmbStableTopKReducerWithRangeBucketsEFCorrectionResIsGrad(Reducer):
    '''
    **This is the version w/ error feedback added.**
    **Simply applying EF will lead to divergence.**
    Now will try to use momentum to regulate it.
    1. All_reduce on gradients.
    2. Select topk indices
    3. Make topk indices as centers, then select the gradients on the left and right by 500-range.
    4. Gradient correction.
    '''
    def __init__(self,
                 random_seed,
                 device,
                 group,
                 group_num,
                 density,
                 stable_topk_interval,
                 stable_topk_threshold,
                 stable_topk_range,
                 stable_topk_warmup_method="Dense"):
        
        # params for comm setup
        super().__init__(random_seed, device, group, group_num)
        
        # compression density
        self.density = density
        # how often will resampling happen
        self.stable_topk_interval = stable_topk_interval
        # how long will the warmup stage last
        self.stable_topk_threshold = stable_topk_threshold
        # which warmup method will be used ([TODO] used to support other methods, such as gtopk)
        assert stable_topk_warmup_method == "Dense", "Currently only support Dense Allreduce as the warmup method."
        self.stable_topk_warmup_method = stable_topk_warmup_method
        # How many buckets will the topk gradients will be divided into
        self.stable_topk_range = stable_topk_range
        
        ## for storing layer dim info
        # self.grad_shapes = {}
        # self.grad_sizes = {}
        
        self.args = get_args()
        
        # mask for extracting the residuals
        self.zero_conditions = None
        
        # Residuals (i.e. non-topk values)
        self.residuals = {}
        
        
        # range formed with topk as centers and self.stable_topk_range as radius
        self.intervals = None
        self.indices = None

        # analyze the importance of gradients
        self.prev_grad_topk_indices = {}
        
        # For amp_C fused_adam function
        if multi_tensor_applier.available:
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_adam = amp_C.multi_tensor_adam
        
        # print configurations into log
        if dist.get_rank() == 0:
            self._init_printer()
          
            
    def _init_printer(self):
        print('===== EmbStableTopKReducerWithRangeBucketsEFCorrectionResIsGrad Reducer =====')
        print(' >> density: ', self.density)
        print(' >> resampling interval: ', self.stable_topk_interval)
        print(' >> number of warmup iterations: ', self.stable_topk_threshold)
        print(' >> warmup method: ', self.stable_topk_warmup_method)
        print(' >> range radius: ', self.stable_topk_range)
        print('==========================================================================')
        
        
    def _dense_allreduce(self, tensor):
        tensor.div_(self.n_workers) 
        all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
    
    
    @abstractmethod
    def param_is_not_shared(param):
        return not hasattr(param, 'shared') or not param.shared


    def clip_grad_norm_fp32(self, parameters, max_norm, do_clip=True, norm_type=2):
        """Clips gradient norm of an iterable of parameters whose gradients
        are in fp32.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        # Filter parameters based on:
        #   - grad should not be none
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        grads = []
        grads_for_norm = []
        for param in parameters:
            grad_not_none = param is not None
            is_not_shared = StableTopKReducerWithValBucketsEFCorrection.param_is_not_shared(param)
            is_not_tp_duplicate = mpu.param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none:
                grad = param.detach()
            if grad_not_none:
                # Make sure the params are in fp32
                assert param.type() == 'torch.cuda.FloatTensor'
                grads.append(grad)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)
        
        # Norm parameters.
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        total_norm = 0.0

        # Calculate norm.
        if norm_type == inf:
            total_norm = max(grad.abs().max() for grad in grads_for_norm)
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
            # # Take max across all model-parallel GPUs.
            all_reduce(total_norm_cuda,
                        op=torch.distributed.ReduceOp.MAX,
                        group=mpu.get_model_parallel_group())
            total_norm = total_norm_cuda[0].item()

        else:
            if norm_type == 2.0:
                if len(grads_for_norm) > 0:
                    dummy_overflow_buf = torch.cuda.IntTensor([0])
                    # Use apex's multi-tensor applier for efficiency reasons.
                    # Multi-tensor applier takes a function and a list of list
                    # and performs the operation on that list all in one kernel.
                    grad_norm, _ = multi_tensor_applier(
                        amp_C.multi_tensor_l2norm,
                        dummy_overflow_buf,
                        [grads_for_norm],
                        False # no per-parameter norm
                    )
                    # Since we will be summing across data parallel groups,
                    # we need the pow(norm-type).
                    total_norm = grad_norm ** norm_type
                else:
                    total_norm = torch.tensor([0.], dtype=torch.float32, device=self.device)
                
            else:
                for grad in grads_for_norm:
                    grad_norm = torch.norm(grad, norm_type)
                    total_norm += grad_norm ** norm_type

            # # Sum across all model-parallel GPUs.
            all_reduce(total_norm,
                        op=torch.distributed.ReduceOp.SUM,
                        group=mpu.get_model_parallel_group())
            total_norm = total_norm.item() ** (1.0 / norm_type)

        # Scale.
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        if clip_coeff < 1.0 and do_clip:
            dummy_overflow_buf = torch.cuda.IntTensor([0])
            multi_tensor_applier(amp_C.multi_tensor_scale,
                                dummy_overflow_buf,
                                [grads, grads],
                                clip_coeff)
        return total_norm

                                            
    
    def gradient_clip(self, tensor):
        params = [tensor]
        return self.clip_grad_norm_fp32(params, self.args.clip_grad)
 

    def gradient_correction(self, tensor):
        '''
        Perform gradient correction (like momentum correction in DGC)
        '''
        assert "optimizer_var" in self.args and "optimizer_layer_order" in self.args

        # layer_idx_set = [set() for _ in range(len(self.args.optimizer_var.param_groups))]
        group_dicts = [{t:i for i,t in enumerate(group['params'])} for group in self.args.optimizer_var.param_groups]
        
        
        for model_params in self.args.optimizer_var.state.keys():
            # Perform correction on embedding layer only, which is usually the first layer
            if type(model_params) == str or model_params.shape != tensor.shape:
                continue
            beta1, beta2, bias_correction = None, None, None
            eps, weight_decay, step = None, None, None
            
            for group_idx, group in enumerate(self.args.optimizer_var.param_groups):
                if model_params in group_dicts[group_idx]:
                    
                    beta1, beta2 = group['betas']
                    bias_correction = 1 if group['bias_correction'] else 0
                    eps = group['eps']
                    weight_decay = group['weight_decay_reducer']
                    if 'step' in group:
                        group['step'] += 1
                    else:
                        group['step'] = 1
                    step = group['step']
                    break
            assert beta1 is not None and \
                beta2 is not None and \
                step is not None and \
                bias_correction is not None and \
                eps is not None and \
                weight_decay is not None 
 
            original_shape = model_params.shape
            
            grads_cur_layer = tensor.view(*original_shape)
            
            exp_avg = self.args.optimizer_var.state[model_params]['exp_avg']
            exp_avg_sq = self.args.optimizer_var.state[model_params]['exp_avg_sq']
            
            model_params_clone = model_params.clone()

            g_32 = [grads_cur_layer]
            p_32 = [model_params_clone]
            m_32 = [exp_avg]
            v_32 = [exp_avg_sq]
            
            multi_tensor_applier(self.multi_tensor_adam,
                                self._dummy_overflow_buf,
                                [g_32, p_32, m_32, v_32],
                                1.0, # avoid using lr here, this causes grad norm to be large
                                beta1,
                                beta2,
                                eps,
                                step,
                                True, #adam_w_mode is default to be True
                                bias_correction,
                                weight_decay)
            
             # w = w - lr * grad - lr * weight_decay * w, so need to recover it
            tensor.copy_(model_params-p_32[0])

    
    def resample_topk_indices(self, tensor):
        if self.stable_topk_range > 0:
            intervals = self._resample_topk_intervals(tensor)
            self.intervals = self.merge_intervals(torch.cat(intervals).tolist())
            self.indices = self.generate_indices_from_intervals(self.intervals)
        else:
            self.indices = self._resample_topk_indices(tensor)   
        
        self.zero_conditions.index_fill_(0,self.indices,0)
        
        if torch.distributed.get_rank() == 0:
            print(f"Real word emb compression ratio: {self.indices.numel()/tensor.numel():.4f}", flush=True)
        dist.barrier()        


    def _resample_topk_intervals(self, tensor):
        intervals = []
        
        numel_cur_layer = tensor.numel()
        ratio = self.density
        k=max(int(numel_cur_layer*ratio), 1)
        _, topk_indices = torch.topk(tensor.view(-1).abs(), k=k)
        topk_indices, _ = torch.sort(topk_indices)

        left_boundary = (topk_indices - self.stable_topk_range).reshape(-1,1)
        left_boundary[left_boundary < 0] = 0
        right_boundary = (topk_indices + self.stable_topk_range + 1).reshape(-1,1)
        right_boundary[right_boundary > numel_cur_layer] = numel_cur_layer
        intervals.append(torch.cat([left_boundary, right_boundary],dim=1))
            
        return intervals
   
    
    def _resample_topk_indices(self, tensor):
    
        ratio = self.density
        numel_cur_layer = tensor.numel()
        k=max(int(numel_cur_layer*ratio), 1)
        _, topk_indices = torch.topk(tensor.view(-1).abs(), k=k)
        topk_indices,_ = torch.sort(topk_indices)
        
        return topk_indices


    def generate_indices_from_intervals(self, intervals):
        indices = []
        for interval in intervals:
            start, end = interval
            indices.append(torch.arange(start,end,device=self.device))
        return torch.cat(indices)
 
    
    def compress_org(self, tensor):
        with torch.no_grad(): 
            
            # # Residuals here store the gradients, not momentums, nor the corrected momentums
            # if dtype not in self.residuals:
            #     self.residuals[dtype] = torch.zeros_like(tensor, dtype=dtype, device=self.device)
            
            
            if 'emb_residuals' not in self.args.optimizer_var.state:
                self.args.optimizer_var.state['emb_residuals'] = torch.zeros_like(tensor.view(-1), dtype=tensor.dtype, device=self.device)
            elif self.args.optimizer_var.state['emb_residuals'].get_device() != self.device:
                self.args.optimizer_var.state['emb_residuals'] = self.args.optimizer_var.state['emb_residuals'].to(self.device)
            
            # As a mask for residual selection
            if self.zero_conditions is None:
                self.zero_conditions = torch.ones_like(tensor.view(-1), dtype=tensor.dtype, device=self.device)
            
            tensor.view(-1).add_(self.args.optimizer_var.state['emb_residuals'])
            tensor.view(-1).div_(self.n_workers)
            
            # Perform all_reduce to get the averaged gradients from all workers
            all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
            
            self.zero_conditions.fill_(1.0)
            
            self.args.optimizer_var.state['emb_residuals'].fill_(0.0)
            
            if self.args.move_grad_clip_to_reducer:   
                self.args.grad_norm = self.gradient_clip(tensor)
            
            # # tensor = self.gradient_correction(dtype, layer_idx_, start_end_idx_per_layer, tensor, reducer='stable')
            self.gradient_correction(tensor)
            
            if 0 <= self.density < 1:
                
                self.resample_topk_indices(tensor)         
                       
            #     # # initialize the zero_conditions as masks for selecting residuals
            #     # # i.e. residuals=tensor*zero_conditions
            #     self.zero_conditions[dtype].fill_(1.0)
                
            #     # initialize the residuals for later accumulation
            #     self.args.optimizer_var.state['residuals'].fill_(0.0)
                
            #     self.zero_conditions[dtype].index_fill_(0,self.indices[dtype],0)

            #     if torch.distributed.get_rank() == 0 and dtype in self.indices:
            #         print(f"Real compression ratio: {len(self.indices[dtype])/tensor.numel():.4f}", flush=True)
            #     dist.barrier()
            
        
    def compress_stable(self, tensor):
        with torch.no_grad():
            # # Make sure compress_org has been called before calling compress_stable
            # # assert dtype in self.buckets_indices and \
            # #         dtype in self.tensor_buckets_start_end_indices
            
            assert 'emb_residuals' in self.args.optimizer_var.state and \
                    self.zero_conditions is not None and \
                    self.indices is not None
            
            if self.density == 1:
                selected_tensor = tensor
            else:
                selected_tensor = torch.index_select(tensor.view(-1),0,self.indices)
                
            selected_tensor.div_(self.n_workers)
            
            
            # Allreduce on selected/compressed gradients
            all_reduce(selected_tensor, op=dist.ReduceOp.SUM, group=self.group)
            
            # # indices = self.resample_topk_indices(dtype, start_end_idx_per_layer, tensor)
            # # self.check_intersection_rate(dtype, indices)

            if 0 <= self.density < 1:
                tensor.view(-1)[self.indices] = selected_tensor
                self.args.optimizer_var.state['emb_residuals'].add_(tensor.view(-1) * self.zero_conditions)
                tensor.view(-1).mul_(1 - self.zero_conditions)
                
                
            if self.args.move_grad_clip_to_reducer:
                # tensor = self.gradient_clip(dtype, start_end_idx_per_layer, tensor)
                self.gradient_clip(tensor)
            
            # # Perform direction correction here
            # tensor = self.gradient_correction(dtype, layer_idx_, start_end_idx_per_layer, tensor, reducer='stable')
            self.gradient_correction(tensor)

    
    def reduce(self, tensor, train_iter):
        if train_iter < self.stable_topk_threshold:
            # Collect topk grad distributions
            self._dense_allreduce(tensor)
            # report whether in warmup stage or not every 100 iterations
            if train_iter % 100 == 0 and dist.get_rank() == 0:
                print(' >> Still in Warm-up Period... ', train_iter)
        else:    
            # Here we assume there are more than one type of data in grad_in_buffers.
            # Even though this is usually unlikely, we need to keep the integrity of the code.
            # [Finished] for dtype, buffer_ in grad_in_buffers.items():
            # all_grads = self.combine_all_grads(grad_in_buffers)
            if train_iter % self.stable_topk_interval == 0 or self.indices is None or self.zero_conditions is None:
                # compress_org performs operation directly on all_grads
                # so processed_tensor and all_grads point to the same underlying tensor
                self.compress_org(tensor)
            else:
                # perform update locally
                self.compress_stable(tensor)
    

class TopKReducerEFCorrectionResIsGradWarmup(Reducer):
    '''
    **This is the version w/ error feedback added.**
    **Simply applying EF will lead to divergence.**
    Instead of using range, we only use density to get the correct topk at every step.
    '''
    def __init__(self,
                 random_seed,
                 device,
                 group,
                 group_num,
                 beta1,
                 beta2,
                 epsilon,
                 density,
                 stable_topk_threshold,
                 stable_topk_density_warmup_steps,
                 stable_topk_warmup_method="Dense"):
        
        # params for comm setup
        super().__init__(random_seed, device, group, group_num)
        
        # compression density
        self.density = density
        # how long will the warmup stage last
        self.stable_topk_threshold = stable_topk_threshold
        # how long will warmup stage last (after switching from dense to other comp method)
        self.stable_topk_density_warmup_steps = stable_topk_density_warmup_steps
        # which warmup method will be used ([TODO] used to support other methods, such as gtopk)
        assert stable_topk_warmup_method == "Dense", "Currently only support Dense Allreduce as the warmup method."
        self.stable_topk_warmup_method = stable_topk_warmup_method
        
        ## for storing layer dim info
        # self.grad_shapes = {}
        # self.grad_sizes = {}
        
        self.args = get_args()
        
        # define warmup coefficient
        self.warmup_coeff = self.density ** (1. / (self.stable_topk_density_warmup_steps + 1))
        self.step = 1 # This is not the global step. Only tracks # of step since start/resume
        
        # mask for extracting the residuals
        self.zero_conditions = {}
        
        # dict for transmitting data in sparse_allreduce
        self.storage = {}
        
        # Store buckets that are merged acorss all workers in the same group
        self.buckets_indices = {}
        self.tensor_buckets_start_end_indices = {}
        self.num_element_per_chunk={}
        self.selected_num_element_per_chunk={}
        
        # A List[int] that contains the numel per layer
        self.num_element_per_layer = {}
        
        # Store intermediate values in adam (i.e. m_t and v_t)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.tensor_corrected = {}

        # range formed with topk as centers and self.stable_topk_range as radius
        self.intervals = {}
        self.indices = {}

        # analyze the importance of gradients
        self.prev_grad_topk_indices = {}
        
        # For amp_C fused_adam function
        if multi_tensor_applier.available:
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_adam = amp_C.multi_tensor_adam
        
        # print configurations into log
        if dist.get_rank() == 0:
            self._init_printer()
          
            
    def _init_printer(self):
        print('===== TopKReducerEFCorrectionResIsGradWarmup Reducer =====')
        print(' >> density: ', self.density)
        print(' >> number of warmup iterations: ', self.stable_topk_threshold)
        print(' >> warmup method: ', self.stable_topk_warmup_method)
        print(' >> density warmup steps: ', self.stable_topk_density_warmup_steps)
        print('==========================================================================')
        
        
    def _dense_allreduce(self, grad_in_buffers, grad_start_idx):
        for dtype, buffer_ in grad_in_buffers.items():
            # fp16 causes data overflow, which are nan/inf in the buffer_
            if self.args.emb_comm_opt:
                non_emb_buffer_ = buffer_.get_flat_to_end(grad_start_idx)
                non_emb_buffer_ /= self.n_workers
                all_reduce(non_emb_buffer_, op=dist.ReduceOp.SUM, group=self.group)
            else:
                buffer_.data /= self.n_workers
                all_reduce(buffer_.data, op=dist.ReduceOp.SUM, group=self.group)
    
    
    @abstractmethod
    def param_is_not_shared(param):
        return not hasattr(param, 'shared') or not param.shared


    def clip_grad_norm_fp32(self, parameters, max_norm, do_clip=True, norm_type=2):
        """Clips gradient norm of an iterable of parameters whose gradients
        are in fp32.

        This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
        added functionality to handle model parallel parameters. Note that
        the gradients are modified in place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        # Filter parameters based on:
        #   - grad should not be none
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        grads = []
        grads_for_norm = []
        for param in parameters:
            grad_not_none = param is not None
            is_not_shared = StableTopKReducerWithValBucketsEFCorrection.param_is_not_shared(param)
            is_not_tp_duplicate = mpu.layers.param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none:
                grad = param.detach()
            if grad_not_none:
                # Make sure the params are in fp32
                assert param.type() == 'torch.cuda.FloatTensor'
                grads.append(grad)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)

        # Norm parameters.
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        total_norm = 0.0

        # Calculate norm.
        if norm_type == inf:
            total_norm = max(grad.abs().max() for grad in grads_for_norm)
            total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
            # # Take max across all model-parallel GPUs.
            all_reduce(total_norm_cuda,
                        op=torch.distributed.ReduceOp.MAX,
                        group=mpu.get_model_parallel_group())
            total_norm = total_norm_cuda[0].item()

        else:
            if norm_type == 2.0:
                dummy_overflow_buf = torch.cuda.IntTensor([0])
                # Use apex's multi-tensor applier for efficiency reasons.
                # Multi-tensor applier takes a function and a list of list
                # and performs the operation on that list all in one kernel.
                grad_norm, _ = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False # no per-parameter norm
                )
                # Since we will be summing across data parallel groups,
                # we need the pow(norm-type).
                total_norm = grad_norm ** norm_type

            else:
                for grad in grads_for_norm:
                    grad_norm = torch.norm(grad, norm_type)
                    total_norm += grad_norm ** norm_type

            # # Sum across all model-parallel GPUs.
            all_reduce(total_norm,
                        op=torch.distributed.ReduceOp.SUM,
                        group=mpu.get_model_parallel_group())
            total_norm = total_norm.item() ** (1.0 / norm_type)

        # Scale.
        clip_coeff = max_norm / (total_norm + 1.0e-6)
        if clip_coeff < 1.0 and do_clip:
            dummy_overflow_buf = torch.cuda.IntTensor([0])
            multi_tensor_applier(amp_C.multi_tensor_scale,
                                dummy_overflow_buf,
                                [grads, grads],
                                clip_coeff)
        return total_norm


    def gradient_clip(self, dtype, start_end_idx_per_layer, tensor):
        # for start_end_idx_cur_layer in start_end_idx_per_layer[dtype].values():    
        #     start_idx, end_idx = start_end_idx_cur_layer
        #     torch.nn.utils.clip_grad_norm_(tensor[start_idx:end_idx], self.args.clip_grad)  
        if dtype not in self.num_element_per_layer:
            self.num_element_per_layer[dtype] = []
            for start_end_idx_cur_layer in start_end_idx_per_layer[dtype].values():   
                start_idx, end_idx = start_end_idx_cur_layer
                self.num_element_per_layer[dtype].append(end_idx - start_idx)
        tensor_splitted_by_layer = torch.split(tensor, self.num_element_per_layer[dtype])
        # grad_norm = self.clip_grad_norm_fp32(tensor_splitted_by_layer, self.args.clip_grad*(self.n_workers)**(-1/2))
        grad_norm = self.clip_grad_norm_fp32(tensor_splitted_by_layer, self.args.clip_grad)
        # print(f"[rank {dist.get_rank()}]grad norm: ", grad_norm)
        tensor_clipped = torch.cat(tensor_splitted_by_layer)
        return tensor_clipped
 
 
    def gradient_correction(self, dtype, layer_idx_, start_end_idx_per_layer, tensor, reducer='stable'):
        '''
        Perform gradient correction (like momentum correction in DGC)
        '''
        assert "optimizer_var" in self.args and "optimizer_layer_order" in self.args

        keys = {i:t for i, t in enumerate(self.args.optimizer_var.state.keys())}

        layer_idx_set = [set() for _ in range(len(self.args.optimizer_var.param_groups))]
        group_dicts = [{t:i for i,t in enumerate(group['params'])} for group in self.args.optimizer_var.param_groups]
        
        for i, layer_idx in enumerate(self.args.optimizer_layer_order):
            model_params = keys[i]
            beta1, beta2, bias_correction = None, None, None
            eps, weight_decay, step = None, None, None
            
            for group_idx, group in enumerate(self.args.optimizer_var.param_groups):
                if model_params in group_dicts[group_idx]:
                    # Every layer should be updated only once

                    assert group_dicts[group_idx][model_params] not in layer_idx_set[group_idx]
                    layer_idx_set[group_idx].add(group_dicts[group_idx][model_params])
                    
                    beta1, beta2 = group['betas']
                    bias_correction = 1 if group['bias_correction'] else 0
                    eps = group['eps']
                    weight_decay = group['weight_decay_reducer']
                    if 'step' in group:
                        group['step'] += 1
                    else:
                        group['step'] = 1
                    step = group['step']
                    break
            assert beta1 is not None and \
                beta2 is not None and \
                step is not None and \
                bias_correction is not None and \
                eps is not None and \
                weight_decay is not None 
                
            start_end_idx = start_end_idx_per_layer[dtype][layer_idx]
            start_idx, end_idx = start_end_idx
            assert model_params.numel() == end_idx - start_idx 
            
            original_shape = model_params.shape
            
            grads_cur_layer = tensor[start_idx:end_idx].reshape(*original_shape)
            
            exp_avg = self.args.optimizer_var.state[keys[i]]['exp_avg']
            exp_avg_sq = self.args.optimizer_var.state[keys[i]]['exp_avg_sq']

            # # [NOTE] take out non-topk momentum 
            # if reducer != 'org':
            #     mask = self.zero_conditions[dtype][start_idx:end_idx].reshape(*original_shape)
            #     exp_avg_nontopk = exp_avg * mask
            #     exp_avg_sq_nontopk = exp_avg_sq * mask
            #     exp_avg.data.mul_(1-mask)
            #     exp_avg_sq.data.mul_(1-mask)
            
            model_params_clone = keys[i].clone()

            g_32 = [grads_cur_layer]
            p_32 = [model_params_clone]
            m_32 = [exp_avg]
            v_32 = [exp_avg_sq]
            
            multi_tensor_applier(self.multi_tensor_adam,
                                self._dummy_overflow_buf,
                                [g_32, p_32, m_32, v_32],
                                1.0, # avoid using lr here, this causes grad norm to be large
                                beta1,
                                beta2,
                                eps,
                                step,
                                True, #adam_w_mode is default to be True
                                bias_correction,
                                weight_decay)
            
            # grads_cur_layer=grads_cur_layer.reshape(*original_shape)
            # exp_avg.mul_(beta1).add_(grads_cur_layer, alpha=1-beta1)
            # exp_avg_sq.mul_(beta2).addcmul_(grads_cur_layer, grads_cur_layer, value=1-beta2)
            # bias_correction1 = 1 - beta1 ** step
            # bias_correction2 = 1 - beta2 ** step
            # if reducer != 'org':
            #     exp_avg_hat = (exp_avg + exp_avg_nontopk)/bias_correction1
            #     exp_avg_sq_hat = (exp_avg_sq + exp_avg_sq_nontopk)/bias_correction2
            # else:
            #     exp_avg_hat = exp_avg/bias_correction1
            #     exp_avg_sq_hat = exp_avg_sq/bias_correction2
            # exp_avg_hat = exp_avg/bias_correction1
            # exp_avg_sq_hat = exp_avg_sq/bias_correction2
            # step_size = 1 # hardcode it to 1. The real learning rate is in SGD optimizer
            # adam_step = exp_avg_hat / exp_avg_sq_hat.add(eps).sqrt()

            # if weight_decay != 0:
            #     adam_step.add_(model_params, alpha=weight_decay)
            
            # # [NOTE] Add back non-topk momentum
            # if reducer != 'org':
            #     exp_avg.data.add_(exp_avg_nontopk)
            #     exp_avg_sq.data.add_(exp_avg_sq_nontopk)
            
            # self.tensor_corrected[dtype][start_idx:end_idx] = (step_size * adam_step).flatten()
            
            # w = w - lr * grad - lr * weight_decay * w, so need to recover it
            tensor[start_idx:end_idx] = (model_params-p_32[0]).flatten()    
        

    def merge_intervals(self, intervals):
        intervals.sort(key=lambda x: x[0])
        sorted_intervals = intervals
        merged_intervals = [sorted_intervals.pop(0)]
        for sorted_interval in sorted_intervals:
            prev_end = merged_intervals[-1][1]
            cur_start = sorted_interval[0]
            cur_end = sorted_interval[1]
            # 3 cases: 
            # prev_end < cur_start, 
            if prev_end < cur_start:
                merged_intervals.append(sorted_interval)
            # prev_end == cur_start, 
            elif prev_end == cur_start:
                merged_intervals[-1][1] = cur_end
            # prev_end > cur_start, 
            else:
                # two sub cases:
                # prev_end <= cur_end,
                # prev_end > cur_end
                if prev_end <= cur_end:
                    merged_intervals[-1][1] = cur_end
        return merged_intervals


    def resample_topk_intervals(self, dtype, start_end_idx_per_layer, tensor):
        intervals = []
        for layer_start_end_idx in start_end_idx_per_layer[dtype].values():
            layer_start_idx, layer_end_idx = layer_start_end_idx
            
            numel_cur_layer = layer_end_idx - layer_start_idx
            ratio = self.density
            k=max(int(numel_cur_layer*ratio), 1)
            _, topk_indices = torch.topk(tensor[layer_start_idx:layer_end_idx].abs(), k=k)
            topk_indices, _ = torch.sort(topk_indices)

            left_boundary = (topk_indices - self.stable_topk_range + layer_start_idx).reshape(-1,1)
            left_boundary[left_boundary<layer_start_idx] = layer_start_idx
            right_boundary = (topk_indices + self.stable_topk_range + layer_start_idx + 1).reshape(-1,1)
            right_boundary[right_boundary>layer_end_idx] = layer_end_idx
            intervals.append(torch.cat([left_boundary, right_boundary],dim=1))
        
        return intervals
    
    def resample_topk_indices(self, module, dtype, start_end_idx_per_layer, tensor, ratio):
        indices = []
        for layer_start_end_idx in start_end_idx_per_layer[dtype].values():
            layer_start_idx, layer_end_idx = layer_start_end_idx
            
            numel_cur_layer = layer_end_idx - layer_start_idx

            k=max(int(numel_cur_layer*ratio), 1)
            _, topk_indices = torch.topk(tensor[layer_start_idx:layer_end_idx].abs(), k=k)
            topk_indices.data.add_(layer_start_idx)
            topk_indices,_ = torch.sort(topk_indices)
            indices.append(topk_indices)

        # for i, (name, param) in enumerate(module.named_parameters()):
        #     if param.requires_grad:
        #         layer_start_idx, layer_end_idx = start_end_idx_per_layer[dtype][i]
        #         numel_cur_layer = layer_end_idx - layer_start_idx
        #         k=max(int(numel_cur_layer*ratio), 1)
        #         _, topk_indices = torch.topk((param/param.main_grad).abs().view(-1), k=k)
        #         topk_indices.data.add_(layer_start_idx)
        #         topk_indices,_ = torch.sort(topk_indices)
        #         indices.append(topk_indices)

        indices = torch.cat(indices)
        return indices

    def check_intersection_rate(self, dtype, indices, update_prev_indices=False):
        cur_indices= set(indices.cpu().numpy())
        if dtype in self.prev_grad_topk_indices:
            intersection = self.prev_grad_topk_indices[dtype].intersection(cur_indices)
            intersection_rate = len(intersection)/len(cur_indices)
            if dist.get_rank() == 0:
                print(f"Intersection rate: {intersection_rate}")
        dist.barrier()
        
        if update_prev_indices:
            self.prev_grad_topk_indices[dtype] = cur_indices

    def generate_indices_from_intervals(self,dtype):
        indices = []
        for interval in self.intervals[dtype]:
            start, end = interval
            indices.append(torch.arange(start,end,device=self.device))
        return torch.cat(indices)

    def warmup_density(self):
        return max(self.warmup_coeff ** (self.step + 1), self.density)
        

    def compress_org(self, 
                     module,
                     dtype,
                     layer_idx_,
                     start_end_idx_per_layer,
                     tensor):
        with torch.no_grad(): 

            # # Residuals here store the gradients, not momentums, nor the corrected momentums
            # if dtype not in self.residuals:
            #     self.residuals[dtype] = torch.zeros_like(tensor, dtype=dtype, device=self.device)
            if 'residuals' not in self.args.optimizer_var.state:
                self.args.optimizer_var.state['residuals'] = torch.zeros_like(tensor, dtype=dtype, device=self.device)
            elif self.args.optimizer_var.state['residuals'].get_device() != self.device:
                self.args.optimizer_var.state['residuals'] = self.args.optimizer_var.state['residuals'].to(self.device)
            # As a mask for residual selection
            if dtype not in self.zero_conditions:
                self.zero_conditions[dtype] = torch.ones_like(tensor,dtype=dtype, device=self.device)
            
            tensor.data.add_(self.args.optimizer_var.state['residuals'])

            ratio = self.warmup_density()
            self.indices[dtype] = self.resample_topk_indices(module, dtype, start_end_idx_per_layer, tensor, ratio)
            
            # initialize the residuals for later accumulation
            self.args.optimizer_var.state['residuals'][:]=tensor
            self.args.optimizer_var.state['residuals'].index_fill_(0,self.indices[dtype],0)

            # Perform all_gather to get the selected gradients and their indices from all workers
            selected_tensor = torch.index_select(tensor,0,self.indices[dtype])
            selected_tensor.div_(self.n_workers)
            indices_1d = [torch.zeros(len(self.indices[dtype]), dtype=torch.int64, device=self.device) for _ in range(self.n_workers)]
            values_1d = [torch.zeros(len(self.indices[dtype]), dtype=torch.float32, device=self.device) for _ in range(self.n_workers)]
            dist.all_gather(indices_1d, self.indices[dtype], group=self.group)
            dist.all_gather(values_1d, selected_tensor, group=self.group)
            
            # Mask for gradient correction
            self.zero_conditions[dtype].fill_(1.0)
            
            # Get the averaged tensor
            tensor.fill_(0.)
            for i, indices in enumerate(indices_1d):
                tensor[indices] += values_1d[i]
                self.zero_conditions[dtype].index_fill_(0,indices,0)
            
            if self.args.move_grad_clip_to_reducer:   
                tensor = self.gradient_clip(dtype, start_end_idx_per_layer, tensor)

            self.gradient_correction(dtype, layer_idx_, start_end_idx_per_layer, tensor, reducer='org')
            
            # if torch.distributed.get_rank() == 0 and dtype in self.indices and ratio != self.density:
            #     print(f"Real compression ratio: {len(self.indices[dtype])/tensor.numel():.6f}", flush=True)
            # dist.barrier()
            
            return tensor
           
    
    def reconstruct_grads(self, 
                          dtype, 
                          allreduce_results, 
                          grad_in_buffers, 
                          grad_out_buffers, 
                          grad_start_idx):
        '''
        Divide the combined gradients into each layer's gradient buffer.
        Update on the gradients will be performed after self.reduce has returned
        '''
        # Normally, only one type of data in gradients buffers
        if self.args.emb_comm_opt:
            emb_grads = grad_in_buffers[dtype].data[:grad_start_idx]
            grad_out_buffers[dtype].data = torch.cat(emb_grads, allreduce_results)
        else:
            grad_out_buffers[dtype].data = allreduce_results

    
    def reduce(self, 
               train_iter,
               module, 
               layer_idx_,
               start_end_idx_per_layer,
               grad_in_buffers, 
               grad_out_buffers, 
               grad_start_idx):
        """
        grad_in_buffers:  A list of continuous memory which contains all the grads from diff layers
                          No need to worry about mapping issues.
                          They have been taken care when param.main_grads are constructed.
                          **[NOTE] grad_in_buffer is the reverse of the layer order**
                          **I.e. grad_in_buffers = L_N, L_{N-1}, ..., L_2, L_1**
        grad_out_buffers: Similar to grad_in_buffers, but will not use it until warmup stage finishes.
                          **[NOTE] grad_out_buffer = L_1, L_2, ..., L_{N-1}, L_N**
        grad_start_idx:   Starting index of the gradient, excluding embedding layer grads.
        return: Whether need to do copy back from grad_in to grad_out or not.
        """
        if train_iter < self.stable_topk_threshold:
            # Collect topk grad distributions
            self._dense_allreduce(grad_in_buffers, grad_start_idx)
            # report whether in warmup stage or not every 100 iterations
            if train_iter % 100 == 0 and dist.get_rank() == 0:
                print(' >> Still in Warm-up Period... ', train_iter)
        else:    
            # Here we assume there are more than one type of data in grad_in_buffers.
            # Even though this is usually unlikely, we need to keep the integrity of the code.
            # [Finished] for dtype, buffer_ in grad_in_buffers.items():
            # all_grads = self.combine_all_grads(grad_in_buffers)
            
            for dtype, buffer_ in grad_in_buffers.items():
                if self.args.emb_comm_opt:
                    all_grads = buffer_.get_flat_to_end(grad_start_idx)
                else:
                    all_grads = buffer_.data
                
                processed_tensor = self.compress_org(module,
                                                         dtype, 
                                                         layer_idx_,
                                                         start_end_idx_per_layer,
                                                         all_grads)
                
            self.reconstruct_grads(dtype,
                                   processed_tensor,
                                   grad_in_buffers,
                                   grad_out_buffers,
                                   grad_start_idx)
        
        self.step += 1
        # if warmup ends, need to use reduced_buffer to update grad_buffer
        # no need to do so during the warmup stage, becuase grad_in_buffers points to models' grad tensor
        return train_iter >= self.stable_topk_threshold


@torch.jit.script
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col

@torch.jit.script
def add_error_feedback(t1, t2):
    torch.add(t1, t2, out=t1)

@torch.jit.script
def update_error_feedback(e, t, o):
    torch.add(t, o, alpha=(-1), out=e)

def all_reduce(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_reduce(*args, **kwargs)

def broadcast(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.broadcast(*args, **kwargs)

def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()
