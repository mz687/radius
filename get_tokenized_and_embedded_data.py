from megatron import print_rank_0
from megatron import get_args
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron import mpu
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.training import cyclic_iter
from megatron.utils import get_ltor_masks_and_position_ids
from megatron import get_tokenizer
from megatron.initialize import initialize_megatron
from megatron.model.language_model import Embedding
from megatron.model.utils import init_method_normal
from datetime import datetime
from tqdm import tqdm
import torch
import numpy
import zfpy
import sys,os
import gc


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    
    return tokens, labels, loss_mask, attention_mask, position_ids


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):

    args = get_args()

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_tensor_model_parallel_rank() == 0:

        # Number of train/valid/test samples.
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) * \
                     args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_samples,
                                      eval_iters * args.global_batch_size,
                                      test_iters * args.global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            train_val_test_num_samples)
        
        # Save the compressed tokenized data
        # if torch.distributed.get_rank() == 0:
        #     tokens = None
        #     for data in tqdm(train_ds):
        #         data = data['text']
        #         if tokens is None:
        #             tokens = data.reshape(1,-1)
        #         else:
        #             tokens = numpy.concatenate((tokens, data.reshape(1,-1)), axis=0)
        #         # print(f"tokens.shape: {tokens.shape}")
        #         if tokens.shape[0] >= 1024**2:
        #             break
        #     tokens_np_compressed = zfpy.compress_numpy(tokens)
        #     output_file = "/work/09308/zhengmk/optimus-cc/optimus-cc/jaeyong-song-Optimus-CC-6249c49/slurm_scrips/logs/345M/dense_dummy_TP1_PP1_DP4/compressed_tokens"
        #     with open(output_file, 'w') as f:
        #         f.writelines(tokens_np_compressed)
        #     print(f"Compressed tokens have been saved to {output_file}")
        # torch.distributed.barrier()
        # assert False

        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        valid_dataloader = build_pretraining_data_loader(
            valid_ds, args.consumed_valid_samples)
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()


    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']

    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(train_dataloader))
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader) if dl_type == 'single' \
                              else iter(cyclic_iter(valid_dataloader))
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader) if dl_type == 'single' \
                             else iter(cyclic_iter(test_dataloader))
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


if __name__ == '__main__':

    gc.collect()


    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
                        allow_no_cuda=True)

    print_rank_0("Start collecting tokenized data...")
    print_rank_0(f"Current time {datetime.now()}")
    args = get_args()
    args.iteration = 0

    train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider
    )

    output_dir = "/work/09308/zhengmk/optimus-cc/optimus-cc/jaeyong-song-Optimus-CC-6249c49/slurm_scrips/logs/345M/dense_dummy_TP1_PP1_DP4"
    output_tokens_compressed_file = os.path.join(output_dir, "compressed_tokens")
    output_embedded_token_compressed_file = os.path.join(output_dir, "compressed_embedded_tokens")
    output_token_numpy_file = os.path.join(output_dir,"tokens.npy")
    output_embedded_token_numpy_file = os.path.join(output_dir,"embedded_tokens.npy")
    
    # collect this number of tokens
    target_num_tokens = 2*1024*1024
    # correspondingly, we collect target_num_tokens*args.hidden_size numbers as embedded_res
    
    embedding_layer = Embedding(hidden_size=args.hidden_size, 
                                vocab_size=args.padded_vocab_size,
                                max_sequence_length=args.max_position_embeddings,
                                embedding_dropout_prob=args.hidden_dropout,
                                init_method=init_method_normal(args.init_method_std),
                                num_tokentypes=0).cuda()

    tokens_collected = None
    embedding_res_collected = None
    
    # Every rank collects its own tokens
    cur_total_examples = 0
    while args.iteration < args.train_iters:
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            train_data_iterator)
        
        # tokens.shape = (batch_size, max_sq_len)
        embedding_res = embedding_layer(tokens, position_ids)
        # embedding_res.shape = (batch_size, max_sq_len, embedding_vec_len)
        
        if cur_total_examples < target_num_tokens:
            if tokens_collected is None:
                # tokens_lists[rank] = tokens.cpu().detach().numpy().astype(numpy.int32)
                tokens_collected = tokens.cpu().detach().numpy().astype(numpy.int32)
            else:
                # tokens_lists[rank] = numpy.concatenate((tokens_lists[rank], tokens.cpu().detach().numpy()), axis=0, dtype=numpy.int32)
                tokens_collected = numpy.concatenate((tokens_collected, tokens.cpu().detach().numpy()), axis=0, dtype=numpy.int32)
            
            if embedding_res_collected is None:
              embedding_res_collected = embedding_res.cpu().detach().numpy().astype(numpy.float32)
            else:
              embedding_res_collected = numpy.concatenate((embedding_res_collected, embedding_res.cpu().detach().numpy()), axis=0, dtype=numpy.float32)
        else:
           break
        
        cur_total_examples += tokens.shape[0] * tokens.shape[1]
        args.iteration += 1
        print_rank_0(f"Iteration: {args.iteration}/{args.train_iters} ({cur_total_examples*100/target_num_tokens}%): cur_total_examples per rank={cur_total_examples}")        

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    # tokens_collected_lst = [torch.zeros_like(tokens_collected, device=tokens_collected.get_device())]
    # embedding_res_collected_lst = [torch.zeros_like(embedding_res_collected, device=embedding_res_collected.get_device())]
    
    # # gather collected tokens and embedded_tokens to rank 0
    # However, there is not enough memory...
    # torch.distributed.gather(tokens_collected, tokens_collected_lst, dst=0)
    # torch.distributed.gather(embedding_res_collected, embedding_res_collected_lst, dst=0)

    if rank == 0:
        tokens_np = None
        tokens_collected_lst = [tokens_collected]
        for tokens in tokens_collected_lst:
            if tokens_np is None:
                # tokens_np = tokens.cpu().detach().numpy().astype(numpy.int32)
                tokens_np = tokens
            else:
                tokens_np = numpy.concatenate((tokens_np,tokens),axis=0)
        
        embedding_res_np = None
        embedding_res_collected_lst = [embedding_res_collected]
        for embedding in embedding_res_collected_lst:
            if embedding_res_np is None:
                # embedding_res_np = embedding.cpu().detach().numpy().astype(numpy.int32)
                embedding_res_np = embedding
            else:
                embedding_res_np = numpy.concatenate((embedding_res_np,embedding),axis=0)
        
        # Save tokens collected to np file
        print(f"num of bytes for tokens_np: {sys.getsizeof(tokens_np)}")
        with open(output_token_numpy_file,"wb") as f:
            numpy.save(f, tokens_np)
        print(f"token numpy array has been saved to {output_token_numpy_file}")
        
        # Save compressed tokens to txt file
        tokens_np_compressed = zfpy.compress_numpy(tokens_np)
        print(f"num of bytes for tokens_np_compressed: {sys.getsizeof(tokens_np_compressed)}")
        print(f"Compression ratio for tokens: {sys.getsizeof(tokens_np_compressed)/sys.getsizeof(tokens_np)*100:.2f}%")
        with open(output_tokens_compressed_file, 'wb') as f:
            f.write(tokens_np_compressed)
        print(f"Compressed tokens have been saved to {output_tokens_compressed_file}\n")
        
        # Save embedded tokens collected to np file
        print(f"num of bytes for embedding_res_np: {sys.getsizeof(embedding_res_np)}")
        with open(output_embedded_token_numpy_file, 'wb') as f:
          numpy.save(f, embedding_res_np)
        print(f"embedded token numpy array has been saved to {output_embedded_token_numpy_file}")
        
        # Save compressed embedded tokens to txt file
        embedded_tokens_np_compressed = zfpy.compress_numpy(embedding_res_np)
        print(f"num of bytes for embedded_tokens_np_compressed: {sys.getsizeof(embedded_tokens_np_compressed)}")
        print(f"Compression ratio for embedded_tokens: {sys.getsizeof(embedded_tokens_np_compressed)/sys.getsizeof(embedding_res_np)*100:.2f}%")
        with open(output_embedded_token_compressed_file, 'wb') as f:
            f.write(embedded_tokens_np_compressed)
        print(f"Compressed tokens have been saved to {output_embedded_token_compressed_file}")
    
    torch.distributed.barrier()