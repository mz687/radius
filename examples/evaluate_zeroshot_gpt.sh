#!/bin/bash

ml pytorch/1.13.1
export PYTHONPATH=/global/homes/m/mzheng/python_vir_envs/optimus-cc-torch-1.13.1/lib/python3.9/site-packages:$PYTHONPATH
export PATH=/global/homes/m/mzheng/python_vir_envs/optimus-cc-torch-1.13.1/bin:$PATH

WORLD_SIZE=4

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TASK="LAMBADA"

VALID_DATA=/global/homes/m/mzheng/data/lambada/lambada_validation.json
VOCAB_FILE=/global/homes/m/mzheng/optimus-cc/tools/gpt2-vocab.json
MERGE_FILE=/global/homes/m/mzheng/optimus-cc/tools/gpt2-merges.txt
CHECKPOINT=/pscratch/sd/m/mzheng/optimus-cc/optimus-cc_checkpoints/GPT2_345M_baseline/

torchrun $DISTRIBUTED_ARGS ../tasks/main.py \
               --experiment_name eval_GPT_345M_Lambada \
               --task $TASK \
               --valid-data $VALID_DATA \
               --tokenizer-type GPT2BPETokenizer \
               --strict-lambada \
               --vocab-file $VOCAB_FILE \
               --merge-file $MERGE_FILE \
               --load $CHECKPOINT \
               --tensor-model-parallel-size 1 \
               --num-layers 24 \
               --hidden-size 1024 \
               --num-attention-heads 16 \
               --micro-batch-size 8 \
               --activations-checkpoint-method uniform \
               --seq-length 1024 \
               --max-position-embeddings 1024 \
               --log-interval 10 \
               --bf16 \
               --no-load-optim \
               --no-load-rng
