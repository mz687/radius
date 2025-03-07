#!/bin/bash
#SBATCH --account ACCOUNT
#SBATCH --qos regular
#SBATCH --nodes 4
#SBATCH --constraint gpu
#SBATCH --gpus 16 # every perlmutter node has 4 A100 40GB
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -c 32 # https://docs.nersc.gov/systems/perlmutter/running-jobs/
#SBATCH --output rtx-%J.out
#SBATCH --error  rtx-%J.out
#SBATCH --mail-user=YOUR_EMAIL_ADDRESS
#SBATCH --mail-type=all

# activate virtual env
ml darshan
ml pytorch/1.13.1 

echo "SLURM_NNODES=${SLURM_NNODES}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"

GPUS_PER_NODE=$SLURM_NTASKS_PER_NODE
NNODES=$SLURM_NNODES

export MASTER_PORT=1234
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR

# Change for multinode config
TENSOR_PARALLEL=1
PIPELINE_PARALLEL=1
DATA_PARALLEL=$(($WORLD_SIZE/$PIPELINE_PARALLEL/$TENSOR_PARALLEL))

echo "TENSOR_PARALLEL=${TENSOR_PARALLEL}"
echo "PIPELINE_PARALLEL=${PIPELINE_PARALLEL}"
echo "DATA_PARALLEL=${DATA_PARALLEL}"

DATA_PATH=<Specify path and file prefix>_text_document
CHECKPOINT_PATH=/path/to/the/checkpoints

output_dir=/path/to/the/generated/logs
plot_gradients_distribution_output_dir=/path/to/where/gradients/plots/are/saved
mkdir -p  $output_dir
mkdir -p  $plot_gradients_distribution_output_dir

global_batch_size=512
echo "global_batch_size=${global_batch_size}"

srun python pretrain_gpt.py \
       --tensor-model-parallel-size $TENSOR_PARALLEL \
       --pipeline-model-parallel-size $PIPELINE_PARALLEL \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 64 \
       --global-batch-size $global_batch_size \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file ./tools/gpt2-vocab.json \
       --merge-file ./tools/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --activations-checkpoint-method uniform \
       --log-interval 1 \
       --save-interval 100000 \
       --eval-interval 100 \
       --eval-iters 10 \
       --experiment_name GPT_345M_Baseline_plot_gradient_distribution \
       --accumulate-allreduce-grads-in-fp32 \
       --bf16 \
       --log_gradients \
       --log_gradients_interval 100 \
       --log_gradients_output_dir $plot_gradients_distribution_output_dir \