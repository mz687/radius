# shifter --image=docker:nvcr.io/nvidia/pytorch:24.01-py3 --module=gpu,nccl-plugin bash

# use `PYTHONUSERBASE=/global/homes/m/mzheng/python_vir_envs/shifter-nemo pip install --user package_name`


export PATH=/global/homes/m/mzheng/python_vir_envs/shifter-nemo/bin:$PATH
export PYTHONPATH=/global/u2/m/mzheng/EleutherAI/lm-evaluation-harness:/global/homes/m/mzheng/megatron-lm-core/Megatron-LM:/usr/local/lib/python3.10/dist-packages:/global/homes/m/mzheng/python_vir_envs/shifter-nemo/lib/python3.10/site-packages:$PYTHONPATH
export TMPDIR=/tmp

# MEGATRON_PATH=/pscratch/sd/m/mzheng/optimus-cc/optimus-cc_checkpoints/GPT2_345M_baseline/iter_0500000/mp_rank_00/model_optim_rng.pt
# CONVERTED_MEGATRON_PATH=/pscratch/sd/m/mzheng/optimus-cc/optimus-cc_checkpoints/GPT2_345M_baseline/iter_0500000/mp_rank_00/pytorch_model.bin
# NEMO_PATH=/global/homes/m/mzheng/optimus-cc/eval_model/convert_megatron-lm_to_nemo/GPT_355M.nemo

BASELINE_PATH=/pscratch/sd/m/mzheng/optimus-cc-checkpoints-finished/2.0B/baseline/hf/
OUTPUT_PATH=/global/homes/m/mzheng/optimus-cc/eval_model/lm-harness/2.0B/results/baseline

# lambada,race,winogrande,wikitext,squad_completion,squadv2,mathqa,piqa,glue,super-glue-lm-eval-v1 \
lm-eval --model hf \
    --model_args pretrained=$BASELINE_PATH \
    --tasks lambada,race,winogrande,wikitext,squad_completion,squadv2,mathqa,piqa,glue,super-glue-lm-eval-v1 \
    --batch_size 64 \
    --output_path $OUTPUT_PATH \
    --device cuda:0 \
    --trust_remote_code \
    # --model_args path=/pscratch/sd/m/mzheng/nemo_checkpoints/GPT/355M/baseline/GPT_355M.nemo,num_layers=24,hidden_size=1024,ffn_hidden_size=4096,num_attention_head=16,encoder_seq_length=1024,max_position_embeddings=1024 \
