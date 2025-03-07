BASELINE_PATH=

python3 ./transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py \
        --num_layers 24 \
        --hidden_size 1024 \
        --num_attention_heads 16 \
        --max_position_embeddings 1024 \
        --path_to_checkpoint $BASELINE_PATH/iter_0500000/mp_rank_00/model_optim_rng.pt \