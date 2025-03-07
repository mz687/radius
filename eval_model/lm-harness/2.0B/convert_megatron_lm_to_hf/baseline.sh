python3 ./transformers/src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py \
        --num_layers 24 \
        --hidden_size 2560 \
        --num_attention_heads 32 \
        --max_position_embedding 1024 \
        --path_to_checkpoint ./GPT2_2.0B_baseline_TP1_PP1/iter_0300000/mp_rank_00/model_optim_rng.pt \