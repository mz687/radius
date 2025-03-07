python3 ./transformers/src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py \
        --num_layers 24 \
        --hidden_size 2560 \
        --num_attention_heads 32 \
        --max_position_embedding 1024 \
        --path_to_checkpoint ./GPT2_2.0B_reducer_StableTopKReducerWithRangeBucketsEFCorrectionResIsGrad_no_set_nontopk_momemtum_zero_stable/TP1_PP1_lr_0.00015_min_lr_1.0e-6_density_0.1_range_0_update_interval_200_warmup_method_Dense_warmup_threshold_100/start_from_60000/iter_0300000/mp_rank_00/model_optim_rng.pt \