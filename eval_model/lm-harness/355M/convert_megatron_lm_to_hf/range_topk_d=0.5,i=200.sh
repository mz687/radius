CHECKPOINT_PATH=

python3 ./transformers/src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py \
        --path_to_checkpoint $CHECKPOINT_PATH/model_optim_rng.pt \