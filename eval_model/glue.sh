ml pytorch/1.13.1

lm_eval --model nemo_lm \
    --model_args path=/path/to/the/pretrained/model/model_optim_rng.pt \
    --tasks glue \
    --batch_size 128