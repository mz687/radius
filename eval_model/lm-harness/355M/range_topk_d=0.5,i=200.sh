RANGE_TOPK_PATH=
OUTPUT_PATH=
lm-eval --model hf \
    --model_args pretrained=$RANGE_TOPK_PATH \
    --tasks lambada,race,winogrande,wikitext,squad_completion,squadv2,mathqa,piqa,glue,super-glue-lm-eval-v1 \
    --batch_size 64 \
    --output_path $OUTPUT_PATH \
    --device cuda:1 \
    --trust_remote_code \