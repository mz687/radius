python preprocess_data.py \
       --input /path/to/openwebtext/dataset/openwebtxt.json \
       --json-keys 'text' \
       --tokenizer-type 'GPT2BPETokenizer' \
       --vocab-file ./tools/gpt2-vocab.json \
       --output-prefix ./openwebtxt_huggingface_json/preprocessed_data \
       --dataset-impl 'mmap' \
       --merge-file ./tools/gpt2-merges.txt \
       --workers 48 \
       --append-eod \

