ml pytorch/1.13.1

PARENT_PATH=/path/to/output/dir
TIME=$PARENT_PATH/time

mkdir -p $TIME
python3 ../../../plot_loss_curves/plot_time.py \
        --model "GPT-2.0B" \
        --column_key 'ppl' \
        --mode train \
        --output_dir  $TIME \
        --switch_iter 60000 \
        --csv_files /path/to/csv/files \


 