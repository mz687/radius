ml pytorch/1.13.1

BASELINE_TRAIN=/path/to/baseline/train/csv/file
BASELINE_VAL=/path/to/baseline/val/csv/file

PARENT_PATH=$PWD
TRAIN_CSV=$PARENT_PATH/csv_file
VAL_CSV=$PARENT_PATH/csv_file

TRAIN_PATH=$PARENT_PATH/train
mkdir -p $TRAIN_PATH

TRAIN_LOSS=$TRAIN_PATH/loss
mkdir -p $TRAIN_LOSS
python3 ../../../plot_loss_curves/plot.py \
        --model "GPT-355M" \
        --plot_interval 10000 \
        --column_key 'loss' \
        --mode train \
        --output_dir $TRAIN_LOSS \
        --csv_files $TRAIN_CSV \
        --csv_files $BASELINE_TRAIN \

TRAIN_PPL=$TRAIN_PATH/ppl
mkdir -p $TRAIN_PPL
python3 ../../../plot_loss_curves/plot.py \
        --model "GPT-355M" \
        --plot_interval 10000 \
        --column_key 'ppl' \
        --mode train \
        --output_dir $TRAIN_PPL \
        --csv_files $TRAIN_CSV \
        --csv_files $BASELINE_TRAIN \

VAL_PATH=$PARENT_PATH/val
mkdir -p $VAL_PATH

VAL_LOSS=$VAL_PATH/loss
mkdir -p $VAL_LOSS
python3 ../../../plot_loss_curves/plot.py \
        --model "GPT-355M" \
        --plot_interval 10000 \
        --column_key 'loss' \
        --mode validation \
        --output_dir $VAL_LOSS \
        --csv_files $VAL_CSV \
        --csv_files $BASELINE_VAL \

VAL_PPL=$VAL_PATH/ppl
mkdir -p $VAL_PPL
python3 ../../../plot_loss_curves/plot.py \
        --model "GPT-355M" \
        --plot_interval 10000 \
        --column_key 'ppl' \
        --mode validation \
        --output_dir $VAL_PPL \
        --csv_files $VAL_CSV \
        --csv_files $BASELINE_VAL \