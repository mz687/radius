ml pytorch/1.13.1

python3 ../../../plot_loss_curves/plot.py \
        --model "GPT-2.0B" \
        --plot_interval 10000 \
        --column_key 'ppl' \
        --output_dir /path/to/output/dir \
        --csv_files /path/to/csv/file \

python3 ../../../plot_loss_curves/plot.py \
        --model "GPT-2.0B" \
        --plot_interval 10000 \
        --column_key 'loss' \
        --output_dir /path/to/output/dir \
        --csv_files /path/to/csv/file \
