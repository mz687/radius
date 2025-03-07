ml pytorch/1.13.1

python3 ../../../plot_loss_curves/plot.py \
        --model "GPT2 345M" \
        --plot_interval 10000 \
        --column_key 'loss' \
        --output_dir /path/to/where/plots/are/saved \
        --csv_files /path/to/the/csv/file/dumped/by/scraper.py \
        