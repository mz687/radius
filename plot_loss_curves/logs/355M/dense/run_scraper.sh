# activate virtual env
ml pytorch/1.13.1

python3 ../../../plot_loss_curves/scraper.py \
        --log_files /path/to/log \
        --output_dir /path/to/output/dir \
        --mode 'train'