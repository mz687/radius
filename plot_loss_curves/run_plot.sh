ml pytorch/1.13.1
export PYTHONPATH=/global/homes/m/mzheng/python_vir_envs/optimus-cc-torch-1.13.1/lib/python3.9/site-packages:$PYTHONPATH
export PATH=/global/homes/m/mzheng/python_vir_envs/optimus-cc-torch-1.13.1/bin:$PATH

interval=100

python3 plot.py \
        --model "GPT2 345M" \
        --plot_interval 10000 \
        --column_key 'loss' \
        --output_dir /global/homes/m/mzheng/optimus-cc/plot_loss_curves/logs/345M/StableTopKReducerWithRangeBucketsEFCorrectionResIsGrad/lr_1.5e-4_density_0.01_range_50_interval_$interval \
        --csv_files /global/homes/m/mzheng/optimus-cc/plot_loss_curves/logs/345M/dense/log_PP1_TP1_DP8.csv \
        --csv_files /global/homes/m/mzheng/optimus-cc/plot_loss_curves/logs/345M/StableTopKReducerWithRangeBucketsEFCorrectionResIsGrad/lr_1.5e-4_density_0.01_range_50_interval_$interval/log_PP1_TP1_DP8.csv \
        
        # # For plotting the baseline only 
        # --model "GPT2 345M" \
        # --plot_interval 100000 \
        # --column_key 'loss' \
        # --output_dir /global/homes/m/mzheng/optimus-cc/plot_loss_curves/logs/345M/dense/full_training_log \
        # --csv_files /global/homes/m/mzheng/optimus-cc/plot_loss_curves/logs/345M/dense/full_training_log/log_PP1_TP1_DP8.csv \
        
        