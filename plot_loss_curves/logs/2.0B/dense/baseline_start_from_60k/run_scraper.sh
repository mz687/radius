# activate virtual env
ml pytorch/1.13.1
export PYTHONPATH=/global/homes/m/mzheng/python_vir_envs/optimus-cc-torch-1.13.1/lib/python3.9/site-packages:$PYTHONPATH
export PATH=/global/homes/m/mzheng/python_vir_envs/optimus-cc-torch-1.13.1/bin:$PATH

# output_dir=/global/homes/m/mzheng/optimus-cc/plot_loss_curves/logs/345M/dense/full_training_log
# mkdir -p $output_dir

interval=100

python3 /global/homes/m/mzheng/optimus-cc/plot_loss_curves/scraper.py \
        --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/baseline/2.0B/16nodes/rtx-30007628.out \
        --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/baseline/2.0B/16nodes/rtx-30045468.out \
        --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/baseline/2.0B/16nodes/rtx-30131487.out \
        --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/baseline/2.0B/16nodes/rtx-30263166.out \
        --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/baseline/2.0B/16nodes/rtx-30369332.out \
        --output_dir /global/homes/m/mzheng/optimus-cc/plot_loss_curves/logs/2.0B/dense/baseline_start_from_60k \
        --mode 'validation'
        

        # resample interval = 10
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/rtx-24603970.out \
        # --log_files  \
        # resample interval = 20
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/rtx-24593514.out \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/rtx-24622892.out \
        # resample interval = 50
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/rtx-24593485.out \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/rtx-24605697.out \
        # resample interval = 100
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/rtx-24593464.out \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/rtx-24603607.out \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/rtx-24616952.out \
        
        # baseline 
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/rtx-21759296.out \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log.txt\
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1708712841.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1709688478.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1709781620.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1710784398.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1710861501.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1711071075.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1711242140.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1711308899.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1711476255.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1711615602.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1711665252.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1712389456.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1712916366.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1713573319.txt \
        # --log_files /global/homes/m/mzheng/optimus-cc/slurm_scrips/perlmutter_scripts/logs/345M/dense_TP1_PP1_DP8/log_1713631191.txt \
        # --output_dir $output_dir \
