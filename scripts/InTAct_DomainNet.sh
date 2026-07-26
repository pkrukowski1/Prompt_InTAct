#!/bin/bash
#SBATCH --job-name=InTAct_KA_Prompt_DomainNet
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100

eval "$(conda shell.bash hook)"
conda activate DIL

method=InTAct_KA-Prompt/

# InTAct hyperparameters (from CODA-Prompt reference on DomainNet)
VAR_LOSS_SCALE=1.0
INTERNAL_REPR_DRIFT_LOSS_SCALE=0.001
FEATURE_LOSS_SCALE=0.0001

for seed in 0 42 3407; do
    echo "Running Python script with seed: $seed"

    CUDA_VISIBLE_DEVICES=0 python main.py --info=seed-${seed} --dataset=domain-net \
    --pool_size=150 --prompt_num=8 --topN=3 --prompt_comp --prompt_per_task=25 --use_prompt_penalty_3 \
    --fuse_prompt --use_ema_c --output_path=output/${method}/DomainNet/ --adapt_ema_c --adapt_h=10 --lr=0.0006 \
    --resize_test=224 --resize_train=224 --greedy_init --tau 0.01 --aux_weight 0.1 --seed=${seed} --n_epochs 10 \
    --query_pos -1 \
    --use_interval_activation \
    --var_loss_scale ${VAR_LOSS_SCALE} \
    --internal_repr_drift_loss_scale ${INTERNAL_REPR_DRIFT_LOSS_SCALE} \
    --feature_loss_scale ${FEATURE_LOSS_SCALE} \
    --use_align_loss
done

python cal_mean_std_af.py output/${method}/DomainNet/
