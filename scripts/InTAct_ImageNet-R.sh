#!/bin/bash
#SBATCH --job-name=InTAct_KA_Prompt_ImageNet-R
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100

eval "$(conda shell.bash hook)"
conda activate DIL

OUTDIR=output/InTAct_KA-Prompt/ImageNet-R/
epochs=5

# InTAct hyperparameters (from CODA-Prompt reference on ImageNet-R 15 tasks)
VAR_LOSS_SCALE=0.1
INTERNAL_REPR_DRIFT_LOSS_SCALE=0.001
FEATURE_LOSS_SCALE=0.1

for seed in 3407 0 42; do
    echo "Running Python script with seed: $seed"

    CUDA_VISIBLE_DEVICES=0 python main.py --info=seed-${seed} --dataset=imagenet-r \
    --pool_size=225 --prompt_num=4 --topN=2 --prompt_comp --prompt_per_task=15 --use_prompt_penalty_3 \
    --output_path=${OUTDIR} --fuse_prompt --use_ema_c --adapt_ema_c --adapt_h=6.5 --lr=0.007 \
    --resize_test=224 --resize_train=224 --greedy_init --tau 0.01 --aux_weight 0.1 --seed=${seed} \
    --query_pos -1 --n_epochs=${epochs} \
    --use_interval_activation \
    --var_loss_scale ${VAR_LOSS_SCALE} \
    --internal_repr_drift_loss_scale ${INTERNAL_REPR_DRIFT_LOSS_SCALE} \
    --feature_loss_scale ${FEATURE_LOSS_SCALE} \
    --use_align_loss
done

python cal_mean_std_af.py ${OUTDIR}
