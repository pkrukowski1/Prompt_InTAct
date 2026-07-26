#!/bin/bash
#SBATCH --job-name=KA_DomainNet_patryk
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100
#SBATCH --array=0-11

eval "$(conda shell.bash hook)"
conda activate DIL

VAR_LOSS_SCALES=("0.001" "0.01" "0.1" "1.0")
INTERNAL_REPRESENTATION_DRIFT_REG_SCALES=("0.0001" "0.001" "0.1")
FEATURE_LOSS_SCALES=("0.0001" "0.001" "0.1")

OUTDIR=/shared/results/common/helm/KA-Prompt-InTAct/DomainNet/M/
EPOCHS=10

FEATURE_LOSS_SCALE=${FEATURE_LOSS_SCALES[0]}

COMBINATIONS=()
for var in "${VAR_LOSS_SCALES[@]}"; do
  for out in "${INTERNAL_REPRESENTATION_DRIFT_REG_SCALES[@]}"; do
    COMBINATIONS+=("${var}_${out}")
  done
done

IFS='_' read -r VAR_LOSS INTERNAL_DRIFT <<< "${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: var=${VAR_LOSS} drift=${INTERNAL_DRIFT} feat=${FEATURE_LOSS_SCALE}"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --info=var${VAR_LOSS}_out${INTERNAL_DRIFT}_drift${FEATURE_LOSS_SCALE} \
    --dataset=domain-net \
    --pool_size=150 \
    --prompt_num=8 \
    --topN=3 \
    --prompt_comp \
    --prompt_per_task=25 \
    --use_prompt_penalty_3 \
    --output_path=${OUTDIR} \
    --fuse_prompt \
    --use_ema_c \
    --adapt_ema_c \
    --adapt_h=10 \
    --lr=0.0006 \
    --resize_test=224 \
    --resize_train=224 \
    --greedy_init \
    --tau 0.01 \
    --aux_weight 0.1 \
    --seed=0 \
    --query_pos -1 \
    --n_epochs=${EPOCHS} \
    --use_interval_activation \
    --var_loss_scale ${VAR_LOSS} \
    --internal_repr_drift_loss_scale ${INTERNAL_DRIFT} \
    --feature_loss_scale ${FEATURE_LOSS_SCALE} \
    --use_align_loss \
    --use_intact_metrics