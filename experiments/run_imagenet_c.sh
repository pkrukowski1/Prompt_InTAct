#!/bin/bash
#SBATCH --job-name=KA_ImN-C_full
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100
#SBATCH --array=0-35

eval "$(conda shell.bash hook)"
conda activate DIL

SCRIPT_DIR=$(dirname "$0")
source ${SCRIPT_DIR}/grid_config.sh

DATASET=imagenet-c
DATASET_DIR=ImageNet-C
OUTDIR=/shared/results/common/helm/KA-Prompt-InTAct/${DATASET_DIR}/M/
EPOCHS=5

COMBINATIONS=()
for var in "${VAR_LOSS_SCALES[@]}"; do
  for out in "${INTERNAL_REPRESENTATION_DRIFT_REG_SCALES[@]}"; do
    for feat in "${FEATURE_LOSS_SCALES[@]}"; do
      COMBINATIONS+=("${var}_${out}_${feat}")
    done
  done
done

IFS='_' read -r VAR_LOSS INTERNAL_DRIFT FEATURE_LOSS <<< "${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: var=${VAR_LOSS} drift=${INTERNAL_DRIFT} feat=${FEATURE_LOSS}"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --info=var${VAR_LOSS}_out${INTERNAL_DRIFT}_drift${FEATURE_LOSS} \
    --dataset=${DATASET} \
    --pool_size=225 \
    --prompt_num=4 \
    --topN=2 \
    --prompt_comp \
    --prompt_per_task=15 \
    --use_prompt_penalty_3 \
    --output_path=${OUTDIR} \
    --fuse_prompt \
    --use_ema_c \
    --adapt_ema_c \
    --adapt_h=6.5 \
    --lr=0.007 \
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
    --feature_loss_scale ${FEATURE_LOSS} \
    --use_align_loss \
    --use_intact_metrics