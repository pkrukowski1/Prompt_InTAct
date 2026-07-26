#!/bin/bash
#SBATCH --job-name=CODA-P_domainnet_dil_piotrek
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100
#SBATCH --array=0-11

source activate prompt_intact

SCRIPT_DIR=$(dirname "$0")
source ${SCRIPT_DIR}/grid_config.sh

# experiment settings
DATASET=DomainNet
N_CLASS=345

# save directory
OUTDIR=./${DATASET}/6-task

# hard coded inputs
GPUID='0'
CONFIG=configs/domainnet_prompt_dil.yaml
REPEAT=1
OVERWRITE=1

###############################################################

FEATURE_LOSS_SCALE=${FEATURE_LOSS_SCALES[2]}

# Create combined array of all combinations
COMBINATIONS=()
for var in "${VAR_LOSS_SCALES[@]}"; do
  for out in "${INTERNAL_REPRESENTATION_DRIFT_REG_SCALES[@]}"; do
    COMBINATIONS+=("${var}_${out}")
  done
done

# Get current combination from array task
IFS='_' read -r VAR_LOSS INTERNAL_DRIFT <<< "${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"

LOGDIR=${OUTDIR}/coda-p/var${VAR_LOSS}_out${INTERNAL_DRIFT}_drift${FEATURE_LOSS_SCALE}
mkdir -p $LOGDIR

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
  --learner_type prompt --learner_name CODAPrompt \
  --prompt_param 100 8 0.0 \
  --use_interval_activation \
  --log_dir $LOGDIR \
  --var_loss_scale $VAR_LOSS \
  --internal_repr_drift_loss_scale $INTERNAL_DRIFT \
  --feature_loss_scale $FEATURE_LOSS_SCALE \
  --use_align_loss