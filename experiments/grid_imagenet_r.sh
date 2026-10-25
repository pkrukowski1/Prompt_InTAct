#!/bin/bash
#SBATCH --job-name=grid_ImageNet-R
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100
#SBATCH --array=0-26

# Grid search over the InTAct hyperparameters on the ImageNet-R validation split (90/10 split of the train lists,
# python tools/make_val_lists.py once). One task per configuration; finished runs are skipped.
# Ranking: python tools/collect_results.py $OUTDIR

VARS="0.01 0.1 1.0"           # lambda_Var
DRIFTS="0.0001 0.001 0.01"    # lambda_IntDrift
FEATS="0 0.01 0.1"            # lambda_Feat
ALIGNS="1.0"                  # lambda_Align
EMA=1                         # 1: keep KA-Prompt's EMA classifier, 0: without it
SEED=3407
OUTDIR=/shared/results/common/helm/KA-Prompt-InTAct/v2/ImageNet-R/val

# KA-Prompt settings from scripts/ImageNet-R.sh
KA="--dataset=imagenet-r --pool_size=225 --prompt_num=4 --topN=2 --prompt_comp --prompt_per_task=15 --use_prompt_penalty_3 --fuse_prompt --lr=0.007 --resize_test=224 --resize_train=224 --greedy_init --tau 0.01 --aux_weight 0.1 --query_pos -1 --n_epochs=5"
EMA_FLAGS="--use_ema_c --adapt_ema_c --adapt_h=6.5"
N_TASKS=15

eval "$(conda shell.bash hook)"
conda activate DIL
[ -f main.py ] || { echo "submit from the repository root"; exit 1; }
[ -d datasets/image_list/imagenet-r_val ] || { echo "run python tools/make_val_lists.py first"; exit 1; }

CONFIGS=()
for var in $VARS; do for drift in $DRIFTS; do for feat in $FEATS; do for align in $ALIGNS; do
  CONFIGS+=("$var $drift $feat $align")
done; done; done; done
ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch, or set SLURM_ARRAY_TASK_ID}
[ "$ID" -lt ${#CONFIGS[@]} ] || { echo "task $ID: the grid has ${#CONFIGS[@]} configurations"; exit 0; }
read -r VAR DRIFT FEAT ALIGN <<< "${CONFIGS[$ID]}"

INTACT="--use_interval_activation --var_loss_scale $VAR --internal_repr_drift_loss_scale $DRIFT --feature_loss_scale $FEAT --use_align_loss --align_loss_scale $ALIGN"
if [ "$EMA" == "1" ]; then VARIANT=intact_ema; FLAGS="$KA $EMA_FLAGS $INTACT --intact_anchor ema"
else VARIANT=intact_noema; FLAGS="$KA $INTACT --intact_anchor raw"; fi
NAME=${VARIANT}_var${VAR}_drift${DRIFT}_feat${FEAT}_align${ALIGN}

RUN=$OUTDIR/$NAME/seed-$SEED
if [ -f "$RUN/results/$((N_TASKS - 1)).txt" ]; then echo "already finished: $RUN"; exit 0; fi
echo "task $ID: $NAME, seed $SEED"
KA_SPLIT="val" python main.py --info="seed-$SEED" --seed="$SEED" --output_path="$OUTDIR/$NAME" $FLAGS
