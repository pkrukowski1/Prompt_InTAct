#!/bin/bash
#SBATCH --job-name=run_ImageNet-R
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100
#SBATCH --array=0-5

# Final runs on the ImageNet-R test lists: tasks 0-2 KA-Prompt, tasks 3-5 KA-Prompt + InTAct (seeds 3407, 0, 42).
# Results: python tools/collect_results.py $OUTDIR

# best configuration of grid_imagenet_r.sh (tools/collect_results.py prints this line)
VAR=""; DRIFT=""; FEAT=""; ALIGN=""; EMA=1
SEEDS=(3407 0 42)
OUTDIR=/shared/results/common/helm/KA-Prompt-InTAct/v2/ImageNet-R/test

# KA-Prompt settings from scripts/ImageNet-R.sh
KA="--dataset=imagenet-r --pool_size=225 --prompt_num=4 --topN=2 --prompt_comp --prompt_per_task=15 --use_prompt_penalty_3 --fuse_prompt --lr=0.007 --resize_test=224 --resize_train=224 --greedy_init --tau 0.01 --aux_weight 0.1 --query_pos -1 --n_epochs=5"
EMA_FLAGS="--use_ema_c --adapt_ema_c --adapt_h=6.5"
N_TASKS=15

eval "$(conda shell.bash hook)"
conda activate DIL
[ -f main.py ] || { echo "submit from the repository root"; exit 1; }

ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch, or set SLURM_ARRAY_TASK_ID}
N=${#SEEDS[@]}
[ "$ID" -lt $((2 * N)) ] || { echo "task $ID: this script has $((2 * N)) tasks"; exit 0; }
SEED=${SEEDS[$((ID % N))]}
if [ "$ID" -lt "$N" ]; then
  NAME=ka; FLAGS="$KA $EMA_FLAGS"
else
  [ -n "$VAR" ] && [ -n "$DRIFT" ] && [ -n "$FEAT" ] && [ -n "$ALIGN" ] || { echo "task $ID: set VAR, DRIFT, FEAT, ALIGN in experiments/run_imagenet_r.sh"; exit 1; }
  INTACT="--use_interval_activation --var_loss_scale $VAR --internal_repr_drift_loss_scale $DRIFT --feature_loss_scale $FEAT --use_align_loss --align_loss_scale $ALIGN --use_intact_metrics"
  if [ "$EMA" == "1" ]; then VARIANT=intact_ema; FLAGS="$KA $EMA_FLAGS $INTACT --intact_anchor ema"
  else VARIANT=intact_noema; FLAGS="$KA $INTACT --intact_anchor raw"; fi
  NAME=${VARIANT}_var${VAR}_drift${DRIFT}_feat${FEAT}_align${ALIGN}
fi

RUN=$OUTDIR/$NAME/seed-$SEED
if [ -f "$RUN/results/$((N_TASKS - 1)).txt" ]; then echo "already finished: $RUN"; exit 0; fi
echo "task $ID: $NAME, seed $SEED"
KA_SPLIT="official" python main.py --info="seed-$SEED" --seed="$SEED" --output_path="$OUTDIR/$NAME" $FLAGS
