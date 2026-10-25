#!/bin/bash
#SBATCH --job-name=grid_noema_ImageNet-Mix
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100
#SBATCH --array=0-26
#SBATCH --output=/shared/results/pkrukowski/InTAct/KA-Prompt/slurm-%x-%A_%a.out

# KA-Prompt without its EMA classifier, so only InTAct protects the classifier. Same grid and output folder
# as grid_imagenet_cr.sh, so both variants are ranked together by tools/collect_results.py.
# Grid search over the InTAct hyperparameters on the official ImageNet-Mix train/test lists, the split of
# KA-Prompt and of run_imagenet_cr_noema.sh, which reuses these runs as its seed 3407. One task per configuration.
# A run that already has results is skipped, never overwritten. Each task's SLURM log is moved into its run
# folder, $OUTDIR/<configuration>/seed-<seed>.
# Ranking: python tools/collect_results.py $OUTDIR

VARS="0.01 0.1 1.0"           # lambda_Var
DRIFTS="0.0001 0.001 0.01"    # lambda_IntDrift
FEATS="0 0.01 0.1"            # lambda_Feat
ALIGNS="1.0"                  # lambda_Align
EMA=0                         # KA-Prompt's EMA classifier off
SEED=3407
OUTDIR=/shared/results/pkrukowski/InTAct/KA-Prompt/ImageNet-Mix/test

# KA-Prompt settings from scripts/ImageNet-Mix.sh
KA="--dataset=imagenet-cr --pool_size=750 --prompt_num=4 --topN=3 --prompt_comp --prompt_per_task=25 --use_prompt_penalty_3 --fuse_prompt --lr=0.005 --resize_test=224 --resize_train=224 --greedy_init --tau 0.01 --aux_weight 1.0 --query_pos -1 --n_epochs=5"
EMA_FLAGS="--use_ema_c --adapt_ema_c --adapt_h=8.7"
N_TASKS=30

eval "$(conda shell.bash hook)"
conda activate intact_with_ka_prompt
[ -f main.py ] || { echo "submit from the repository root"; exit 1; }

CONFIGS=()
for var in $VARS; do for drift in $DRIFTS; do for feat in $FEATS; do for align in $ALIGNS; do
  CONFIGS+=("$var $drift $feat $align")
done; done; done; done
ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch, or set SLURM_ARRAY_TASK_ID}
[ "$ID" -lt ${#CONFIGS[@]} ] || { echo "task $ID: the grid has ${#CONFIGS[@]} configurations"; exit 0; }
read -r VAR DRIFT FEAT ALIGN <<< "${CONFIGS[$ID]}"

INTACT="--use_interval_activation --var_loss_scale $VAR --internal_repr_drift_loss_scale $DRIFT --feature_loss_scale $FEAT --use_align_loss --align_loss_scale $ALIGN --use_intact_metrics"
if [ "$EMA" == "1" ]; then VARIANT=intact_ema; FLAGS="$KA $EMA_FLAGS $INTACT --intact_anchor ema"
else VARIANT=intact_noema; FLAGS="$KA $INTACT --intact_anchor raw"; fi
NAME=${VARIANT}_var${VAR}_drift${DRIFT}_feat${FEAT}_align${ALIGN}

RUN=$OUTDIR/$NAME/seed-$SEED
# SLURM opens the log in the folder of #SBATCH --output; keep it in the run folder instead
LOG=/shared/results/pkrukowski/InTAct/KA-Prompt/slurm-${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
mkdir -p "$RUN" && [ -f "$LOG" ] && mv -n "$LOG" "$RUN/"
# never overwrite a run folder that already has results
DONE=$(find "$RUN/results" -maxdepth 1 -name '[0-9]*.txt' 2>/dev/null | wc -l)
if [ "$DONE" -ge "$N_TASKS" ]; then echo "already finished: $RUN"; exit 0; fi
if [ "$DONE" -gt 0 ]; then echo "$RUN already has results of $DONE/$N_TASKS tasks; not overwriting it (delete it to rerun)"; exit 0; fi
echo "task $ID: $NAME, seed $SEED"
KA_SPLIT="official" python main.py --info="seed-$SEED" --seed="$SEED" --output_path="$OUTDIR/$NAME" $FLAGS
