#!/bin/bash
#SBATCH --job-name=grid_raw_ImageNet-R
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=rtx4090_batch
#SBATCH --array=0-17
#SBATCH --output=/shared/results/pkrukowski/InTAct/KA-Prompt/slurm-%x-%A_%a.out

VARS="0.01 0.1 1.0"           # lambda_Var
DRIFTS="0.0001 0.001 0.01"    # lambda_IntDrift
FEATS="0.01 0.1"              # lambda_Feat
ALIGNS="1.0"                  # lambda_Align
EMA=1                         # KA-Prompt's EMA classifier on
ANCHOR=raw                    # L_IntDrift anchored to the trained classifier, not the EMA one
SEED=3407
OUTDIR=/shared/results/pkrukowski/InTAct/KA-Prompt/ImageNet-R/test

# KA-Prompt settings from scripts/ImageNet-R.sh
KA="--dataset=imagenet-r --pool_size=225 --prompt_num=4 --topN=2 --prompt_comp --prompt_per_task=15 --use_prompt_penalty_3 --fuse_prompt --lr=0.007 --resize_test=224 --resize_train=224 --greedy_init --tau 0.01 --aux_weight 0.1 --query_pos -1 --n_epochs=5"
EMA_FLAGS="--use_ema_c --adapt_ema_c --adapt_h=6.5"
N_TASKS=15

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
if [ "$EMA" != "1" ]; then VARIANT=intact_noema; FLAGS="$KA $INTACT --intact_anchor raw"
elif [ "$ANCHOR" == "raw" ]; then VARIANT=intact_ema_raw; FLAGS="$KA $EMA_FLAGS $INTACT --intact_anchor raw"
else VARIANT=intact_ema; FLAGS="$KA $EMA_FLAGS $INTACT --intact_anchor ema"; fi
NAME=${VARIANT}_var${VAR}_drift${DRIFT}_feat${FEAT}_align${ALIGN}

RUN=$OUTDIR/$NAME/seed-$SEED
DONE=$(find "$RUN/results" -maxdepth 1 -name '[0-9]*.txt' 2>/dev/null | wc -l)
# an unfinished run without resume.pt (started before resuming existed) is kept aside and started again
if [ "$DONE" -gt 0 ] && [ "$DONE" -lt "$N_TASKS" ] && [ ! -f "$RUN/resume.pt" ]; then
  OLD=$OUTDIR/$NAME/crashed_seed-${SEED}_$(date +%Y%m%d-%H%M%S)
  mv "$RUN" "$OLD" && echo "moved the unfinished run without resume.pt to $OLD" && DONE=0
fi
# SLURM opens the log in the folder of #SBATCH --output; keep it in the run folder instead
LOG=/shared/results/pkrukowski/InTAct/KA-Prompt/slurm-${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
mkdir -p "$RUN" && [ -f "$LOG" ] && mv -n "$LOG" "$RUN/"
if [ "$DONE" -ge "$N_TASKS" ]; then echo "already finished: $RUN"; exit 0; fi
# skip a run that another job is still training (e.g. when an array is submitted twice)
OWNER=$(cat "$RUN/job.txt" 2>/dev/null)
if [ -n "$OWNER" ] && [ "$OWNER" != "${SLURM_JOB_ID:-}" ] && squeue -h -j "$OWNER" 2>/dev/null | grep -q .; then
  echo "job $OWNER is still training $RUN"; exit 0
fi
[ -z "${SLURM_JOB_ID:-}" ] || echo "$SLURM_JOB_ID" > "$RUN/job.txt"
echo "task $ID: $NAME, seed $SEED$([ -f "$RUN/resume.pt" ] && echo ', resuming after its last finished task')"
KA_SPLIT="official" python main.py --info="seed-$SEED" --seed="$SEED" --output_path="$OUTDIR/$NAME" $FLAGS --resume
