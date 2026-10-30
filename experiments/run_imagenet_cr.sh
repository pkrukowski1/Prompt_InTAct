#!/bin/bash
#SBATCH --job-name=run_ImageNet-Mix
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=rtx4090_batch
#SBATCH --array=0-9
#SBATCH --output=/shared/results/pkrukowski/InTAct/KA-Prompt/slurm-%x-%A_%a.out

# best configuration of grid_imagenet_cr.sh or grid_imagenet_cr_raw.sh (tools/collect_results.py prints this line)
VAR="1.0"; DRIFT="0.0001"; FEAT="0.01"; ALIGN="1.0"; EMA=1; ANCHOR=raw
SEEDS=(3407 0 42 8744 7256 3697 5718 7540 4645 2968)
OUTDIR=/shared/results/pkrukowski/InTAct/KA-Prompt/ImageNet-Mix/test

# KA-Prompt settings from scripts/ImageNet-Mix.sh
KA="--dataset=imagenet-cr --pool_size=750 --prompt_num=4 --topN=3 --prompt_comp --prompt_per_task=25 --use_prompt_penalty_3 --fuse_prompt --lr=0.005 --resize_test=224 --resize_train=224 --greedy_init --tau 0.01 --aux_weight 1.0 --query_pos -1 --n_epochs=5"
EMA_FLAGS="--use_ema_c --adapt_ema_c --adapt_h=8.7"
N_TASKS=30

eval "$(conda shell.bash hook)"
conda activate intact_with_ka_prompt
[ -f main.py ] || { echo "submit from the repository root"; exit 1; }

ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch, or set SLURM_ARRAY_TASK_ID}
[ "$ID" -lt ${#SEEDS[@]} ] || { echo "task $ID: this script has ${#SEEDS[@]} tasks"; exit 0; }
SEED=${SEEDS[$ID]}
[ -n "$VAR" ] && [ -n "$DRIFT" ] && [ -n "$FEAT" ] && [ -n "$ALIGN" ] || { echo "task $ID: set VAR, DRIFT, FEAT, ALIGN in experiments/run_imagenet_cr.sh"; exit 1; }
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
