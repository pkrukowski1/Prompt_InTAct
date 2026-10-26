#!/bin/bash
#SBATCH --job-name=run_noema_DomainNet
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100
#SBATCH --array=0-5
#SBATCH --output=/shared/results/pkrukowski/InTAct/KA-Prompt/slurm-%x-%A_%a.out

# Final runs on the DomainNet test lists without KA-Prompt's EMA classifier: tasks 0-2 KA-Prompt w/o EMA,
# tasks 3-5 KA-Prompt w/o EMA + InTAct (seeds 3407, 0, 42). Same output folder as run_domainnet.sh.
# Finished runs are skipped; a crashed run continues from its last finished task (resume.pt in its run folder).
# Each task's SLURM log is moved into its run folder, $OUTDIR/<configuration>/seed-<seed>.
# Results: python tools/collect_results.py $OUTDIR

# best configuration of grid_domainnet_noema.sh (tools/collect_results.py prints this line)
VAR=""; DRIFT=""; FEAT=""; ALIGN=""; EMA=0
SEEDS=(3407 0 42)
OUTDIR=/shared/results/pkrukowski/InTAct/KA-Prompt/DomainNet/test

# KA-Prompt settings from scripts/DomainNet.sh
KA="--dataset=domain-net --pool_size=150 --prompt_num=8 --topN=3 --prompt_comp --prompt_per_task=25 --use_prompt_penalty_3 --fuse_prompt --lr=0.0006 --resize_test=224 --resize_train=224 --greedy_init --tau 0.01 --aux_weight 0.1 --n_epochs 10 --query_pos -1"
EMA_FLAGS="--use_ema_c --adapt_ema_c --adapt_h=10"
N_TASKS=6

eval "$(conda shell.bash hook)"
conda activate intact_with_ka_prompt
[ -f main.py ] || { echo "submit from the repository root"; exit 1; }

ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch, or set SLURM_ARRAY_TASK_ID}
N=${#SEEDS[@]}
[ "$ID" -lt $((2 * N)) ] || { echo "task $ID: this script has $((2 * N)) tasks"; exit 0; }
SEED=${SEEDS[$((ID % N))]}
if [ "$ID" -lt "$N" ]; then
  NAME=ka_noema; FLAGS="$KA"
else
  [ -n "$VAR" ] && [ -n "$DRIFT" ] && [ -n "$FEAT" ] && [ -n "$ALIGN" ] || { echo "task $ID: set VAR, DRIFT, FEAT, ALIGN in experiments/run_domainnet_noema.sh"; exit 1; }
  INTACT="--use_interval_activation --var_loss_scale $VAR --internal_repr_drift_loss_scale $DRIFT --feature_loss_scale $FEAT --use_align_loss --align_loss_scale $ALIGN --use_intact_metrics"
  if [ "$EMA" == "1" ]; then VARIANT=intact_ema; FLAGS="$KA $EMA_FLAGS $INTACT --intact_anchor ema"
  else VARIANT=intact_noema; FLAGS="$KA $INTACT --intact_anchor raw"; fi
  NAME=${VARIANT}_var${VAR}_drift${DRIFT}_feat${FEAT}_align${ALIGN}
fi

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
