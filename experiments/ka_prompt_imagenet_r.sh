#!/bin/bash
#SBATCH --job-name=ka_ImageNet-R
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100
#SBATCH --array=0-2
#SBATCH --output=/shared/results/pkrukowski/InTAct/KA-Prompt/slurm-%x-%A_%a.out

SEEDS=(3407 0 42)
OUTDIR=/shared/results/pkrukowski/InTAct/KA-Prompt/ImageNet-R/test

# KA-Prompt settings from scripts/ImageNet-R.sh
KA="--dataset=imagenet-r --pool_size=225 --prompt_num=4 --topN=2 --prompt_comp --prompt_per_task=15 --use_prompt_penalty_3 --fuse_prompt --lr=0.007 --resize_test=224 --resize_train=224 --greedy_init --tau 0.01 --aux_weight 0.1 --query_pos -1 --n_epochs=5"
EMA_FLAGS="--use_ema_c --adapt_ema_c --adapt_h=6.5"
N_TASKS=15

eval "$(conda shell.bash hook)"
conda activate intact_with_ka_prompt
[ -f main.py ] || { echo "submit from the repository root"; exit 1; }

ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch, or set SLURM_ARRAY_TASK_ID}
[ "$ID" -lt ${#SEEDS[@]} ] || { echo "task $ID: this script has ${#SEEDS[@]} tasks"; exit 0; }
SEED=${SEEDS[$ID]}

RUN=$OUTDIR/ka/seed-$SEED
DONE=$(find "$RUN/results" -maxdepth 1 -name '[0-9]*.txt' 2>/dev/null | wc -l)
# an unfinished run without resume.pt (started before resuming existed) is kept aside and started again
if [ "$DONE" -gt 0 ] && [ "$DONE" -lt "$N_TASKS" ] && [ ! -f "$RUN/resume.pt" ]; then
  OLD=$OUTDIR/ka/crashed_seed-${SEED}_$(date +%Y%m%d-%H%M%S)
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
echo "task $ID: KA-Prompt, seed $SEED$([ -f "$RUN/resume.pt" ] && echo ', resuming after its last finished task')"
KA_SPLIT="official" python main.py --info="seed-$SEED" --seed="$SEED" --output_path="$OUTDIR/ka" $KA $EMA_FLAGS --resume
