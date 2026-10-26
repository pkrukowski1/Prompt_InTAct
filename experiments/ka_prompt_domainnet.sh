#!/bin/bash
#SBATCH --job-name=ka_DomainNet
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100
#SBATCH --array=0-2
#SBATCH --output=/shared/results/pkrukowski/InTAct/KA-Prompt/slurm-%x-%A_%a.out

# KA-Prompt on DomainNet with the settings of its official scripts/DomainNet.sh: one task per seed (3407, 0, 42),
# official train/test lists. Same runs and output folder as tasks 0-2 of run_domainnet.sh, so nothing runs twice.
# Finished runs are skipped; a crashed run continues from its last finished task (resume.pt in its run folder).
# Each task's SLURM log is moved into its run folder, $OUTDIR/ka/seed-<seed>.
# Results: python tools/collect_results.py $OUTDIR   (KA-Prompt's own summary: python cal_mean_std_af.py $OUTDIR/ka)

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
