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
# A run that already has results is skipped, never overwritten. Each task's SLURM log is moved into its run
# folder, $OUTDIR/ka/seed-<seed>.
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
# SLURM opens the log in the folder of #SBATCH --output; keep it in the run folder instead
LOG=/shared/results/pkrukowski/InTAct/KA-Prompt/slurm-${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
mkdir -p "$RUN" && [ -f "$LOG" ] && mv -n "$LOG" "$RUN/"
# never overwrite a run folder that already has results
DONE=$(find "$RUN/results" -maxdepth 1 -name '[0-9]*.txt' 2>/dev/null | wc -l)
if [ "$DONE" -ge "$N_TASKS" ]; then echo "already finished: $RUN"; exit 0; fi
if [ "$DONE" -gt 0 ]; then echo "$RUN already has results of $DONE/$N_TASKS tasks; not overwriting it (delete it to rerun)"; exit 0; fi
echo "task $ID: KA-Prompt, seed $SEED"
KA_SPLIT="official" python main.py --info="seed-$SEED" --seed="$SEED" --output_path="$OUTDIR/ka" $KA $EMA_FLAGS
