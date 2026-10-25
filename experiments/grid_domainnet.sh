#!/bin/bash
#SBATCH --job-name=grid_DomainNet
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100
#SBATCH --array=0-26

# Grid search over the InTAct hyperparameters on the DomainNet validation split (90/10 split of the train lists,
# python tools/make_val_lists.py once). One task per configuration; finished runs are skipped.
# Ranking: python tools/collect_results.py $OUTDIR
# Two accounts: sbatch --array=0-13 experiments/grid_domainnet.sh  and  sbatch --array=14-26 experiments/grid_domainnet.sh

VARS="0.01 0.1 1.0"           # lambda_Var
DRIFTS="0.0001 0.001 0.01"    # lambda_IntDrift
FEATS="0 0.01 0.1"            # lambda_Feat
ALIGNS="1.0"                  # lambda_Align
EMA=1                         # 1: keep KA-Prompt's EMA classifier, 0: without it
SEED=3407
OUTDIR=/shared/results/pkrukowski/InTAct/KA-Prompt/DomainNet/val

# KA-Prompt settings from scripts/DomainNet.sh
KA="--dataset=domain-net --pool_size=150 --prompt_num=8 --topN=3 --prompt_comp --prompt_per_task=25 --use_prompt_penalty_3 --fuse_prompt --lr=0.0006 --resize_test=224 --resize_train=224 --greedy_init --tau 0.01 --aux_weight 0.1 --n_epochs 10 --query_pos -1"
EMA_FLAGS="--use_ema_c --adapt_ema_c --adapt_h=10"
N_TASKS=6

eval "$(conda shell.bash hook)"
conda activate DIL
[ -f main.py ] || { echo "submit from the repository root"; exit 1; }
[ -d datasets/image_list/domainnet_val ] || { echo "run python tools/make_val_lists.py first"; exit 1; }

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
