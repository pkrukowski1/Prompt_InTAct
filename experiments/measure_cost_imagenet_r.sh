#!/bin/bash
#SBATCH --job-name=cost_ImageNet-R
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=dgxh100
#SBATCH --output=/shared/results/pkrukowski/InTAct/KA-Prompt/slurm-%x-%j.out

# Efficiency table (ImageNet-R): KA-Prompt and KA-Prompt + InTAct measured in the same job, on the same GPU.
# Parameters, peak GPU memory and time per batch; see tools/measure_cost.py. Nothing is saved except the JSON files.

VAR="1.0"; DRIFT="0.001"; FEAT="0.1"; ALIGN="1.0"   # final configuration of run_imagenet_r.sh (raw anchor)
SEED=3407
TASKS=2
EPOCHS=5
OUTDIR=/shared/results/pkrukowski/InTAct/KA-Prompt/ImageNet-R/cost

# KA-Prompt settings from scripts/ImageNet-R.sh
KA="--dataset=imagenet-r --pool_size=225 --prompt_num=4 --topN=2 --prompt_comp --prompt_per_task=15 --use_prompt_penalty_3 --fuse_prompt --lr=0.007 --resize_test=224 --resize_train=224 --greedy_init --tau 0.01 --aux_weight 0.1 --query_pos -1 --n_epochs=5"
EMA_FLAGS="--use_ema_c --adapt_ema_c --adapt_h=6.5"
INTACT="--use_interval_activation --var_loss_scale $VAR --internal_repr_drift_loss_scale $DRIFT --feature_loss_scale $FEAT --use_align_loss --align_loss_scale $ALIGN --intact_anchor raw"

eval "$(conda shell.bash hook)"
conda activate intact_with_ka_prompt
[ -f main.py ] || { echo "submit from the repository root"; exit 1; }
mkdir -p "$OUTDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for VARIANT in ka intact; do
  FLAGS="$KA $EMA_FLAGS"
  [ "$VARIANT" == "intact" ] && FLAGS="$FLAGS $INTACT"
  KA_SPLIT="official" python tools/measure_cost.py --seed="$SEED" $FLAGS \
    --cost_tasks "$TASKS" --cost_epochs "$EPOCHS" --cost_label "$VARIANT" --cost_out "$OUTDIR/$VARIANT.json"
done
