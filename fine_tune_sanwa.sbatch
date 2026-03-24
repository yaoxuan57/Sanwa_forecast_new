#!/bin/bash
#SBATCH -A prj0000000262
#SBATCH -p h200p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH -J pl_ft
#SBATCH -o ~/logs/%x-%j.out

set -euo pipefail

# ---- paths ----
IMAGE="/scratch/prj0000000262/images/pytorch-2.4-cuda12.1.sif"
CODE_DIR="$HOME/Sanwa_forecast_new"
DATA_DIR="/scratch/prj0000000262/Sanwa_forecast_new"
RUN_DIR="/scratch/prj0000000262/Sanwa_forecast_new/Sanwa_forecast_ft"

VENV_DIR="/scratch/prj0000000262/ngyx/taisin/taisin-pl-py311"
PIP_CACHE_DIR="/scratch/prj0000000262/ngyx/taisin/.pip_cache"

SCRIPT="fine_tune_sanwa_norm_arg.py"
JOBTAG="${SLURM_JOB_ID:-local}"

mkdir -p "$RUN_DIR" "$HOME/logs" "$PIP_CACHE_DIR"

echo "[INFO] Host: $(hostname)"
echo "[INFO] Job:  ${SLURM_JOB_ID:-N/A}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# ---- stability ----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# ---- apptainer ----
APPTAINER_BIN="$(command -v apptainer || true)"
if [ -z "$APPTAINER_BIN" ]; then
  APPTAINER_BIN="$(command -v singularity || true)"
fi
[ -n "$APPTAINER_BIN" ] || { echo "FATAL: apptainer/singularity not found"; exit 127; }
[ -f "$IMAGE" ] || { echo "FATAL: image not found: $IMAGE"; exit 2; }

BIND_OPTS="--bind ${CODE_DIR}:/workspace --bind /scratch:/scratch --bind ${HOME}/logs:${HOME}/logs"

# IMPORTANT: keep srun -n 1 (one task). Lightning will spawn 8 GPU workers internally.
srun -n 1 env \
  DATA_DIR="$DATA_DIR" \
  RUN_DIR="$RUN_DIR" \
  VENV_DIR="$VENV_DIR" \
  PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  SCRIPT="$SCRIPT" \
  JOBTAG="$JOBTAG" \
  "$APPTAINER_BIN" exec --nv $BIND_OPTS "$IMAGE" bash -lc '
set -euo pipefail

cd /workspace
echo "[IN ] CWD: $(pwd)"
python3 -V

# venv
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
. "$VENV_DIR/bin/activate"
echo "[OK] venv ready"

export PIP_CACHE_DIR="$PIP_CACHE_DIR"
python -m pip install -U pip wheel
pip install -r requirements.txt

cd "$RUN_DIR"

python -u "/workspace/$SCRIPT" \
  --run_tag "$JOBTAG" \
  --data_path "$DATA_DIR" \
  --data_id "step_50" \
  --data_percentage "1" \
  --model_id "Sanwa_forecast" \
  --model_type "tiny" \
  --patch_size 64 \
  --dropout 0.1 \
  --num_gpus 8 \
  --num_workers 0 \
  --pin_memory false \
  --num_channels 11 \
  --horizon 50 \
  --x_norm zscore \
  --y_norm zscore \
  --num_epochs 200 \
  --patience 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --wt_decay 1e-4 \
  --random_seed 42 \
  --task_type RUL \
  --load_from_pretrained true \
  --pretrained_model_dir "/home/users/astar/simt/ngyx/Sanwa_forecast_new/pretrained_models/Tiny" \
  --pretraining_epoch_id 1
'

echo "[DONE] Outputs in: $RUN_DIR"