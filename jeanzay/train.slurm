#!/bin/bash
# Entraînement ByT5-small sur Jean Zay. Avant : bash jeanzay/setup.sh (une fois, sur la frontale).
#   sbatch jeanzay/train.slurm
#   squeue -u $USER                       # suivre le job
#   tail -f logs/byt5-wolof-<jobid>.out    # suivre l'entraînement
#
# Remplacer <projet> par ton identifiant de projet (commande `idracct`).
# Partition A100 ci-dessous ; pour V100 : -A <projet>@v100 et -C v100-32g ; pour H100 : -A <projet>@h100 et -C h100.
#SBATCH --job-name=byt5-wolof
#SBATCH --account=<projet>@a100
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source jeanzay/env.sh
source "$VENV/bin/activate"

# Pas d'Internet sur les nœuds de calcul : tout vient du cache rempli par setup.sh
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR="$WORK/wolof-normalizer-runs/byt5-small"

nvidia-smi --query-gpu=name,memory.total --format=csv
srun python scripts/train_byt5.py --output-dir "$OUTPUT_DIR" --epochs 5 --batch-size 32 --grad-accum 1
srun python scripts/run_evaluation.py --data-repo Lahad/wolof-normalization-pairs --model "$OUTPUT_DIR/best"
