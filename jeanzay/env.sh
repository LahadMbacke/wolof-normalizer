# Variables communes à setup.sh et train.slurm.
# Adapter PYTORCH_MODULE à un module existant : `module avail pytorch-gpu`
PYTORCH_MODULE="pytorch-gpu/py3/2.5.0"

module purge
module load "$PYTORCH_MODULE"

export VENV="$WORK/venvs/wolof-normalizer"
export HF_HOME="$WORK/hf_cache"
