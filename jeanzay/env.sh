# Variables communes à setup.sh et train.slurm.
# On ne prend du module IDRIS que Python : l'environnement virtuel est isolé (pas de
# --system-site-packages), car les paquets du module (peft, transformer_engine...) cassent
# l'import de transformers 5. PyTorch est installé dans l'environnement par setup.sh.
# Pour forcer un module : PYTORCH_MODULE=pytorch-gpu/py3/2.8.0 bash jeanzay/setup.sh
module purge
if [ -z "${PYTORCH_MODULE:-}" ]; then
    PYTORCH_MODULE=$(module -t avail pytorch-gpu/py3 2>&1 | grep '^pytorch-gpu/py3/' | sort -V | tail -1)
fi
module load "$PYTORCH_MODULE"
unset PYTHONPATH
echo "Module : $PYTORCH_MODULE"

export VENV="$WORK/venvs/wolof-normalizer-isolated"
export HF_HOME="$WORK/hf_cache"
