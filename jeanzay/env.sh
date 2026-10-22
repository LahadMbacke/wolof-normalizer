# Variables communes à setup.sh et train.slurm.
# transformers 5 demande PyTorch >= 2.6 : on prend par défaut le module pytorch-gpu le plus récent.
# Pour en forcer un : PYTORCH_MODULE=pytorch-gpu/py3/2.7.0 bash jeanzay/setup.sh  (voir `module avail pytorch-gpu`)
module purge
if [ -z "${PYTORCH_MODULE:-}" ]; then
    PYTORCH_MODULE=$(module -t avail pytorch-gpu/py3 2>&1 | grep '^pytorch-gpu/py3/' | sort -V | tail -1)
fi
module load "$PYTORCH_MODULE"
echo "Module : $PYTORCH_MODULE"

# Un environnement Python par module, pour ne pas mélanger les versions de Python et PyTorch
export VENV="$WORK/venvs/wolof-normalizer-${PYTORCH_MODULE##*/}"
export HF_HOME="$WORK/hf_cache"
