#!/bin/bash
# Installation sur Jean Zay, à lancer UNE fois depuis une frontale (qui a accès à Internet) :
#   cd $WORK && git clone https://github.com/LahadMbacke/wolof-normalizer.git
#   cd wolof-normalizer && bash jeanzay/setup.sh
#
# Les nœuds de calcul n'ont pas Internet : on télécharge ici le modèle et les données
# dans $WORK (le $HOME est trop petit), puis le job tourne hors ligne.
set -euo pipefail

source "$(dirname "$0")/env.sh"

# Environnement isolé des paquets du module (voir env.sh)
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip

# PyTorch compilé pour CUDA 12.6 : les wheels CUDA 12.8 ne gèrent plus les V100 (sm_70).
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu126

# Versions testées ensemble
pip install -e ".[eval]" \
    "transformers==5.20.2" "accelerate==1.15.0" "datasets==5.1.0" \
    "huggingface_hub==2.2.0" "tokenizers==0.23.3" "safetensors==0.8.0"

python - <<'EOF'
import torch
from transformers import Seq2SeqTrainer
print("PyTorch", torch.__version__, "| architectures :", torch.cuda.get_arch_list())
assert "sm_70" in torch.cuda.get_arch_list(), "ce PyTorch ne gère pas les V100"
print("Import du Trainer : OK")
EOF

# Pré-téléchargement du modèle de base et du jeu de données
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download("google/byt5-small")
snapshot_download("Lahad/wolof-normalization-pairs", repo_type="dataset")
print("Modèle et données en cache dans", os.environ["HF_HOME"])
EOF

echo "Installation terminée. Lancer l'entraînement avec : sbatch jeanzay/train.slurm"
