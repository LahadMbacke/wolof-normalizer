#!/bin/bash
# Installation sur Jean Zay, à lancer UNE fois depuis une frontale (qui a accès à Internet) :
#   cd $WORK && git clone https://github.com/LahadMbacke/wolof-normalizer.git
#   cd wolof-normalizer && bash jeanzay/setup.sh
#
# Les nœuds de calcul n'ont pas Internet : on télécharge ici le modèle et les données
# dans $WORK (le $HOME est trop petit), puis le job tourne hors ligne.
set -euo pipefail

source "$(dirname "$0")/env.sh"

# Environnement Python au-dessus du module PyTorch d'IDRIS
if [ ! -d "$VENV" ]; then
    python -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -e ".[eval]" "transformers==5.20.2" accelerate datasets

# Pré-téléchargement du modèle de base et du jeu de données
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("google/byt5-small")
snapshot_download("Lahad/wolof-normalization-pairs", repo_type="dataset")
print("Modèle et données en cache dans", __import__("os").environ["HF_HOME"])
EOF

echo "Installation terminée. Lancer l'entraînement avec : sbatch jeanzay/train.slurm"
