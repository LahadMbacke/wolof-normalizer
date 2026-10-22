"""Chargement du jeu de données soynade-research/Wolof-Non-Standard-Orthography (CC-BY-SA 4.0).

Les données sont téléchargées dans .cache/ (non versionné) et ne sont pas redistribuées ici.
"""

import re
import unicodedata
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

URL = (
    "https://huggingface.co/datasets/soynade-research/Wolof-Non-Standard-Orthography"
    "/resolve/main/data/train-00000-of-00001.parquet"
)
CACHE = Path(__file__).resolve().parent.parent / ".cache" / "wolof_non_standard.parquet"

_TAG_RE = re.compile(r"</?NON_STANDARD>")
_WORD_RE = re.compile(r"\w+")


def _clean(text: str) -> str:
    text = _TAG_RE.sub("", text).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return unicodedata.normalize("NFC", text.strip())


def load_pairs() -> list[tuple[str, str]]:
    """Renvoie la liste des paires (non standard, standard), dans l'ordre du jeu de données."""
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, CACHE)
    rows = pq.read_table(CACHE, columns=["non_standardized", "wo"]).to_pylist()
    return [(_clean(r["non_standardized"]), _clean(r["wo"])) for r in rows]


def split(pairs, train_ratio=0.8):
    cut = int(len(pairs) * train_ratio)
    return pairs[:cut], pairs[cut:]


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]
