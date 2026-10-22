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
GALSENAI_URL = (
    "https://huggingface.co/datasets/galsenai/english-wolof-smol-translation"
    "/resolve/main/data/train-00000-of-00001.parquet"
)
WIKIPEDIA_URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia"
    "/resolve/main/20231101.wo/train-00000-of-00001.parquet"
)
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE = CACHE_DIR / "wolof_non_standard.parquet"

_TAG_RE = re.compile(r"</?NON_STANDARD>")
_WORD_RE = re.compile(r"\w+")


def _clean(text: str) -> str:
    text = _TAG_RE.sub("", text).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return unicodedata.normalize("NFC", text.strip())


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_pairs() -> list[tuple[str, str]]:
    """Renvoie la liste des paires (non standard, standard), dans l'ordre du jeu de données."""
    rows = pq.read_table(_download(URL, CACHE), columns=["non_standardized", "wo"]).to_pylist()
    return [(_clean(r["non_standardized"]), _clean(r["wo"])) for r in rows]


def load_galsenai() -> list[str]:
    """Phrases wolof standard de galsenai/english-wolof-smol-translation."""
    path = _download(GALSENAI_URL, CACHE_DIR / "galsenai_smol.parquet")
    return [_clean(r["wo"]) for r in pq.read_table(path, columns=["wo"]).to_pylist() if r["wo"]]


def load_wikipedia(min_chars: int = 30, max_chars: int = 200) -> list[str]:
    """Phrases de Wikipédia en wolof (CC BY-SA), filtrées sur la longueur et le bruit."""
    path = _download(WIKIPEDIA_URL, CACHE_DIR / "wikipedia_wo.parquet")
    sentences = []
    for row in pq.read_table(path, columns=["text"]).to_pylist():
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", row["text"]):
            sentence = " ".join(sentence.split())
            if min_chars <= len(sentence) <= max_chars and not re.search(r"[0-9]{3,}|[|=<>{}\[\]]", sentence):
                sentences.append(unicodedata.normalize("NFC", sentence))
    return sentences


def split(pairs, train_ratio=0.8):
    cut = int(len(pairs) * train_ratio)
    return pairs[:cut], pairs[cut:]


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]
