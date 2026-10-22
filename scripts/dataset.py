"""Chargement du jeu de données soynade-research/Wolof-Non-Standard-Orthography (CC-BY-SA 4.0).

Les données sont téléchargées dans .cache/ (non versionné) et ne sont pas redistribuées ici.
"""

import json
import re
import unicodedata
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

URL = (
    "https://huggingface.co/datasets/soynade-research/Wolof-Non-Standard-Orthography"
    "/resolve/main/data/train-00000-of-00001.parquet"
)
SMOL_URL = "https://huggingface.co/datasets/google/smol/resolve/main/{}/en_wo.jsonl"
WIKIPEDIA_URL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia"
    "/resolve/main/20231101.wo/train-00000-of-00001.parquet"
)
HUB_DATASET = "Lahad/wolof-normalization-pairs"
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


def load_smol() -> list[str]:
    """Phrases wolof standard de google/smol (CC BY 4.0) : SmolSent, SmolDoc et GATITOS."""
    sentences = []
    for subset in ["smolsent", "smoldoc", "gatitos"]:
        path = _download(SMOL_URL.format(subset), CACHE_DIR / f"smol_{subset}_en_wo.jsonl")
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            sentences += [row["trg"]] if "trg" in row else row["trgs"]
    return [_clean(s) for s in dict.fromkeys(sentences) if s.strip()]


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


def hub_split(name: str, repo_id: str = HUB_DATASET) -> Path:
    """Télécharge une partie (train, validation, test) du jeu de données publié sur Hugging Face."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, f"data/{name}.jsonl", repo_type="dataset"))


def split(pairs, train_ratio=0.8):
    cut = int(len(pairs) * train_ratio)
    return pairs[:cut], pairs[cut:]


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]
