"""Normalisation de textes wolof écrits « à la française » (SMS, réseaux sociaux)
vers l'orthographe officielle (alphabet CLAD).

Stratégie, du plus sûr au moins sûr :
1. lexique de variantes connues (y compris des expressions de plusieurs mots) ;
2. mot déjà présent dans le vocabulaire officiel : on le garde ;
3. règles de correspondance graphique (kh -> x, ou -> u, ...), appliquées
   seulement si le résultat existe dans le vocabulaire officiel ;
4. sinon, le mot est laissé tel quel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

# Correspondances graphiques « à la française » -> orthographe officielle.
# L'ordre compte : « tch » doit passer avant « ch ».
GRAPHEME_RULES: list[tuple[str, str]] = [
    ("tch", "c"),
    ("kh", "x"),
    ("ch", "c"),
    ("dj", "j"),
    ("gn", "ñ"),
    ("ny", "ñ"),
    ("ou", "u"),
    ("eu", "ë"),
]

_VOWELS = "aeiouàéëó"
_TOKEN_RE = re.compile(r"\w+|\W+")


@dataclass(frozen=True)
class Change:
    original: str
    normalized: str
    reason: str  # "lexicon" ou "rules"


def _load_tsv(name: str) -> dict[str, str]:
    text = resources.files("wolof_normalizer.data").joinpath(name).read_text(encoding="utf-8")
    entries = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        variant, official = line.split("\t")[:2]
        entries[variant.strip().lower()] = official.strip()
    return entries


@lru_cache(maxsize=1)
def _lexicon() -> dict[str, str]:
    # Le lexique écrit à la main est prioritaire sur le lexique extrait automatiquement.
    return {**_load_tsv("lexicon_mined.tsv"), **_load_tsv("lexicon.tsv")}


@lru_cache(maxsize=1)
def _vocabulary() -> frozenset[str]:
    text = resources.files("wolof_normalizer.data").joinpath("vocab.txt").read_text(encoding="utf-8")
    words = {w.strip().lower() for w in text.splitlines() if w.strip() and not w.startswith("#")}
    # Les formes officielles du lexique font aussi partie du vocabulaire.
    for official in _lexicon().values():
        words.update(official.lower().split())
    return frozenset(words)


@lru_cache(maxsize=1)
def _multiword_pattern() -> re.Pattern[str] | None:
    variants = sorted((v for v in _lexicon() if " " in v), key=len, reverse=True)
    if not variants:
        return None
    alternatives = "|".join(r"\s+".join(map(re.escape, v.split())) for v in variants)
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


def _apply_rules(word: str) -> str:
    for source, target in GRAPHEME_RULES:
        word = word.replace(source, target)
    # « di » / « dy » devant voyelle se prononce [j] : diam -> jam
    return re.sub(rf"d[iy](?=[{_VOWELS}])", "j", word)


def _rule_candidates(word: str) -> list[str]:
    base = _apply_rules(word)
    candidates = [base]
    # Les consonnes finales géminées sont souvent écrites simples : rek -> rekk
    if base and base[-1] not in _VOWELS:
        candidates.append(base + base[-1])
    return candidates


def _match_case(original: str, normalized: str) -> str:
    if len(original) > 1 and original.isupper():
        return normalized.upper()
    if original[:1].isupper():
        return normalized[:1].upper() + normalized[1:]
    return normalized


def normalize_word(word: str) -> tuple[str, str | None]:
    """Normalise un mot isolé. Renvoie (mot, raison) ; raison vaut None si inchangé."""
    lower = word.lower()
    lexicon = _lexicon()
    if lower in lexicon:
        return _match_case(word, lexicon[lower]), "lexicon"
    vocabulary = _vocabulary()
    if lower in vocabulary:
        return word, None
    for candidate in _rule_candidates(lower):
        if candidate in vocabulary:
            return _match_case(word, candidate), "rules"
    return word, None


def normalize_with_changes(text: str) -> tuple[str, list[Change]]:
    """Normalise un texte et renvoie aussi la liste des modifications faites."""
    changes: list[Change] = []

    pattern = _multiword_pattern()
    if pattern is not None:
        def replace_expression(match: re.Match[str]) -> str:
            original = match.group(0)
            key = " ".join(original.lower().split())
            normalized = _match_case(original, _lexicon()[key])
            changes.append(Change(original, normalized, "lexicon"))
            return normalized

        text = pattern.sub(replace_expression, text)

    parts = []
    for token in _TOKEN_RE.findall(text):
        if token[0].isalpha():
            normalized, reason = normalize_word(token)
            if reason is not None and normalized != token:
                changes.append(Change(token, normalized, reason))
            token = normalized
        parts.append(token)
    return "".join(parts), changes


def normalize(text: str) -> str:
    """Normalise un texte wolof vers l'orthographe officielle."""
    return normalize_with_changes(text)[0]
