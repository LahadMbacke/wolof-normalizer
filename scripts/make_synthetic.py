"""Construit les données d'entraînement du modèle (paires bruitée -> standard).

Sources :
- soynade-research/Wolof-Non-Standard-Orthography : vraies paires (80 % train, dont 200 en dev ;
  les 20 % restants forment le test, jamais vus à l'entraînement) ;
- google/smol (CC BY 4.0) et Wikipédia en wolof (CC BY-SA 4.0) : phrases standard, auxquelles
  on ajoute un bruit « écriture SMS » (lexique extrait utilisé à l'envers + règles graphiques).
  Les phrases du test sont exclues de ces sources.

Écrit data_build/{train,dev,test}.jsonl, une paire {"source", "target", "origin"} par ligne.

    python scripts/make_synthetic.py [--smol-copies 2] [--wiki-copies 1] [--seed 0]
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from dataset import load_pairs, load_smol, load_wikipedia, split, words

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src" / "wolof_normalizer" / "data"
OUTPUT = ROOT / "data_build"
DEV_SIZE = 200

# Règles graphiques inverses : (motif, [(remplacement, poids), ...]), appliquées avec une probabilité.
CHAR_NOISE = [
    ("x", [("kh", 0.9), ("x", 0.1)]),
    ("c", [("th", 0.45), ("ch", 0.3), ("tch", 0.1), ("s", 0.05), ("c", 0.1)]),
    ("ñ", [("gn", 0.85), ("ny", 0.1), ("n", 0.05)]),
    ("ŋ", [("ng", 0.9), ("n", 0.1)]),
    ("j", [("dj", 0.35), ("di", 0.35), ("j", 0.3)]),
    ("uu", [("ou", 0.8), ("u", 0.2)]),
    ("u", [("ou", 0.85), ("u", 0.15)]),
    ("ë", [("eu", 0.75), ("e", 0.2), ("é", 0.05)]),
    ("aa", [("a", 0.6), ("aa", 0.4)]),
    ("ee", [("é", 0.5), ("e", 0.3), ("ee", 0.2)]),
    ("oo", [("o", 0.6), ("oo", 0.4)]),
    ("ii", [("i", 0.7), ("ii", 0.3)]),
    ("à", [("a", 0.9), ("à", 0.1)]),
    ("é", [("é", 0.6), ("e", 0.4)]),
    ("ó", [("o", 0.95), ("ó", 0.05)]),
    ("ã", [("an", 0.7), ("a", 0.3)]),
]
_GEMINATE_RE = re.compile(r"([bdfgjklmnprstwy])\1")
_FINAL_CONSONANT_RE = re.compile(r"[bdfgklmnprstw]$")


def load_reverse_lexicon() -> dict[str, list[tuple[str, int]]]:
    """officiel -> [(variante, occurrences)], à partir des lexiques du paquet."""
    reverse = defaultdict(list)
    for name, default_count in [("lexicon_mined.tsv", 1), ("lexicon.tsv", 3)]:
        for line in (DATA / name).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split("\t")
            variant, official = fields[0], fields[1]
            count = int(fields[2]) if len(fields) > 2 else default_count
            if " " not in variant and " " not in official:
                reverse[official.lower()].append((variant, count))
    return reverse


def _choose(options, rng):
    return rng.choices([o for o, _ in options], weights=[w for _, w in options])[0]


def char_noise(word: str, rng: random.Random) -> str:
    for pattern, options in CHAR_NOISE:
        if pattern in word:
            word = word.replace(pattern, _choose(options, rng))
    if rng.random() < 0.5:
        word = _GEMINATE_RE.sub(r"\1", word)
    if rng.random() < 0.12 and _FINAL_CONSONANT_RE.search(word):
        word += "e"  # habitude française : am -> ame, benn -> bene
    return word


def _match_case(original: str, noisy: str) -> str:
    if original[:1].isupper():
        return noisy[:1].upper() + noisy[1:]
    return noisy


def noisify(sentence: str, reverse_lexicon, rng: random.Random) -> str:
    def replace(match: re.Match) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower in reverse_lexicon and rng.random() < 0.6:
            noisy = _choose(reverse_lexicon[lower], rng)
        elif rng.random() < 0.85:
            noisy = char_noise(lower, rng)
        else:
            noisy = lower
        return _match_case(word, noisy)

    noisy = re.sub(r"\w+", replace, sentence)
    if rng.random() < 0.25:
        noisy = noisy.lower()
    if rng.random() < 0.3:
        noisy = noisy.rstrip(".!?")
    if rng.random() < 0.15:
        noisy = noisy.replace(",", "")
    return noisy.strip()


def _key(text: str) -> str:
    return " ".join(words(text))


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smol-copies", type=int, default=2)
    parser.add_argument("--wiki-copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    train_pairs, test_pairs = split(load_pairs())
    train_pairs, dev_pairs = train_pairs[:-DEV_SIZE], train_pairs[-DEV_SIZE:]
    excluded = {_key(target) for _, target in test_pairs + dev_pairs}

    reverse_lexicon = load_reverse_lexicon()
    train = [{"source": s, "target": t, "origin": "soynade"} for s, t in train_pairs]
    seen = {(row["source"], row["target"]) for row in train}

    sources = [("smol", load_smol(), args.smol_copies),
               ("wikipedia", load_wikipedia(), args.wiki_copies)]
    for origin, sentences, copies in sources:
        for sentence in sentences:
            if _key(sentence) in excluded:
                continue
            for _ in range(copies):
                # Quelques phrases restent propres, pour que le modèle apprenne à ne pas toucher au texte correct.
                noisy = sentence if rng.random() < 0.05 else noisify(sentence, reverse_lexicon, rng)
                if (noisy, sentence) not in seen:
                    seen.add((noisy, sentence))
                    train.append({"source": noisy, "target": sentence, "origin": origin})

    rng.shuffle(train)
    OUTPUT.mkdir(exist_ok=True)
    write_jsonl(OUTPUT / "train.jsonl", train)
    write_jsonl(OUTPUT / "dev.jsonl", [{"source": s, "target": t, "origin": "soynade"} for s, t in dev_pairs])
    write_jsonl(OUTPUT / "test.jsonl", [{"source": s, "target": t, "origin": "soynade"} for s, t in test_pairs])

    counts = defaultdict(int)
    for row in train:
        counts[row["origin"]] += 1
    print(f"train : {len(train)} paires {dict(counts)}")
    print(f"dev   : {len(dev_pairs)} paires réelles | test : {len(test_pairs)} paires réelles")


if __name__ == "__main__":
    main()
