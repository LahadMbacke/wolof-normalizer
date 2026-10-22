"""Propose des entrées de lexique à partir de la partie entraînement (80 %) du jeu de données.

On ne garde que les phrases où source et référence ont le même nombre de mots, on aligne
les mots position par position, et on retient les variantes fréquentes dont la forme
officielle est presque toujours la même. Le résultat est écrit dans candidates.tsv,
à relire avant de copier des lignes dans le lexique.

    python scripts/mine_lexicon.py [--min-count 3] [--min-share 0.8]
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from dataset import load_pairs, split, words

OUTPUT = Path(__file__).resolve().parent.parent / "candidates.tsv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--min-share", type=float, default=0.8)
    args = parser.parse_args()

    train, _ = split(load_pairs())
    targets: dict[str, Counter] = defaultdict(Counter)
    for source, reference in train:
        source_words, reference_words = words(source), words(reference)
        if len(source_words) != len(reference_words):
            continue
        for variant, official in zip(source_words, reference_words):
            targets[variant][official] += 1

    candidates = []
    for variant, counter in targets.items():
        official, count = counter.most_common(1)[0]
        total = sum(counter.values())
        if official != variant and count >= args.min_count and count / total >= args.min_share:
            candidates.append((count, variant, official, count / total))

    candidates.sort(reverse=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        f.write("# variante\tofficiel\toccurrences\tpart\n")
        for count, variant, official, share in candidates:
            f.write(f"{variant}\t{official}\t{count}\t{share:.2f}\n")
    print(f"{len(candidates)} candidats écrits dans {OUTPUT.name}")


if __name__ == "__main__":
    main()
