"""Évalue le normaliseur sur la partie test (20 %) du jeu de données.

Métrique : WER (taux d'erreur par mot, distance d'édition sur les mots, en minuscules,
ponctuation ignorée) entre la sortie et la phrase en orthographe officielle.

    python scripts/evaluate.py
"""

from dataset import load_pairs, split, words

from wolof_normalizer import normalize


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def wer(hypotheses: list[str], references: list[str]) -> float:
    errors = sum(edit_distance(words(h), words(r)) for h, r in zip(hypotheses, references))
    return errors / sum(len(words(r)) for r in references)


def main() -> None:
    _, test = split(load_pairs())
    sources = [s for s, _ in test]
    references = [r for _, r in test]
    outputs = [normalize(s) for s in sources]

    print(f"Phrases de test : {len(test)}")
    print(f"WER sans normalisation : {wer(sources, references):.1%}")
    print(f"WER avec normalisation : {wer(outputs, references):.1%}")


if __name__ == "__main__":
    main()
