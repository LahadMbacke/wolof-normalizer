"""Taux d'erreur par mot (WER), en minuscules, ponctuation ignorée."""

from dataset import words


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
