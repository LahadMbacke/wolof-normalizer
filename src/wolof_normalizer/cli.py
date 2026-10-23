"""Ligne de commande : wolof-normalize "texte"  ou  echo "texte" | wolof-normalize

Avec --model, utilise le modèle ByT5 (pip install "wolof-normalizer[model]")."""

import argparse
import sys

from .model import DEFAULT_MODEL, ModelNormalizer
from .normalizer import normalize_with_changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wolof-normalize",
        description="Normalise un texte wolof vers l'orthographe officielle (CLAD).",
    )
    parser.add_argument("text", nargs="*", help="texte à normaliser (sinon lu sur l'entrée standard)")
    parser.add_argument("-v", "--verbose", action="store_true", help="affiche les modifications faites")
    parser.add_argument("-m", "--model", action="store_true", help="utilise le modèle neuronal ByT5")
    parser.add_argument("--model-name", default=DEFAULT_MODEL,
                        help=f"modèle Hugging Face ou dossier local (par défaut {DEFAULT_MODEL})")
    args = parser.parse_args(argv)

    lines = [" ".join(args.text)] if args.text else sys.stdin.read().splitlines()
    if args.model:
        try:
            normalizer = ModelNormalizer(args.model_name)
        except ImportError as error:
            parser.exit(1, f"{error}\n")
        for normalized in normalizer.normalize_batch(lines):
            print(normalized)
        return 0

    for line in lines:
        normalized, changes = normalize_with_changes(line)
        print(normalized)
        if args.verbose:
            for change in changes:
                print(f"  {change.original} -> {change.normalized} ({change.reason})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
