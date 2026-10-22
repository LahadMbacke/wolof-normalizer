"""Ligne de commande : wolof-normalize "texte"  ou  echo "texte" | wolof-normalize"""

import argparse
import sys

from .normalizer import normalize_with_changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wolof-normalize",
        description="Normalise un texte wolof vers l'orthographe officielle (CLAD).",
    )
    parser.add_argument("text", nargs="*", help="texte à normaliser (sinon lu sur l'entrée standard)")
    parser.add_argument("-v", "--verbose", action="store_true", help="affiche les modifications faites")
    args = parser.parse_args(argv)

    lines = [" ".join(args.text)] if args.text else sys.stdin.read().splitlines()
    for line in lines:
        normalized, changes = normalize_with_changes(line)
        print(normalized)
        if args.verbose:
            for change in changes:
                print(f"  {change.original} -> {change.normalized} ({change.reason})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
