"""Évalue le normaliseur sur la partie test (20 %) du jeu de données.

Métrique : WER (taux d'erreur par mot, distance d'édition sur les mots, en minuscules,
ponctuation ignorée) entre la sortie et la phrase en orthographe officielle.

    python scripts/run_evaluation.py                                # lexique + règles
    python scripts/run_evaluation.py --model runs/byt5-small/best   # + modèle ByT5

Le test (20 % de soynade) est le même que la partie « test » de Lahad/wolof-normalization-pairs ;
--data-repo le lit directement depuis Hugging Face.
"""

import argparse
import json

from dataset import hub_split, load_pairs, split
from metrics import wer

from wolof_normalizer import normalize


def generate(model_path: str, sentences: list[str], batch_size: int = 32) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
    outputs = []
    for start in range(0, len(sentences), batch_size):
        batch = tokenizer(sentences[start:start + batch_size], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            generated = model.generate(**batch, max_length=512, num_beams=4)
        outputs += tokenizer.batch_decode(generated, skip_special_tokens=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="dossier ou nom Hugging Face d'un modèle ByT5 fine-tuné")
    parser.add_argument("--data-repo", help="lire le test depuis ce jeu de données Hugging Face")
    args = parser.parse_args()

    if args.data_repo:
        rows = [json.loads(line) for line in hub_split("test", args.data_repo).open(encoding="utf-8")]
        test = [(row["source"], row["target"]) for row in rows]
    else:
        _, test = split(load_pairs())
    sources = [s for s, _ in test]
    references = [r for _, r in test]
    lexicon_outputs = [normalize(s) for s in sources]

    print(f"Phrases de test : {len(test)}")
    print(f"WER sans normalisation  : {wer(sources, references):.1%}")
    print(f"WER lexique + règles    : {wer(lexicon_outputs, references):.1%}")
    if args.model:
        print(f"WER modèle              : {wer(generate(args.model, sources), references):.1%}")
        print(f"WER lexique puis modèle : {wer(generate(args.model, lexicon_outputs), references):.1%}")


if __name__ == "__main__":
    main()
