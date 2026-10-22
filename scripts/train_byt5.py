"""Fine-tune ByT5 (google/byt5-small) pour normaliser le wolof non standard.

Prévu pour Colab (GPU T4/L4/A100). Les données viennent par défaut du jeu publié sur
Hugging Face (Lahad/wolof-normalization-pairs) ; --local-data utilise plutôt data_build/
produit par make_synthetic.py.

    python scripts/train_byt5.py --output-dir runs/byt5-small
    python scripts/train_byt5.py --model google/byt5-base --batch-size 8   # plus gros, plus lent
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from metrics import edit_distance
from dataset import HUB_DATASET, hub_split, words

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_build"


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, tokenizer, max_bytes: int):
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        # ByT5 travaille sur les octets UTF-8 : on écarte les phrases trop longues.
        self.rows = [r for r in rows if max(len(r["source"].encode()), len(r["target"].encode())) < max_bytes]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        features = self.tokenizer(row["source"])
        features["labels"] = self.tokenizer(text_target=row["target"])["input_ids"]
        return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/byt5-small")
    parser.add_argument("--data-repo", default=HUB_DATASET, help="jeu de données Hugging Face")
    parser.add_argument("--local-data", action="store_true", help="utiliser data_build/ au lieu du Hub")
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "byt5-small"))
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-bytes", type=int, default=256)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="moins de mémoire, environ 30 %% plus lent")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=-1, help="pour un test rapide")
    parser.add_argument("--push-to-hub", default=None, help="ex. Lahad/wolof-normalizer-byt5-small")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if args.local_data:
        train_path, dev_path = DATA / "train.jsonl", DATA / "dev.jsonl"
    else:
        train_path, dev_path = hub_split("train", args.data_repo), hub_split("validation", args.data_repo)
    train = JsonlDataset(train_path, tokenizer, args.max_bytes)
    dev = JsonlDataset(dev_path, tokenizer, args.max_bytes)
    print(f"train : {len(train)} | dev : {len(dev)}")

    def compute_metrics(prediction):
        predictions, labels = prediction
        predictions[predictions < 0] = tokenizer.pad_token_id
        labels[labels < 0] = tokenizer.pad_token_id
        hypotheses = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        references = tokenizer.batch_decode(labels, skip_special_tokens=True)
        errors = sum(edit_distance(words(h), words(r)) for h, r in zip(hypotheses, references))
        return {"wer": errors / sum(len(words(r)) for r in references)}

    steps_per_epoch = -(-len(train) // (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)

    # T5 est instable en fp16 : bf16 si le GPU le permet (L4, A100), sinon fp32 (T4).
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=int(0.05 * total_steps),
        optim="adafactor",
        # Lots de phrases de longueur proche : moins de padding, donc moins de mémoire et plus rapide.
        train_sampling_strategy="group_by_length",
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=use_bf16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=args.max_bytes,
        logging_steps=50,
        report_to="none",
        push_to_hub=args.push_to_hub is not None,
        hub_model_id=args.push_to_hub,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )
    # Reprend au dernier checkpoint si une session précédente a été coupée (Colab).
    resume = any(Path(args.output_dir).glob("checkpoint-*"))
    if resume:
        print("Reprise au dernier checkpoint de", args.output_dir)
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(Path(args.output_dir) / "best")
    tokenizer.save_pretrained(Path(args.output_dir) / "best")
    print("Meilleur WER dev :", trainer.state.best_metric)
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
