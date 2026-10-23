"""Normalisation par le modèle ByT5 fine-tuné (Lahad/wolof-normalizer-byt5-small).

Nécessite l'option « model » : pip install "wolof-normalizer[model]"
"""

from __future__ import annotations

DEFAULT_MODEL = "Lahad/wolof-normalizer-byt5-small"


class ModelNormalizer:
    """Normaliseur neuronal. Le modèle (~1,2 Go) est téléchargé au premier usage puis mis en cache."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None, num_beams: int = 4):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                'Le modèle demande des dépendances en plus : pip install "wolof-normalizer[model]"'
            ) from error

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_beams = num_beams
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device).eval()

    def normalize_batch(self, texts: list[str], batch_size: int = 16) -> list[str]:
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True)
            with self._torch.no_grad():
                generated = self.model.generate(**batch.to(self.device), max_length=512, num_beams=self.num_beams)
            outputs += self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return outputs

    def normalize(self, text: str) -> str:
        return self.normalize_batch([text])[0]
