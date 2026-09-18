from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable

from .pool.graph_builder import ContextPair, context_pair
from .schemas import EvidenceItem
from .utils import clamp01


def required_context_pairs(evidence: Iterable[EvidenceItem]) -> set[ContextPair]:
    groups = defaultdict(set)
    for item in evidence:
        if (
            item.direction in {"support_high", "support_low"}
            and item.dialogue_context
            and item.dialogue_context != "unknown"
        ):
            groups[(item.trait, item.direction)].add(item.dialogue_context)
    pairs = set()
    for contexts in groups.values():
        ordered = sorted(contexts)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                pairs.add(context_pair(left, right))
    return pairs


class BGEContextSimilarity:
    """Dense semantic similarity for open-ended dialogue-context labels."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.model_path = Path(str(cfg.get("model_path", ""))).expanduser()
        self.device_name = str(cfg.get("device", "cpu"))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.max_length = int(cfg.get("max_length", 64))
        self._tokenizer = None
        self._model = None
        self._device = None
        self._cache = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Local BGE-M3 model path not found: {self.model_path}")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BGE-M3 context similarity requires torch and transformers in the runtime environment."
            ) from exc

        if self.device_name == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.device_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True
        )
        self._model = AutoModel.from_pretrained(
            str(self.model_path), local_files_only=True
        ).to(self._device)
        self._model.eval()

    @staticmethod
    def _text(label: str) -> str:
        return str(label).replace("_", " ").strip()

    def _encode_missing(self, labels: Iterable[str]) -> None:
        missing = sorted({label for label in labels if label not in self._cache})
        if not missing:
            return
        self._load()

        import torch
        import torch.nn.functional as functional

        for start in range(0, len(missing), self.batch_size):
            batch_labels = missing[start:start + self.batch_size]
            encoded = self._tokenizer(
                [self._text(label) for label in batch_labels],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.inference_mode():
                # The local sentence-transformers config specifies CLS pooling.
                vectors = self._model(**encoded).last_hidden_state[:, 0]
                vectors = functional.normalize(vectors, p=2, dim=1).cpu()
            for label, vector in zip(batch_labels, vectors):
                self._cache[label] = vector

    def similarities(self, pairs: Iterable[ContextPair]) -> Dict[ContextPair, float]:
        requested = {context_pair(*pair) for pair in pairs if pair[0] != pair[1]}
        if not requested:
            return {}
        self._encode_missing(label for pair in requested for label in pair)
        return {
            pair: clamp01(float((self._cache[pair[0]] @ self._cache[pair[1]]).item()))
            for pair in requested
        }
