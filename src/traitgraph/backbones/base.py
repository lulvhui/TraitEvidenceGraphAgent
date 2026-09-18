from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from ..schemas import TurnRecord


class MultimodalBackbone(ABC):
    """Model-agnostic interface.

    The framework never reads hidden states. Each backbone keeps its own native
    video/audio/text preprocessing. A role receives raw TurnRecord objects and a
    textual prompt, then returns text (normally JSON).
    """

    @abstractmethod
    def generate(
        self,
        role: str,
        prompt: str,
        turns: Sequence[TurnRecord],
        generation_config: Dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError
