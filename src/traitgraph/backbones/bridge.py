from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

from .base import MultimodalBackbone
from ..schemas import TurnRecord


class PythonBridgeBackbone(MultimodalBackbone):
    """Load a model-specific bridge without coupling the framework to its internals.

    The bridge module must expose:
      load_model(config: dict) -> Any
      generate(model, role: str, prompt: str, turns: list[dict], generation_config: dict) -> str
    """

    def __init__(self, bridge_path: str, model_config: Dict[str, Any] | None = None):
        path = Path(bridge_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        spec = importlib.util.spec_from_file_location("traitgraph_external_bridge", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import bridge: {path}")
        module = importlib.util.module_from_spec(spec)
        # Register before execution so dataclasses / type inspection inside an
        # external bridge can resolve module metadata correctly.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "load_model") or not hasattr(module, "generate"):
            raise AttributeError("Bridge must define load_model() and generate().")
        self.module = module
        self.model_config = model_config or {}
        self.model = module.load_model(self.model_config)

    def generate(
        self,
        role: str,
        prompt: str,
        turns: Sequence[TurnRecord],
        generation_config: Dict[str, Any] | None = None,
    ) -> str:
        return str(
            self.module.generate(
                self.model,
                role=role,
                prompt=prompt,
                turns=[t.to_dict() for t in turns],
                generation_config=generation_config or {},
            )
        )

    def diagnostic_cursor(self) -> int:
        return len(self.diagnostics())

    def diagnostics(self) -> list[dict]:
        getter = getattr(self.module, "get_diagnostics", None)
        if getter is None:
            return []
        values = getter(self.model)
        return [dict(value) for value in values]

    def diagnostics_since(self, cursor: int) -> list[dict]:
        return self.diagnostics()[max(0, int(cursor)):]
