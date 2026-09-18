from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def bnb_4bit_skip_modules(config: dict[str, Any]) -> list[str]:
    """Return validated module patterns that must stay outside BnB 4-bit.

    Transformers uses the historically named ``llm_int8_skip_modules`` field
    for both 8-bit and 4-bit BitsAndBytes quantizers.  Keeping parsing here
    prevents a YAML scalar from accidentally becoming a list of characters and
    makes SFT, MARL, and inference use the same exclusion contract.
    """
    raw = config.get("bnb_4bit_skip_modules", [])
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise TypeError("bnb_4bit_skip_modules must be a string or a list of strings")
    patterns: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("bnb_4bit_skip_modules entries must be non-empty strings")
        patterns.append(item.strip())
    return patterns


def restore_bnb_4bit_skipped_module_dtype(
    model: Any,
    patterns: Iterable[str],
    *,
    dtype: Any,
    parent_prefix: str = "thinker",
) -> list[str]:
    """Restore skipped frozen modules after PEFT's blanket FP32 upcast.

    Quantization is performed on the complete Omni model, so exclusions use
    names beginning with ``thinker.``. SFT and MARL subsequently operate on
    ``omni.thinker`` alone; checking both forms preserves the same exact module
    selection after that extraction.
    """
    compiled = [re.compile(pattern) for pattern in patterns]
    if not compiled:
        return []
    restored: list[str] = []
    for name, module in model.named_modules():
        candidates = (name, f"{parent_prefix}.{name}" if name else parent_prefix)
        if not any(regex.match(candidate) for regex in compiled for candidate in candidates):
            continue
        has_floating_parameter = False
        for parameter in module.parameters(recurse=False):
            if parameter.is_floating_point():
                has_floating_parameter = True
                if parameter.dtype != dtype:
                    parameter.data = parameter.data.to(dtype=dtype)
        if has_floating_parameter:
            restored.append(name)
    return restored


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
