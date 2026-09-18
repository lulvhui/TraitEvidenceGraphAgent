from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import yaml
from packaging.version import Version


TRAITS = {"O", "C", "E", "A", "N"}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a portable Qwen3-Omni TraitGraph bundle")
    parser.add_argument("--dataset", choices=("IELTS", "RecruitView"), required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    config_path = root / "configs" / "qwen3_omni_local.yaml"
    if not config_path.is_file():
        fail(errors, f"Missing config: {config_path}")
        config = {}
    else:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    backbone = dict(config.get("backbone", {}) or {})

    model_path = root / str(backbone.get("model_path", ""))
    if not model_path.is_dir():
        fail(errors, f"Missing Qwen3-Omni model directory: {model_path}")
    else:
        config_json = model_path / "config.json"
        if not config_json.is_file():
            fail(errors, f"Model config is missing: {config_json}")
        else:
            model_config = json.loads(config_json.read_text(encoding="utf-8"))
            if model_config.get("model_type") != "qwen3_omni_moe":
                fail(
                    errors,
                    "Wrong model type: expected qwen3_omni_moe, got "
                    f"{model_config.get('model_type')!r}",
                )
        if not (model_path / "model.safetensors.index.json").is_file():
            fail(errors, f"Model weight index is missing: {model_path}")

    bge_path = root / str((config.get("context_similarity") or {}).get("model_path", ""))
    if not bge_path.is_dir():
        fail(errors, f"Missing BGE-M3 directory: {bge_path}")
    elif not (bge_path / "config.json").is_file():
        fail(errors, f"BGE-M3 config.json is missing: {bge_path}")

    input_path = root / "inputs" / args.dataset / f"{args.split}.input.jsonl"
    if not input_path.is_file():
        fail(errors, f"Missing input manifest: {input_path}")
        rows = []
    else:
        rows = [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    missing_media = []
    for line_number, row in enumerate(rows, start=1):
        missing = {
            key for key in ("person_id", "turn_id", "video_path", "source_dataset")
            if key not in row
        }
        if missing:
            fail(errors, f"Input line {line_number} is missing fields: {sorted(missing)}")
            continue
        if str(row["source_dataset"]) != args.dataset:
            fail(errors, f"Input line {line_number} has wrong source_dataset")
        media = root / str(row["video_path"])
        if not media.is_file():
            missing_media.append(str(row["video_path"]))
    if missing_media:
        fail(
            errors,
            f"Missing {len(missing_media)} media files; first five: {missing_media[:5]}",
        )

    if not shutil.which("ffmpeg"):
        fail(errors, "ffmpeg is not available in PATH")

    try:
        import torch
    except ImportError:
        fail(errors, "PyTorch is not installed")
        torch = None
    if torch is not None:
        if not torch.cuda.is_available():
            fail(errors, "CUDA is not available to PyTorch")
        else:
            total_gib = sum(
                torch.cuda.get_device_properties(index).total_memory
                for index in range(torch.cuda.device_count())
            ) / 2**30
            if not bool(backbone.get("load_in_4bit", False)) and total_gib < 80:
                warnings.append(
                    f"Only {total_gib:.1f} GiB aggregate visible VRAM; full BF16 may OOM"
                )

    try:
        import transformers

        if Version(transformers.__version__) < Version("5.2.0"):
            fail(errors, f"Transformers {transformers.__version__} is too old; need >=5.2.0")
        from transformers import (  # noqa: F401
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )
    except ImportError as exc:
        fail(errors, f"Qwen3-Omni Transformers classes are unavailable: {exc}")

    if importlib.util.find_spec("qwen_omni_utils") is None:
        fail(errors, "qwen-omni-utils is not installed")
    if str(backbone.get("attn_implementation", "")) == "flash_attention_2":
        if importlib.util.find_spec("flash_attn") is None:
            fail(
                errors,
                "flash-attn is missing while config requests flash_attention_2; "
                "install it or deliberately change the config to sdpa",
            )

    report = {
        "valid": not errors,
        "dataset": args.dataset,
        "split": args.split,
        "people": len({str(row.get("person_id")) for row in rows}),
        "turns": len(rows),
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
