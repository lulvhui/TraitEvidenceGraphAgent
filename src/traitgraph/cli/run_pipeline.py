from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import yaml

from traitgraph.backbones import PythonBridgeBackbone
from traitgraph.pipeline import TraitEvidenceGraphPipeline
from traitgraph.schemas import TurnRecord
from traitgraph.utils import dump_json, load_jsonl


def _turn_sort_key(turn: TurnRecord) -> Tuple[int, object]:
    try:
        return (0, int(turn.turn_id))
    except (TypeError, ValueError):
        return (1, str(turn.turn_id))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the Trait Evidence Graph pipeline with a real multimodal backbone bridge."
    )
    ap.add_argument("--input", required=True, help="Unified IELTS/RecruitView JSONL")
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument(
        "--debug-output",
        default="",
        help="Optional separate JSON path for routing, graph, and verifier diagnostics",
    )
    ap.add_argument("--config", default="configs/qwen3vl_8B/qwen3vl_local.yaml")
    ap.add_argument(
        "--bridge-path",
        default="",
        help="Optional override for backbone.bridge_path in the config",
    )
    ap.add_argument(
        "--project-root",
        default="",
        help=(
            "Optional root used to resolve relative video_path values such as "
            "data/IELTS/... . Overrides backbone.project_root in YAML."
        ),
    )
    ap.add_argument(
        "--progress-log-every",
        type=int,
        default=1,
        help="In redirected logs, print progress every N people (default: 1).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed people in existing output/debug files and continue.",
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    backbone_cfg = dict(cfg.get("backbone", {}) or {})
    if args.project_root:
        backbone_cfg["project_root"] = args.project_root

    bridge_path = args.bridge_path or str(backbone_cfg.get("bridge_path", "") or "")
    if not bridge_path:
        raise ValueError(
            "No backbone bridge configured. Set backbone.bridge_path in the config "
            "or pass --bridge-path."
        )

    rows = load_jsonl(args.input)
    by_person: dict[tuple[str, str], list[TurnRecord]] = {}
    for line_no, row in enumerate(rows, start=1):
        try:
            turn = TurnRecord(**row)
        except TypeError as exc:
            raise TypeError(f"Invalid input schema at JSONL line {line_no}: {exc}") from exc

        if not turn.source_dataset:
            raise ValueError(f"Missing source_dataset at JSONL line {line_no}")
        if not turn.video_path:
            raise ValueError(f"Missing video_path at JSONL line {line_no}")

        key = (turn.source_dataset, turn.person_id)
        by_person.setdefault(key, []).append(turn)

    for turns in by_person.values():
        turns.sort(key=_turn_sort_key)

    backbone = PythonBridgeBackbone(bridge_path, backbone_cfg)
    pipeline = TraitEvidenceGraphPipeline(backbone, cfg)
    items = list(by_person.items())
    outputs: list[dict] = []
    debug_outputs: list[dict] = []
    if args.resume and Path(args.output).is_file():
        outputs = list(json.loads(Path(args.output).read_text(encoding="utf-8")))
        if args.debug_output and Path(args.debug_output).is_file():
            debug_outputs = list(
                json.loads(Path(args.debug_output).read_text(encoding="utf-8"))
            )
    output_ids = {
        (str(item.get("source_dataset", "")), str(item.get("person_id", "")))
        for item in outputs
    }
    debug_ids = {
        (str(item.get("source_dataset", "")), str(item.get("person_id", "")))
        for item in debug_outputs
    }
    completed = output_ids & debug_ids if args.debug_output else output_ids
    pending_items = [item for item in items if item[0] not in completed]
    started = time.monotonic()
    progress = None
    if sys.stderr.isatty():
        from tqdm.auto import tqdm

        progress = tqdm(
            total=len(pending_items), desc="TraitGraph", unit="person", dynamic_ncols=True
        )
    for index, ((dataset, person_id), turns) in enumerate(pending_items, start=1):
        result = pipeline.infer_person(turns)
        outputs.append(result.to_dict())
        if args.debug_output:
            debug_outputs.append(result.to_debug_dict())
        dump_json(args.output, outputs)
        if args.debug_output:
            dump_json(args.debug_output, debug_outputs)
        if progress is not None:
            progress.set_postfix_str(f"{dataset}/{person_id}")
            progress.update(1)
        elif args.progress_log_every > 0 and (
            index % args.progress_log_every == 0 or index == len(pending_items)
        ):
            elapsed = time.monotonic() - started
            rate = index / elapsed if elapsed else 0.0
            remaining = (len(pending_items) - index) / rate if rate else 0.0
            print(
                f"[Pipeline] {index}/{len(pending_items)} "
                f"({100 * index / len(pending_items):.1f}%) "
                f"person={dataset}/{person_id} elapsed={elapsed:.0f}s eta={remaining:.0f}s",
                flush=True,
            )
    if progress is not None:
        progress.close()
    dump_json(args.output, outputs)
    if args.debug_output:
        dump_json(args.debug_output, debug_outputs)
    print(f"Saved {len(outputs)} person results to {args.output}")


if __name__ == "__main__":
    main()
