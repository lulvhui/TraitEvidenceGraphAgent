#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/run_split.sh IELTS|RecruitView dev|test [GPU_IDS]" >&2
  exit 2
fi

DATASET=$1
SPLIT=$2
GPU_IDS=${3:-0,1,2,3}
case "$DATASET" in IELTS|RecruitView) ;; *) echo "Invalid dataset: $DATASET" >&2; exit 2 ;; esac
case "$SPLIT" in dev|test) ;; *) echo "Invalid split: $SPLIT" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
INPUT="$ROOT/inputs/$DATASET/$SPLIT.input.jsonl"
OUTPUT_ROOT="$ROOT/outputs/$DATASET/qwen3_omni_30b_a3b/trait_evidence_graph"
RESULT="$OUTPUT_ROOT/$SPLIT/result.json"
DEBUG="$OUTPUT_ROOT/$SPLIT/debug.json"
LOG="$OUTPUT_ROOT/logs/$SPLIT.log"

mkdir -p "$(dirname "$RESULT")" "$(dirname "$LOG")"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export DECORD_REWIND_RETRY_MAX=${DECORD_REWIND_RETRY_MAX:-64}

"$PYTHON_BIN" scripts/preflight.py --dataset "$DATASET" --split "$SPLIT" --root "$ROOT"
"$PYTHON_BIN" -m traitgraph.cli.run_pipeline \
  --config "$ROOT/configs/qwen3_omni_local.yaml" \
  --input "$INPUT" \
  --output "$RESULT" \
  --debug-output "$DEBUG" \
  --project-root "$ROOT" \
  --resume \
  2>&1 | tee -a "$LOG"
"$PYTHON_BIN" scripts/validate_output.py \
  --input "$INPUT" --result "$RESULT" --debug "$DEBUG"
