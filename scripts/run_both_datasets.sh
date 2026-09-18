#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/run_both_datasets.sh [GPU_IDS]" >&2
  exit 2
fi
GPU_IDS=${1:-0,1,2,3}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Finish and validate both development sets before touching either test set.
bash "$ROOT/scripts/run_split.sh" IELTS dev "$GPU_IDS"
bash "$ROOT/scripts/run_split.sh" RecruitView dev "$GPU_IDS"
bash "$ROOT/scripts/run_split.sh" IELTS test "$GPU_IDS"
bash "$ROOT/scripts/run_split.sh" RecruitView test "$GPU_IDS"
