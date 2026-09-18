from __future__ import annotations

import argparse
import json
from pathlib import Path


TRAITS = {"O", "C", "E", "A", "N"}


def identities(rows: list[dict]) -> set[tuple[str, str]]:
    return {
        (str(row.get("source_dataset", "")), str(row.get("person_id", "")))
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TraitGraph public/debug outputs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--debug", type=Path, required=True)
    args = parser.parse_args()

    input_rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = json.loads(args.result.read_text(encoding="utf-8"))
    debug = json.loads(args.debug.read_text(encoding="utf-8"))
    expected = identities(input_rows)
    errors = []
    if identities(results) != expected:
        errors.append("result identities do not exactly match the input split")
    if identities(debug) != expected:
        errors.append("debug identities do not exactly match the input split")
    for row in results:
        traits = row.get("traits") or {}
        if set(traits) != TRAITS:
            errors.append(f"{row.get('person_id')}: trait keys are incomplete")
            continue
        for trait, value in traits.items():
            if not str((value or {}).get("reasoning", "")).strip():
                errors.append(f"{row.get('person_id')}/{trait}: empty reasoning")
    report = {
        "valid": not errors,
        "expected_people": len(expected),
        "result_people": len(results),
        "traits": len(results) * 5,
        "errors": errors[:20],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
