from __future__ import annotations

import copy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable


VALID_VIDEO_READERS = {"decord", "torchvision", "torchcodec"}


def set_qwenvl_video_reader(reader: str) -> None:
    """Select a qwen-vl-utils reader even after that module was imported."""
    if reader not in VALID_VIDEO_READERS:
        raise ValueError(f"Unsupported video reader: {reader}")
    os.environ["FORCE_QWENVL_VIDEO_READER"] = reader
    try:
        from qwen_vl_utils import vision_process
    except ImportError:
        return
    backend = vision_process.get_video_reader_backend
    if vision_process.FORCE_QWENVL_VIDEO_READER != reader:
        vision_process.FORCE_QWENVL_VIDEO_READER = reader
        backend.cache_clear()
    if backend.cache_info().currsize == 0:
        # qwen-vl-utils prints the selected backend to stderr whenever this
        # one-entry cache is cold. Prime it silently; real decode warnings and
        # exceptions occur later and remain visible.
        with contextlib.redirect_stderr(io.StringIO()):
            backend()


def probe_video(
    item: dict,
    reader: str,
    image_patch_size: int,
    timeout_seconds: float,
) -> tuple[bool, str]:
    """Check a video's decoder in an isolated, time-bounded process."""
    del image_patch_size  # Kept in the public callback contract.
    command = [
        sys.executable,
        str(Path(__file__).with_name("video_probe.py")),
        "--item-json",
        json.dumps(item, ensure_ascii=False),
        "--reader",
        reader,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_seconds:g}s"
    if result.returncode == 0:
        return True, ""
    detail = (result.stderr or result.stdout or "decode failed").strip().splitlines()
    return False, detail[-1] if detail else "decode failed"


def prepare_video_messages(
    messages: list,
    *,
    primary_reader: str,
    fallback_reader: str,
    image_patch_size: int,
    primary_timeout_seconds: float,
    fallback_timeout_seconds: float,
    cache: dict[str, str | None],
    probe: Callable[[dict, str, int, float], tuple[bool, str]] = probe_video,
) -> tuple[list, str, list[str]]:
    """Choose a responsive reader per video and retain transcript on failure."""
    cleaned = copy.deepcopy(messages)
    selected_reader = primary_reader
    warnings: list[str] = []
    for message_group in cleaned:
        group = message_group if isinstance(message_group, list) else [message_group]
        for message in group:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            kept = []
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "video":
                    kept.append(item)
                    continue
                key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                if key not in cache:
                    ok, primary_error = probe(
                        item, primary_reader, image_patch_size, primary_timeout_seconds
                    )
                    if ok:
                        cache[key] = primary_reader
                    elif fallback_reader != primary_reader:
                        ok, fallback_error = probe(
                            item,
                            fallback_reader,
                            image_patch_size,
                            fallback_timeout_seconds,
                        )
                        cache[key] = fallback_reader if ok else None
                        if ok:
                            warnings.append(
                                f"{item.get('video')}: {primary_reader} {primary_error}; "
                                f"using {fallback_reader}"
                            )
                        else:
                            warnings.append(
                                f"{item.get('video')}: {primary_reader} {primary_error}; "
                                f"{fallback_reader} {fallback_error}; using transcript only"
                            )
                    else:
                        cache[key] = None
                        warnings.append(
                            f"{item.get('video')}: {primary_reader} {primary_error}; "
                            "using transcript only"
                        )
                reader = cache[key]
                if reader is not None:
                    kept.append(item)
                    if reader == fallback_reader:
                        selected_reader = fallback_reader
            message["content"] = kept
    return cleaned, selected_reader, warnings
