from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from urllib.parse import unquote, urlparse


FRAME_FACTOR = 2
DEFAULT_FPS = 2.0
MIN_FRAMES = 4
MAX_FRAMES = 768


def local_path(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return unquote(parsed.path)
    if parsed.scheme:
        raise ValueError(f"Probe only supports local videos, got {parsed.scheme!r}")
    return str(Path(value).expanduser())


def floor_to_factor(value: float, factor: int = FRAME_FACTOR) -> int:
    return math.floor(value / factor) * factor


def ceil_to_factor(value: float, factor: int = FRAME_FACTOR) -> int:
    return math.ceil(value / factor) * factor


def round_to_factor(value: float, factor: int = FRAME_FACTOR) -> int:
    return round(value / factor) * factor


def sample_indices(item: dict, total_frames: int, video_fps: float) -> list[int]:
    """Match qwen-vl-utils frame selection before invoking a decoder."""
    if total_frames <= 0 or video_fps <= 0:
        raise ValueError("video must have positive frame count and fps")

    duration = total_frames / video_fps
    start_seconds = max(0.0, min(float(item.get("video_start", 0.0)), duration))
    end_seconds = max(
        0.0,
        min(float(item.get("video_end", duration)), duration),
    )
    start = math.ceil(start_seconds * video_fps) if "video_start" in item else 0
    end = math.floor(end_seconds * video_fps) if "video_end" in item else total_frames - 1
    end = min(end, total_frames - 1)
    if start >= end:
        raise ValueError(f"invalid video frame range [{start}, {end}]")
    available = end - start + 1

    if "nframes" in item:
        count = round_to_factor(float(item["nframes"]))
    else:
        requested_fps = float(item.get("fps", DEFAULT_FPS))
        minimum = ceil_to_factor(float(item.get("min_frames", MIN_FRAMES)))
        maximum = floor_to_factor(
            float(item.get("max_frames", min(MAX_FRAMES, available)))
        )
        count = available / video_fps * requested_fps
        count = floor_to_factor(min(min(max(count, minimum), maximum), available))
    if not FRAME_FACTOR <= count <= available:
        raise ValueError(
            f"sampled frame count must be in [{FRAME_FACTOR}, {available}], got {count}"
        )
    if count == 1:
        return [start]
    return [
        round(start + index * (end - start) / (count - 1))
        for index in range(count)
    ]


def probe_decord(path: str, item: dict) -> None:
    from decord import VideoReader

    video = VideoReader(path)
    total_frames = len(video)
    indices = sample_indices(item, total_frames, float(video.get_avg_fps()))
    video.get_batch(indices)


def probe_torchvision(path: str) -> None:
    from torchvision.io import VideoReader

    readable_frames = 0
    for _ in VideoReader(path, "video"):
        readable_frames += 1
    if readable_frames < 2:
        raise ValueError(f"video has {readable_frames} readable frames")


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated video decoder probe")
    parser.add_argument("--item-json", required=True)
    parser.add_argument("--reader", choices=("decord", "torchvision"), required=True)
    args = parser.parse_args()

    item = json.loads(args.item_json)
    path = local_path(str(item["video"]))
    if args.reader == "decord":
        probe_decord(path, item)
    else:
        probe_torchvision(path)


if __name__ == "__main__":
    main()
