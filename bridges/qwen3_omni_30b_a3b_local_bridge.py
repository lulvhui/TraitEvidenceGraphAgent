"""Local Qwen3-Omni-30B-A3B bridge for text-only TraitGraph outputs.

Qwen3-Omni receives every retained interview turn as synchronized video/audio
plus its transcript.  TraitGraph only consumes JSON text, so the speech Talker
is disabled and generation explicitly requests no audio output.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traitgraph.utils import bnb_4bit_skip_modules
from traitgraph.video_safety import prepare_video_messages, set_qwenvl_video_reader


@dataclass
class Qwen3OmniBundle:
    model: Any
    processor: Any
    config: dict
    call_log: list[dict] = field(default_factory=list)
    video_reader_cache: dict[str, str | None] = field(default_factory=dict)


def _ensure_ffmpeg_on_path() -> str:
    found = shutil.which("ffmpeg") or shutil.which("avconv")
    if found:
        return found
    environment_bin = Path(sys.executable).resolve().parent
    candidate = environment_bin / "ffmpeg"
    if candidate.is_file():
        os.environ["PATH"] = (
            f"{environment_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        )
        return str(candidate)
    raise RuntimeError(
        "Qwen3-Omni audio extraction requires ffmpeg in PATH or in the active "
        "Python environment's bin directory."
    )


def _prepare_runtime() -> None:
    cache_dir = os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "traitgraph_qwen3_omni_numba_cache"),
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DECORD_REWIND_RETRY_MAX", "64")
    _ensure_ffmpeg_on_path()


def _torch_dtype(name: str):
    import torch

    normalized = str(name or "auto").lower()
    if normalized == "auto":
        return "auto"
    values = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in values:
        raise ValueError(f"Unsupported dtype: {name}")
    return values[normalized]


def _resolve_local_path(raw_path: str, project_root: str = "") -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        root = Path(project_root).expanduser() if project_root else Path.cwd()
        path = root / path
    return path.resolve()


def _media_uri(raw_path: str, project_root: str, strict: bool) -> str:
    value = str(raw_path or "").strip()
    if not value:
        return ""
    if value.startswith(("file://", "http://", "https://", "data:")):
        return value
    path = _resolve_local_path(value, project_root)
    if strict and not path.is_file():
        raise FileNotFoundError(f"Media file does not exist: {path}")
    return path.as_uri()


def _turn_text(turn: dict) -> str:
    question = str(turn.get("question", "") or "").strip()
    answer = str(turn.get("answer_text", "") or "").strip()
    parts = []
    if question:
        parts.append(f"Question: {question}")
    if answer:
        parts.append(f"Answer: {answer}")
    return "\n".join(parts)


def _build_messages(prompt: str, turns: list[dict], config: dict) -> list[dict]:
    use_audio = bool(config.get("use_audio_in_video", True))
    content: list[dict[str, Any]] = []
    if bool(config.get("include_modality_notice", True)) and turns:
        modalities = (
            "transcript, video frames, and the video's synchronized audio track"
            if use_audio
            else "transcript and video frames"
        )
        content.append({
            "type": "text",
            "text": (
                f"Available raw modalities in this Qwen3-Omni run: {modalities}. "
                "Ground vocal or prosodic claims only in audible evidence.\n\n"
            ),
        })
    content.append({"type": "text", "text": prompt.strip()})

    project_root = str(config.get("project_root", "") or "")
    strict_paths = bool(config.get("strict_media_paths", True))
    video_config = dict(config.get("video", {}) or {})
    for index, turn in enumerate(turns, start=1):
        turn_id = str(turn.get("turn_id", index))
        content.append({"type": "text", "text": f"\n\n[RAW TURN {turn_id}]"})
        video_path = str(turn.get("video_path", "") or "").strip()
        if video_path:
            video_item: dict[str, Any] = {
                "type": "video",
                "video": _media_uri(video_path, project_root, strict_paths),
            }
            for key in ("fps", "min_pixels", "max_pixels", "total_pixels"):
                if video_config.get(key) is not None:
                    video_item[key] = video_config[key]
            content.append(video_item)
        transcript = _turn_text(turn)
        if transcript:
            content.append({"type": "text", "text": f"Transcript:\n{transcript}"})
        elif not video_path:
            content.append({
                "type": "text",
                "text": "(No transcript or media available.)",
            })
    return [{"role": "user", "content": content}]


def _limit_audio_duration(
    audios: Any,
    max_seconds_per_input: float | None,
    sample_rate: int,
) -> tuple[Any, dict[str, int]]:
    if audios is None:
        return None, {"inputs": 0, "truncated_inputs": 0, "max_samples": 0}
    if max_seconds_per_input is None:
        return audios, {
            "inputs": len(audios),
            "truncated_inputs": 0,
            "max_samples": 0,
        }
    seconds = float(max_seconds_per_input)
    if seconds <= 0:
        raise ValueError("audio.max_seconds_per_input must be positive or null")
    if sample_rate <= 0:
        raise ValueError("audio.sample_rate must be positive")
    max_samples = int(seconds * sample_rate)
    limited = []
    truncated = 0
    for waveform in audios:
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
            truncated += 1
        limited.append(waveform)
    return limited, {
        "inputs": len(limited),
        "truncated_inputs": truncated,
        "max_samples": max_samples,
    }


def _generation_kwargs(config: dict, override: dict | None = None) -> dict:
    values = {**dict(config.get("generation", {}) or {}), **dict(override or {})}
    values.setdefault("max_new_tokens", 384)
    values["return_audio"] = False
    values["thinker_return_dict_in_generate"] = True
    values.setdefault("do_sample", False)
    if not values["do_sample"]:
        values.pop("temperature", None)
        values.pop("top_p", None)
        values.pop("top_k", None)
    return values


def load_model(config: dict) -> Qwen3OmniBundle:
    try:
        import torch
        from transformers import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-Omni requires Transformers >= 5.2.0 with "
            "Qwen3OmniMoeForConditionalGeneration support."
        ) from exc

    model_path = str(config.get("model_path", "") or "").strip()
    if not model_path:
        raise ValueError("backbone.model_path is required for Qwen3-Omni")
    model_path = str(_resolve_local_path(
        model_path, str(config.get("project_root", "") or "")
    ))
    local_only = bool(config.get("local_files_only", True))
    if local_only and not Path(model_path).is_dir():
        raise FileNotFoundError(f"Local Qwen3-Omni model path not found: {model_path}")
    if bool(config.get("require_cuda", True)) and not torch.cuda.is_available():
        raise RuntimeError("Qwen3-Omni requires CUDA for this inference configuration")

    reader = str(config.get("video_reader", "") or "").strip()
    if reader:
        os.environ["FORCE_QWENVL_VIDEO_READER"] = reader
    common = {
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
        "local_files_only": local_only,
    }
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path, **common)
    load_kwargs: dict[str, Any] = {
        **common,
        "device_map": config.get("device_map", "auto"),
    }
    attention = str(config.get("attn_implementation", "") or "").strip()
    if attention:
        load_kwargs["attn_implementation"] = attention
    if bool(config.get("load_in_4bit", False)):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Optional 4-bit loading requires bitsandbytes support"
            ) from exc
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(
                config.get("bnb_4bit_use_double_quant", True)
            ),
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=bnb_4bit_skip_modules(config),
        )
    dtype = _torch_dtype(str(config.get("dtype", "bfloat16")))
    try:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, **load_kwargs
        )
    except TypeError:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, **load_kwargs
        )
    if bool(config.get("disable_talker", True)):
        model.disable_talker()
    model.eval()
    return Qwen3OmniBundle(model=model, processor=processor, config=dict(config))


def get_diagnostics(bundle: Qwen3OmniBundle) -> list[dict]:
    return list(bundle.call_log)


def generate(
    bundle: Qwen3OmniBundle,
    role: str,
    prompt: str,
    turns: list[dict],
    generation_config: dict,
) -> str:
    import torch

    _prepare_runtime()
    try:
        from qwen_omni_utils import process_mm_info
    except ImportError as exc:
        raise RuntimeError("Install qwen-omni-utils before Qwen3-Omni inference") from exc

    model, processor, config = bundle.model, bundle.processor, bundle.config
    use_audio = bool(config.get("use_audio_in_video", True))
    if not use_audio:
        raise ValueError("TraitGraph Qwen3-Omni inference requires audio in video")

    messages = _build_messages(prompt, turns, config)
    primary_reader = str(config.get("video_reader", "decord") or "decord")
    fallback_reader = str(
        config.get("fallback_video_reader", primary_reader) or primary_reader
    )
    messages, reader, notices = prepare_video_messages(
        messages,
        primary_reader=primary_reader,
        fallback_reader=fallback_reader,
        image_patch_size=int(config.get("image_patch_size", 14)),
        primary_timeout_seconds=float(config.get("video_probe_timeout_seconds", 15)),
        fallback_timeout_seconds=float(
            config.get("video_fallback_timeout_seconds", 15)
        ),
        cache=bundle.video_reader_cache,
    )
    set_qwenvl_video_reader(reader)
    for notice in notices:
        print(f"warning: Qwen3-Omni video decode fallback: {notice}", flush=True)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(
        messages, use_audio_in_video=use_audio
    )
    audio_config = dict(config.get("audio", {}) or {})
    audios, audio_limit = _limit_audio_duration(
        audios,
        audio_config.get("max_seconds_per_input"),
        int(audio_config.get("sample_rate", 16_000)),
    )
    retained_videos = sum(
        item.get("type") == "video"
        for message in messages
        for item in message.get("content", [])
        if isinstance(item, dict)
    )
    audio_inputs = len(audios) if audios is not None else 0
    if bool(config.get("require_audio_track", True)) and retained_videos:
        if audio_inputs != retained_videos:
            raise RuntimeError(
                "Qwen3-Omni strict A/V preflight failed: "
                f"decoded {audio_inputs} audio tracks for {retained_videos} videos"
            )

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    inputs.pop("token_type_ids", None)
    generation = _generation_kwargs(config, generation_config)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            use_audio_in_video=use_audio,
            **generation,
        )
    text_result = generated[0] if isinstance(generated, tuple) else generated
    sequences = getattr(text_result, "sequences", text_result)
    input_length = inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(
        sequences[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output = decoded[0].strip() if decoded else ""
    bundle.call_log.append({
        "role": role,
        "num_turns": len(turns),
        "video_inputs": retained_videos,
        "audio_inputs": audio_inputs,
        "audio_max_seconds_per_input": audio_config.get("max_seconds_per_input"),
        "audio_truncated_inputs": audio_limit["truncated_inputs"],
        "use_audio_in_video": use_audio,
        "text_only_output": True,
    })
    return output
