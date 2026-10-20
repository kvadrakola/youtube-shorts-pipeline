"""Standalone media transcription and optional custom DeepSeek analysis."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from config.settings import get_settings  # noqa: E402
from modules.clip_selector import _deepseek_client, _segments_text  # noqa: E402
from modules.ingestion import (  # noqa: E402
    _extract_wav,
    _require_ffmpeg,
    download_lecture_audio,
)
from modules.transcriber import transcribe_and_diarize  # noqa: E402
from utils.logger import configure_logger, logger  # noqa: E402

STANDALONE_ROOT = PIPELINE_ROOT / "data" / "standalone"
_SAFE_NAME = re.compile(r"[^\w.-]+", re.UNICODE)


def _is_youtube_url(source: str) -> bool:
    parsed = urlparse(source.strip())
    host = (parsed.hostname or "").lower().removeprefix("www.")
    return parsed.scheme in {"http", "https"} and (
        host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")
    )


def _source_identifier(source: str, is_youtube: bool) -> str:
    if is_youtube:
        parsed = urlparse(source.strip())
        host = (parsed.hostname or "").lower().removeprefix("www.")
        if host == "youtu.be":
            candidate = parsed.path.strip("/").split("/", 1)[0]
        else:
            candidate = (parse_qs(parsed.query).get("v") or [""])[0]
            if not candidate:
                parts = [part for part in parsed.path.split("/") if part]
                candidate = parts[1] if len(parts) > 1 and parts[0] in {"embed", "live", "shorts"} else ""
        raw_name = candidate or "youtube_video"
    else:
        raw_name = Path(source).stem

    safe_name = _SAFE_NAME.sub("_", raw_name).strip(" ._-")
    return safe_name or "media"


def _format_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _plain_transcript(segments: list[dict[str, Any]]) -> str:
    lines = [str(segment.get("text") or "").strip() for segment in segments]
    return "\n".join(line for line in lines if line) + "\n"


def _markdown_transcript(segments: list[dict[str, Any]]) -> str:
    blocks = ["# Transcript"]
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = _format_timestamp(float(segment.get("start", 0.0)))
        end = _format_timestamp(float(segment.get("end", segment.get("start", 0.0))))
        speaker = str(segment.get("speaker") or "UNKNOWN")
        blocks.append(f"## [{start} – {end}] {speaker}\n\n{text}")
    return "\n\n".join(blocks) + "\n"


def _load_custom_prompt(prompt: str | None, prompt_file: str | None) -> str | None:
    if prompt_file is not None:
        path = Path(prompt_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        value = path.read_text(encoding="utf-8-sig")
    else:
        value = prompt or ""
    value = value.strip()
    return value or None


def _prepare_audio(source: str, output_dir: Path, is_youtube: bool) -> Path:
    if is_youtube:
        return Path(
            download_lecture_audio(
                source,
                audio_output_dir=output_dir,
                video_output_dir=output_dir,
                write_shorts_info=False,
            )
        )

    media_path = Path(source).expanduser().resolve()
    if not media_path.is_file():
        raise FileNotFoundError(f"Local media file not found: {media_path}")
    destination = output_dir / f"{_source_identifier(source, False)}_raw.wav"
    _extract_wav(
        _require_ffmpeg(),
        media_path,
        destination,
        get_settings().TARGET_SAMPLE_RATE,
    )
    return destination


def _analyze(transcript: dict[str, Any], prompt: str) -> str:
    settings = get_settings()
    transcript_body = _segments_text(transcript["segments"])
    response = _deepseek_client().chat.completions.create(
        model=settings.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "Analyze the following timestamped transcript:\n\n" + transcript_body,
            },
        ],
        temperature=0.3,
        max_tokens=8192,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("DeepSeek returned an empty response")
    return content


def run(
    source: str,
    *,
    prompt: str | None = None,
    prompt_file: str | None = None,
) -> dict[str, Path]:
    """Transcribe one source and optionally analyze it, returning generated file paths."""
    settings = get_settings()
    configure_logger(settings.LOG_LEVEL)
    custom_prompt = _load_custom_prompt(prompt, prompt_file)
    is_youtube = _is_youtube_url(source)
    if "://" in source and not is_youtube:
        raise ValueError("Only YouTube URLs or local media file paths are supported")

    output_dir = STANDALONE_ROOT / _source_identifier(source, is_youtube)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Standalone output directory: {}", output_dir.resolve())

    audio_path = _prepare_audio(source, output_dir, is_youtube)
    transcribed = transcribe_and_diarize(
        str(audio_path),
        batch_size=32,
        device="cuda",
    )
    transcript = {
        "source": source,
        "audio_path": str(audio_path.resolve()),
        "duration_seconds": transcribed["duration_seconds"],
        "language": transcribed.get("language"),
        "segments": transcribed["segments"],
    }

    paths = {
        "json": output_dir / "transcript.json",
        "text": output_dir / "transcript.txt",
        "markdown": output_dir / "transcript.md",
    }
    paths["json"].write_text(
        json.dumps(transcript, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["text"].write_text(_plain_transcript(transcript["segments"]), encoding="utf-8")
    paths["markdown"].write_text(_markdown_transcript(transcript["segments"]), encoding="utf-8")

    if custom_prompt is not None:
        logger.info("Sending transcript to DeepSeek model {}", settings.DEEPSEEK_MODEL)
        analysis = _analyze(transcript, custom_prompt)
        analysis_path = output_dir / "analysis_result.md"
        analysis_path.write_text(analysis + "\n", encoding="utf-8")
        paths["analysis"] = analysis_path
        preview = analysis if len(analysis) <= 1500 else analysis[:1500].rstrip() + "\n…"
        print(f"\nAnalysis result:\n{preview}\n")

    print("Generated files:")
    for path in paths.values():
        print(f"  {path.resolve()}")
    return paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a YouTube video or local media file, then optionally analyze it with DeepSeek.",
    )
    parser.add_argument("source", help="YouTube URL or local audio/video file path")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", help="Custom DeepSeek analysis prompt")
    prompt_group.add_argument("--prompt-file", help="UTF-8 text file containing the analysis prompt")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run(args.source, prompt=args.prompt, prompt_file=args.prompt_file)
    except Exception:
        logger.exception("Standalone transcription/analysis failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
