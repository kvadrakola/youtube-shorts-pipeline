"""Phase 1 orchestrator: ingest, denoise, transcribe, diarize, and cut Shorts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from config.settings import get_settings  # noqa: E402
from modules.audio_cleaner import clean_audio_deepfilternet  # noqa: E402
from modules.ingestion import download_lecture_audio  # noqa: E402
from modules.clip_selector import select_clips_from_transcript  # noqa: E402
from modules.shorts import build_shorts  # noqa: E402
from run_render_only import run_render_only  # noqa: E402
from modules.transcriber import LectureTranscript, transcribe_and_diarize  # noqa: E402
from utils.logger import configure_logger, logger  # noqa: E402

DEFAULT_URL = "https://www.youtube.com/watch?v=VIDEO_ID"
_PIPELINE_VENV = PIPELINE_ROOT / ".venv" / "Scripts" / "python.exe"


def _warn_if_missing_deepfilter() -> None:
    import importlib.util
    import sys

    if importlib.util.find_spec("df") is not None:
        return
    hint = (
        "Run .\\scripts\\install_deepfilter.ps1 or use .\\run_phase1.ps1 "
        "(pipeline .venv already includes DeepFilterNet)."
    )
    if _PIPELINE_VENV.is_file() and Path(sys.executable).resolve() != _PIPELINE_VENV.resolve():
        logger.warning(
            "DeepFilterNet is not installed for {}. {} Raw audio will be used if denoise is skipped.",
            sys.executable,
            hint,
        )
    else:
        logger.warning("DeepFilterNet is not installed. {} Raw audio will be used if denoise is skipped.", hint)


def _transcript_path(cleaned_wav: str) -> Path:
    settings = get_settings()
    stem = Path(cleaned_wav).stem
    if stem.endswith("_cleaned"):
        stem = stem[: -len("_cleaned")]
    if stem.endswith("_raw"):
        stem = stem[: -len("_raw")]
    return settings.JSON_OUTPUT_DIR / f"{stem}.json"


def run_pipeline(
    youtube_url: str,
    *,
    language: str | None = None,
    batch_size: int | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str | None = None,
) -> Path:
    """Run ingestion through Shorts and return the transcript JSON path."""
    settings = get_settings()
    configure_logger(settings.LOG_LEVEL)
    settings.ensure_output_dirs()
    if device:
        get_settings.cache_clear()

    if not settings.HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is empty. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running."
        )
    logger.info("Phase 1 starting for {}", youtube_url)
    raw_wav = download_lecture_audio(youtube_url)
    cleaned_wav = clean_audio_deepfilternet(raw_wav)
    configure_logger(settings.LOG_LEVEL, force=True)
    transcribed = transcribe_and_diarize(
        cleaned_wav,
        language=language,
        batch_size=batch_size,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        device=device,
    )
    document = LectureTranscript(
        source_url=youtube_url,
        audio_clean_path=cleaned_wav,
        duration_seconds=transcribed["duration_seconds"],
        segments=transcribed["segments"],
    )
    output_path = _transcript_path(cleaned_wav)
    payload = document.model_dump(mode="json")
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Transcript written to {}", output_path)
    selection = select_clips_from_transcript(
        payload,
        limit=12,
        min_seconds=30.0,
        max_seconds=60.0,
        model=settings.DEEPSEEK_MODEL,
    )
    shorts_dir = settings.SHORTS_OUTPUT_DIR / output_path.stem
    logger.info("Rendering multilingual shorts -> {}", shorts_dir)
    shorts_manifest = build_shorts(
        payload,
        clips=selection.clips,
        output_dir=shorts_dir,
    )
    logger.info("Shorts output directory: {}", shorts_manifest["output_dir"])
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 audio ingestion, denoising, transcription, diarization, and Shorts.",
    )
    parser.add_argument(
        "youtube_url",
        nargs="?",
        default=DEFAULT_URL,
        help=f"YouTube lecture URL (default: {DEFAULT_URL})",
    )
    parser.add_argument("--language", default=None, help="Force a Whisper language code, for example en")
    parser.add_argument("--batch-size", type=int, default=None, help="Whisper batch size, 1-32")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument(
        "--render-only",
        metavar="VIDEO_ID",
        nargs="?",
        const="__all__",
        help="Skip ingestion/ASR/selection; re-render from data/shorts/<VIDEO_ID>/stories.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logger(get_settings().LOG_LEVEL)
    if args.render_only is not None:
        from run_render_only import _discover_video_ids

        video_ids = (
            _discover_video_ids()
            if args.render_only == "__all__"
            else [args.render_only]
        )
        try:
            run_render_only(video_ids)
        except Exception:
            logger.exception("Render-only failed")
            return 1
        return 0
    _warn_if_missing_deepfilter()
    if args.device:
        import os

        os.environ["DEVICE"] = args.device
        get_settings.cache_clear()
    try:
        output = run_pipeline(
            args.youtube_url,
            language=args.language,
            batch_size=args.batch_size,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            device=args.device,
        )
    except Exception:
        logger.exception("Phase 1 failed")
        return 1
    logger.info("Phase 1 complete: {}", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
