"""Fast render-only path: regenerate ASS + ffmpeg from saved stories.json (no ASR/selection)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from config.settings import PROJECT_ROOT, get_settings  # noqa: E402
from modules.outro import normalize_outro_language  # noqa: E402
from modules.shorts import TARGET_LANGUAGES, restyle_existing_shorts  # noqa: E402
from utils.logger import configure_logger, logger  # noqa: E402


def _parse_languages(value: str) -> tuple[str, ...]:
    languages = tuple(
        dict.fromkeys(
            part.strip().lower()
            for part in value.split(",")
            if part.strip()
        )
    )
    invalid = [language for language in languages if language not in TARGET_LANGUAGES]
    if not languages or invalid:
        supported = ",".join(TARGET_LANGUAGES)
        detail = (
            f"unsupported value(s): {','.join(invalid)}"
            if invalid
            else "no languages provided"
        )
        raise argparse.ArgumentTypeError(
            f"{detail}; expected a comma-separated subset of {supported}"
        )
    return languages


def _discover_video_ids() -> list[str]:
    settings = get_settings()
    return sorted(
        path.parent.name
        for path in settings.SHORTS_OUTPUT_DIR.glob("*/stories.json")
    )


def run_render_only(
    video_ids: list[str],
    *,
    publish: bool = True,
    publish_dir: Path | None = None,
    languages: tuple[str, ...] = TARGET_LANGUAGES,
) -> None:
    out_dir = (publish_dir or (PROJECT_ROOT / "out")).resolve()
    for video_id in video_ids:
        logger.info(
            "Render-only: regenerating ASS + MP4 for {} ({})",
            video_id,
            ",".join(languages),
        )
        document = restyle_existing_shorts(
            video_id,
            publish_dir=out_dir if publish else None,
            languages=languages,
        )
        count = sum(
            1
            for story in document.get("stories") or []
            if normalize_outro_language(str(story.get("language") or "en"))
            in languages
        )
        if publish:
            logger.info("Published renders under {} for {}", out_dir, video_id)
        logger.info("Re-rendered {} Shorts for {}", count, video_id)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Skip download, denoise, WhisperX, and clip selection. "
            "Rebuild ASS subtitles and re-render MP4s from data/shorts/<id>/stories.json."
        ),
    )
    parser.add_argument(
        "video_ids",
        nargs="*",
        help="YouTube video IDs (default: every folder with stories.json).",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Do not copy finished MP4s into out/en, out/es, out/ru.",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=None,
        help="Publish root (default: pipeline/out).",
    )
    parser.add_argument(
        "--lang",
        "-Lang",
        type=_parse_languages,
        default=TARGET_LANGUAGES,
        metavar="LANGS",
        help="Comma-separated languages to render (en,es,ru; default: en,es,ru).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logger(get_settings().LOG_LEVEL)
    video_ids = args.video_ids or _discover_video_ids()
    if not video_ids:
        logger.error("No saved stories.json found under data/shorts/")
        return 1
    try:
        run_render_only(
            video_ids,
            publish=not args.no_publish,
            publish_dir=args.publish_dir,
            languages=args.lang,
        )
    except Exception:
        logger.exception("Render-only failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
