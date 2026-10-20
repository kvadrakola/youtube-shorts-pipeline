"""Re-render previously selected Shorts with the current brand treatment."""

from __future__ import annotations

import argparse

from config.settings import get_settings
from modules.shorts import restyle_existing_shorts
from utils.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video_ids",
        nargs="*",
        help="YouTube video IDs to re-render; omit to process every saved story set.",
    )
    args = parser.parse_args()

    settings = get_settings()
    video_ids = args.video_ids or sorted(
        path.parent.name
        for path in settings.SHORTS_OUTPUT_DIR.glob("*/stories.json")
    )
    if not video_ids:
        raise SystemExit("No saved per-video stories.json files found.")

    for video_id in video_ids:
        logger.info("Applying brand subtitle/outro style to {}", video_id)
        document = restyle_existing_shorts(video_id)
        logger.info("Re-rendered {} Shorts for {}", len(document.get("stories") or []), video_id)


if __name__ == "__main__":
    main()
