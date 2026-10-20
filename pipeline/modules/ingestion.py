"""YouTube download and 16 kHz mono extraction."""

from __future__ import annotations

import random
import re
from datetime import datetime
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from config.settings import _CHROME_USER_AGENTS, get_settings
from utils.logger import logger

_VIDEO_ID_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def _resolve_user_agent(configured: str) -> str:
    if configured.strip().lower() == "rotate":
        return random.choice(_CHROME_USER_AGENTS)
    return configured


def _http_retry_sleep(n: int = 0, **_kwargs: Any) -> float:
    interval = float(get_settings().YTDLP_SLEEP_INTERVAL)
    return min(interval * (2 ** max(n, 0)), 300.0)


class _YtDlpLogger:
    def debug(self, message: str) -> None:
        if message.startswith("[debug]"):
            return
        logger.debug(message)

    def info(self, message: str) -> None:
        logger.info(message)

    def warning(self, message: str) -> None:
        logger.warning(message)

    def error(self, message: str) -> None:
        logger.error(message)


def _progress(status: dict[str, Any]) -> None:
    state = status.get("status")
    if state == "finished":
        logger.info("yt-dlp finished {}", status.get("filename"))


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH. Install ffmpeg before running ingestion.")
    return ffmpeg


def _sanitize_video_id(video_id: str) -> str:
    cleaned = _VIDEO_ID_PATTERN.sub("", video_id).strip("._-")
    if not cleaned:
        raise RuntimeError("yt-dlp returned an empty video id")
    return cleaned


def _locate_download(info: dict[str, Any], staging: Path, video_id: str) -> Path:
    requested = info.get("requested_downloads") or []
    for item in requested:
        filepath = item.get("filepath")
        if filepath and Path(filepath).is_file():
            return Path(filepath)
    prepared = info.get("_filename")
    if prepared and Path(prepared).is_file():
        return Path(prepared)
    matches = [path for path in staging.glob(f"{video_id}.*") if path.is_file()]
    if not matches:
        raise RuntimeError(f"Downloaded media for {video_id} was not found in {staging}")
    return max(matches, key=lambda path: path.stat().st_size)


def _extract_wav(ffmpeg: str, source: Path, destination: Path, sample_rate: int) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    logger.info("Extracting 16 kHz mono wav to {}", destination)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed to extract wav: {detail}")


def _remux_mp4(ffmpeg: str, source: Path, destination: Path) -> None:
    """Keep a playable mp4 for Shorts. Copy streams when the source is already mp4."""
    if source.suffix.lower() == ".mp4" and source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
        return
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not destination.is_file():
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed to store source video: {detail}")


def _write_shorts_info(
    video_id: str,
    youtube_url: str,
    info: dict[str, Any],
) -> None:
    """Persist download metadata beside future Shorts for this video."""
    settings = get_settings()
    out_dir = settings.SHORTS_OUTPUT_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    title = str(info.get("title") or "").strip()
    description = str(info.get("description") or "").strip()
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    downloaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = (
        f"Download Date: {downloaded_at}\n"
        f"Video Title: {title}\n"
        f"YouTube URL: {watch_url}\n"
        "\n"
        "--- Original Description ---\n"
        f"{description}\n"
    )
    info_path = out_dir / "info.txt"
    info_path.write_text(body, encoding="utf-8")
    logger.info("Wrote shorts metadata to {}", info_path.resolve())


def _video_id_from_url(youtube_url: str) -> str | None:
    parsed = urlparse(youtube_url.strip())
    host = (parsed.hostname or "").lower()
    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/")[0]
    else:
        candidate = (parse_qs(parsed.query).get("v") or [""])[0]
    if not candidate:
        return None
    try:
        return _sanitize_video_id(candidate)
    except RuntimeError:
        return None


def download_lecture_audio(
    youtube_url: str,
    *,
    audio_output_dir: str | Path | None = None,
    video_output_dir: str | Path | None = None,
    write_shorts_info: bool = True,
) -> str:
    """Download source video audio and return the absolute path of the 16 kHz mono wav.

    The matching source video is stored in ``video_output_dir`` as
    ``{id}_source.mp4``. Optional output directories let standalone tools reuse
    ingestion without writing into the main pipeline's data directories.
    """
    if not youtube_url or not youtube_url.strip():
        raise ValueError("youtube_url is empty")

    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed. pip install -r requirements.txt") from exc

    settings = get_settings()
    audio_dir = (
        Path(audio_output_dir).expanduser().resolve()
        if audio_output_dir is not None
        else settings.AUDIO_OUTPUT_DIR
    )
    video_dir = (
        Path(video_output_dir).expanduser().resolve()
        if video_output_dir is not None
        else settings.VIDEO_OUTPUT_DIR
    )
    audio_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    existing_id = _video_id_from_url(youtube_url)
    if existing_id:
        existing_wav = audio_dir / f"{existing_id}_raw.wav"
        existing_video = video_dir / f"{existing_id}_source.mp4"
        if existing_wav.is_file() and existing_video.is_file() and existing_wav.stat().st_size > 0:
            logger.info("Reusing existing download {}", existing_wav.resolve())
            return str(existing_wav.resolve())
    ffmpeg = _require_ffmpeg()
    staging = video_dir / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    sleep_seconds = float(settings.YTDLP_SLEEP_INTERVAL)
    ydl_opts: dict[str, Any] = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(staging / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "restrictfilenames": True,
        "quiet": True,
        "no_warnings": False,
        "sleep_interval": sleep_seconds,
        "max_sleep_interval": sleep_seconds,
        "sleep_interval_requests": sleep_seconds,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 5,
        "file_access_retries": 5,
        "socket_timeout": 30,
        "http_headers": {
            "User-Agent": _resolve_user_agent(settings.USER_AGENT),
            "Accept-Language": "en-US,en;q=0.9",
        },
        "retry_sleep_functions": {
            "http": _http_retry_sleep,
            "fragment": _http_retry_sleep,
            "extractor": _http_retry_sleep,
        },
        "logger": _YtDlpLogger(),
        "progress_hooks": [_progress],
        "ffmpeg_location": str(Path(ffmpeg).parent),
    }
    cookies = settings.COOKIES_FILE_PATH
    if cookies.is_file():
        ydl_opts["cookiefile"] = str(cookies)
        logger.info("Using cookies file {}", cookies)
    else:
        logger.warning("Cookies file not found at {}. Continuing without cookies.", cookies)

    logger.info("Downloading {}", youtube_url)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
    except DownloadError as exc:
        message = str(exc)
        if "429" in message:
            logger.error(
                "YouTube returned HTTP 429. Increase YTDLP_SLEEP_INTERVAL or provide cookies.txt."
            )
        raise RuntimeError(f"yt-dlp download failed: {message}") from exc

    if info is None:
        raise RuntimeError("yt-dlp returned no metadata")
    entries = info.get("entries")
    if entries:
        info = next(item for item in entries if item)
    video_id = _sanitize_video_id(str(info.get("id") or ""))
    source = _locate_download(info, staging, video_id)

    wav_path = audio_dir / f"{video_id}_raw.wav"
    video_path = video_dir / f"{video_id}_source.mp4"
    _extract_wav(ffmpeg, source, wav_path, settings.TARGET_SAMPLE_RATE)
    _remux_mp4(ffmpeg, source, video_path)
    if write_shorts_info:
        _write_shorts_info(video_id, youtube_url, info)
    if source.resolve() not in {wav_path.resolve(), video_path.resolve()}:
        source.unlink(missing_ok=True)

    logger.info("Raw audio saved to {}", wav_path.resolve())
    logger.info("Source video saved to {}", video_path.resolve())
    return str(wav_path.resolve())
