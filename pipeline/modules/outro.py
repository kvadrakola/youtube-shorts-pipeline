"""Build the branded outro clip from the HTML card and a YouTube music reference."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from config.settings import get_settings
from utils.logger import logger

_WIDTH = 1080
_HEIGHT = 1920
# The HTML card is authored at 480px wide; 2.25x device pixels yields 1080x1920.
_CSS_WIDTH = 480
_CSS_HEIGHT = 853
_DEVICE_SCALE = _WIDTH / _CSS_WIDTH
_RENDER_VERSION = 8

OUTRO_TARGET_LANGUAGES = ("en", "es", "ru")

def _brand_website_host() -> str:
    url = get_settings().BRAND_WEBSITE_URL.strip()
    return url.replace("https://", "").replace("http://", "").rstrip("/")


def get_outro_language_copy() -> dict[str, dict[str, str]]:
    """Localized outro strings driven by ``BRAND_*`` settings."""
    settings = get_settings()
    host = _brand_website_host()
    name = settings.BRAND_NAME.strip()
    handle = settings.BRAND_SOCIAL_HANDLE.strip()
    badge = settings.BRAND_OUTRO_SCHOLAR_BADGE.strip() or name
    headline_en = settings.BRAND_OUTRO_HEADLINE_EN.strip() or "Discover more"
    tagline_en = settings.BRAND_OUTRO_TAGLINE_EN.strip()
    footer = f"© {name} · {host}" if name and host else f"© {name or host}"
    secondary_en = settings.BRAND_OUTRO_SECONDARY_CTA_EN.strip() or "Link in description"
    return {
        "en": {
            "lecture_cta": f"Full lecture & subscribe at {host}",
            "secondary_cta": secondary_en,
            "scholar_badge": badge,
            "headline": headline_en,
            "tagline": tagline_en,
            "website_label": host,
            "social_handle": handle,
            "footer_line": footer,
        },
        "es": {
            "lecture_cta": f"Conferencia completa y suscripción en {host}",
            "secondary_cta": "Enlace en la descripción",
            "scholar_badge": badge,
            "headline": "Descubre más de\ntus conferencias",
            "tagline": (
                "Convierte charlas largas en Shorts con subtítulos, outro y metadatos listos."
            ),
            "website_label": host,
            "social_handle": handle,
            "footer_line": footer,
        },
        "ru": {
            "lecture_cta": f"Полная лекция и подписка на {host}",
            "secondary_cta": "Ссылка в описании",
            "scholar_badge": badge,
            "headline": "Больше материала\nиз ваших лекций",
            "tagline": (
                "Длинные выступления в Shorts: субтитры, аутро и метаданные для загрузки."
            ),
            "website_label": host,
            "social_handle": handle,
            "footer_line": footer,
        },
    }


def normalize_outro_language(language: str | None) -> str:
    """Map pipeline language codes to supported outro locales."""
    code = (language or "en").strip().lower()
    if code in get_outro_language_copy():
        return code
    return "en"


def outro_copy_for_language(language: str | None) -> dict[str, str]:
    return get_outro_language_copy()[normalize_outro_language(language)]


def outro_video_path(language: str | None = None) -> Path:
    """Per-language rendered outro MP4 under ``assets/``."""
    settings = get_settings()
    lang = normalize_outro_language(language)
    base = settings.BRAND_OUTRO_PATH
    return base.with_name(f"{base.stem}_{lang}{base.suffix}")


def resolve_custom_outro_mp4(language: str | None = None) -> Path | None:
    """Return a user-provided outro MP4 when configured, else ``None``."""
    settings = get_settings()
    lang = normalize_outro_language(language)
    if settings.BRAND_OUTRO_CUSTOM_MP4 and settings.BRAND_OUTRO_CUSTOM_MP4.is_file():
        return settings.BRAND_OUTRO_CUSTOM_MP4.resolve()
    per_lang = settings.BRAND_OUTRO_CUSTOM_DIR / f"outro_{lang}.mp4"
    if per_lang.is_file():
        return per_lang.resolve()
    return None


def resolve_outro_video_path(language: str | None = None) -> Path:
    """Outro clip used during Shorts render: custom MP4 or rendered cache path."""
    custom = resolve_custom_outro_mp4(language)
    if custom is not None:
        return custom
    return outro_video_path(language)

_OUTRO_MUSIC_FADE_IN_SECONDS = 0.8
_OUTRO_MUSIC_FADE_OUT_SECONDS = 0.8
_OUTRO_MUSIC_VOLUME = 0.4


def outro_music_audio_filter_chain(duration: float) -> str:
    """Attenuate outro music to 40% and fade in/out (8s clip fades out from 7.2s)."""
    fade_out_start = max(0.0, duration - _OUTRO_MUSIC_FADE_OUT_SECONDS)
    return (
        f"volume={_OUTRO_MUSIC_VOLUME},"
        f"afade=t=in:st=0:d={_OUTRO_MUSIC_FADE_IN_SECONDS},"
        f"afade=t=out:st={fade_out_start:.1f}:d={_OUTRO_MUSIC_FADE_OUT_SECONDS}"
    )

_CAPTURE_CSS = """
html, body {
  width: 100% !important;
  height: 100% !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
  background: #030409 !important;
}
.shorts-canvas {
  width: 100vw !important;
  height: 100vh !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}
"""

_WAIT_FOR_ASSETS_JS = """
async () => {
  await document.fonts.ready;
  const results = await Promise.all(
    Array.from(document.images).map(async (img) => {
      if (!img.complete || !img.naturalWidth) {
        try { await img.decode(); } catch (error) { return false; }
      }
      return img.naturalWidth > 0;
    })
  );
  return results.filter((ok) => !ok).length;
}
"""

_SEEK_ANIMATIONS_JS = """
(ms) => {
  for (const animation of document.getAnimations()) {
    animation.pause();
    animation.currentTime = ms;
  }
}
"""


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH")
    return ffmpeg


def _media_duration_seconds(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not path.is_file():
        return None
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    try:
        return float(probe.stdout.strip())
    except ValueError:
        return None


def _music_filter_stamp_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.name}.filter")


def _music_filter_is_current(destination: Path, duration: float) -> bool:
    stamp = _music_filter_stamp_path(destination)
    if not stamp.is_file():
        return False
    try:
        recorded = stamp.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return recorded == outro_music_audio_filter_chain(duration)


def _write_silent_outro_music(destination: Path, duration: float, *, force: bool = False) -> Path:
    """Create a stereo silent MP3 for HTML outro muxing when no music is configured."""
    ffmpeg = _require_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        existing_duration = _media_duration_seconds(destination)
        if existing_duration is not None and abs(existing_duration - duration) <= 0.15:
            return destination
    temporary = destination.with_name(f".{destination.stem}.silent.tmp{destination.suffix}")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=channel_layout=stereo:sample_rate=48000",
        "-t",
        f"{duration:.3f}",
        "-ac",
        "2",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(temporary),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not temporary.is_file():
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed to create silent outro audio: {detail}")
    temporary.replace(destination)
    _music_filter_stamp_path(destination).write_text("silent\n", encoding="utf-8")
    logger.info("Using silent outro audio ({:.1f}s) at {}", duration, destination)
    return destination


def ensure_outro_music_file(
    *,
    destination: Path | None = None,
    duration: float | None = None,
    force: bool = False,
) -> Path:
    """Return outro audio: user file, downloaded URL, or generated silence."""
    settings = get_settings()
    destination = destination or settings.OUTRO_MUSIC_PATH
    duration = duration or settings.OUTRO_DURATION_SECONDS
    url = (settings.OUTRO_MUSIC_URL or "").strip()
    if url:
        return download_outro_music(url=url, destination=destination, duration=duration, force=force)
    if destination.is_file() and not force:
        existing_duration = _media_duration_seconds(destination)
        if existing_duration is not None and abs(existing_duration - duration) <= 0.15:
            return destination
    return _write_silent_outro_music(destination, duration, force=force)


def download_outro_music(
    url: str | None = None,
    destination: Path | None = None,
    *,
    start_seconds: float | None = None,
    duration: float | None = None,
    force: bool = False,
) -> Path:
    """Download the reference track with yt-dlp and cut a looped, faded snippet to MP3."""
    settings = get_settings()
    url = (url or settings.OUTRO_MUSIC_URL or "").strip()
    if not url:
        raise RuntimeError(
            "OUTRO_MUSIC_URL is empty. Set OUTRO_MUSIC_URL, place audio at OUTRO_MUSIC_PATH, "
            "or rely on ensure_outro_music_file() for silent audio."
        )
    destination = destination or settings.OUTRO_MUSIC_PATH
    start_seconds = settings.OUTRO_MUSIC_START_SECONDS if start_seconds is None else start_seconds
    duration = duration or settings.OUTRO_DURATION_SECONDS
    if destination.is_file() and not force:
        existing_duration = _media_duration_seconds(destination)
        filter_current = _music_filter_is_current(destination, duration)
        if (
            existing_duration is not None
            and abs(existing_duration - duration) <= 0.15
            and filter_current
        ):
            return destination
        if existing_duration is not None:
            reason = "audio filter changed" if not filter_current else "duration mismatch"
            logger.info(
                "Re-cutting outro music ({}; {:.2f}s on disk, {:.1f}s expected)",
                reason,
                existing_duration,
                duration,
            )

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed. pip install -r requirements.txt") from exc

    ffmpeg = _require_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="outro_music_") as temp_dir:
        ydl_opts: dict[str, object] = {
            "format": "bestaudio/best",
            "outtmpl": str(Path(temp_dir) / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "noprogress": True,
            "no_warnings": True,
            "http_headers": {"User-Agent": settings.USER_AGENT},
            "ffmpeg_location": str(Path(ffmpeg).parent),
        }
        if settings.COOKIES_FILE_PATH.is_file():
            ydl_opts["cookiefile"] = str(settings.COOKIES_FILE_PATH)
        logger.info("Downloading outro music from {}", url)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
        sources = sorted(Path(temp_dir).glob("source.*"))
        if not sources:
            raise RuntimeError(f"yt-dlp produced no audio file for {url}")

        audio_filter = f"{outro_music_audio_filter_chain(duration)},aresample=48000"
        temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-stream_loop",
            "-1",
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            str(sources[0]),
            "-vn",
            "-t",
            f"{duration:.3f}",
            "-af",
            audio_filter,
            "-ac",
            "2",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(temporary),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not temporary.is_file():
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"ffmpeg failed to cut outro music: {detail}")
        temporary.replace(destination)
        _music_filter_stamp_path(destination).write_text(
            outro_music_audio_filter_chain(duration) + "\n",
            encoding="utf-8",
        )
    logger.info("Saved {:.1f}s outro music to {}", duration, destination)
    return destination


def _apply_outro_copy_js(language: str) -> str:
    copy = outro_copy_for_language(language)
    payload = json.dumps(copy, ensure_ascii=False)
    lang = normalize_outro_language(language)
    return f"""
() => {{
  const copy = {payload};
  const lecture = document.getElementById("outro-lecture-cta");
  const secondary =
    document.getElementById("outro-secondary-cta") ||
    document.getElementById("outro-patreon-cta");
  const badge = document.getElementById("outro-scholar-badge");
  const headline = document.getElementById("outro-headline");
  const tagline = document.getElementById("outro-tagline");
  if (lecture) lecture.textContent = copy.lecture_cta;
  if (secondary) secondary.textContent = copy.secondary_cta;
  if (badge) badge.textContent = copy.scholar_badge;
  if (tagline) tagline.textContent = copy.tagline;
  const website = document.getElementById("outro-brand-website");
  const handle = document.getElementById("outro-brand-handle");
  const footer = document.getElementById("outro-footer-copyright");
  if (website) website.textContent = copy.website_label;
  if (handle) handle.textContent = copy.social_handle;
  if (footer) footer.textContent = copy.footer_line;
  if (headline) {{
    headline.replaceChildren();
    copy.headline.split("\\n").forEach((line, index) => {{
      if (index) headline.appendChild(document.createElement("br"));
      headline.appendChild(document.createTextNode(line));
    }});
  }}
  document.documentElement.lang = "{lang}";
  for (const node of document.querySelectorAll(".outro-cta-subline")) {{
    node.style.display = "none";
  }}
}}
"""


def _cache_key(
    html_path: Path,
    music_path: Path,
    duration: float,
    fps: int,
    language: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(html_path.read_bytes())
    digest.update(music_path.read_bytes())
    digest.update(
        json.dumps(
            {
                "version": _RENDER_VERSION,
                "duration": duration,
                "fps": fps,
                "size": [_WIDTH, _HEIGHT],
                "language": normalize_outro_language(language),
                "copy": outro_copy_for_language(language),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _cache_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.stem}.cache.json")


def _is_cached(destination: Path, key: str) -> bool:
    cache = _cache_path(destination)
    if not destination.is_file() or not cache.is_file():
        return False
    try:
        return json.loads(cache.read_text(encoding="utf-8")).get("key") == key
    except (OSError, ValueError):
        return False


def _delete_outro_render_artifacts(destination: Path) -> None:
    """Remove a stale rendered outro MP4 and its sidecar cache."""
    for path in (destination, _cache_path(destination)):
        if not path.is_file():
            continue
        try:
            path.unlink()
            logger.info("Removed stale outro artifact {}", path)
        except OSError as exc:
            logger.warning("Could not delete {}: {}", path, exc)


def render_html_outro_to_video(
    html_path: Path | None = None,
    music_path: Path | None = None,
    destination: Path | None = None,
    *,
    language: str = "en",
    duration: float | None = None,
    fps: int | None = None,
    force: bool = False,
) -> Path:
    """Capture the HTML outro frame-by-frame at 1080x1920 and mux it with the outro music.

    Animations are paused and seeked to each frame time, so the capture is exact
    regardless of how fast the browser can take screenshots.
    """
    settings = get_settings()
    html_path = html_path or settings.OUTRO_HTML_PATH
    music_path = music_path or settings.OUTRO_MUSIC_PATH
    language = normalize_outro_language(language)
    destination = destination or outro_video_path(language)
    duration = duration or settings.OUTRO_DURATION_SECONDS
    fps = fps or settings.OUTRO_FPS
    if not html_path.is_file():
        raise FileNotFoundError(f"Outro HTML not found: {html_path}")
    if not music_path.is_file():
        music_path = ensure_outro_music_file(destination=music_path, duration=duration, force=force)

    key = _cache_key(html_path, music_path, duration, fps, language)
    if not force and _is_cached(destination, key):
        logger.info("Using cached outro {}", destination)
        return destination
    if destination.is_file() or _cache_path(destination).is_file():
        _delete_outro_render_artifacts(destination)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "playwright is not installed. pip install -r requirements.txt && "
            "python -m playwright install chromium"
        ) from exc

    ffmpeg = _require_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    frame_count = int(round(duration * fps))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "image2pipe",
        "-framerate",
        str(fps),
        "-c:v",
        "png",
        "-i",
        "-",
        "-i",
        str(music_path),
        "-filter_complex",
        (
            f"[0:v]scale={_WIDTH}:{_HEIGHT}:flags=lanczos,setsar=1,fps={fps},format=yuv420p[v];"
            "[1:a]aresample=48000,aformat=channel_layouts=stereo,apad[a]"
        ),
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(temporary),
    ]

    logger.info("Rendering {} outro frames ({}) from {}", frame_count, language, html_path)
    with tempfile.TemporaryFile() as stderr_log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=stderr_log)
        assert process.stdin is not None
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                try:
                    page = browser.new_page(
                        viewport={"width": _CSS_WIDTH, "height": _CSS_HEIGHT},
                        device_scale_factor=_DEVICE_SCALE,
                    )
                    page.goto(html_path.resolve().as_uri(), wait_until="networkidle", timeout=90_000)
                    page.add_style_tag(content=_CAPTURE_CSS)
                    page.evaluate(_apply_outro_copy_js(language))
                    broken_images = page.evaluate(_WAIT_FOR_ASSETS_JS)
                    if broken_images:
                        logger.warning("{} outro image(s) failed to load", broken_images)
                    page.wait_for_timeout(300)
                    for index in range(frame_count):
                        page.evaluate(_SEEK_ANIMATIONS_JS, index * 1000.0 / fps)
                        process.stdin.write(page.screenshot(type="png"))
                finally:
                    browser.close()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
        returncode = process.wait()
        stderr_log.seek(0)
        detail = stderr_log.read().decode("utf-8", errors="replace").strip()
    if returncode != 0 or not temporary.is_file():
        raise RuntimeError(f"ffmpeg failed to encode outro: {detail}")

    temporary.replace(destination)
    _cache_path(destination).write_text(
        json.dumps(
            {
                "key": key,
                "html": str(html_path),
                "music": str(music_path),
                "language": language,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Rendered outro {}", destination)
    return destination


def static_outro_path() -> Path:
    """Path to the pre-rendered outro MP4 checked into or produced under ``assets/``."""
    return get_settings().BRAND_OUTRO_PATH


def outro_clip_duration(path: Path | None = None) -> float:
    """Duration of the outro clip in seconds (from the file when possible)."""
    settings = get_settings()
    target = path or static_outro_path()
    measured = _media_duration_seconds(target) if target.is_file() else None
    if measured is not None and measured > 0.05:
        return measured
    return settings.OUTRO_DURATION_SECONDS


def ensure_outro_video(*, language: str = "en", force: bool = False) -> Path:
    """Return the localized outro MP4, rebuilding when the render cache is stale.

    When ``BRAND_OUTRO_CUSTOM_MP4`` or ``BRAND_OUTRO_CUSTOM_DIR/outro_<lang>.mp4`` exists,
    that file is returned and HTML rendering is skipped.

    Otherwise a matching ``outro_rendered_<lang>.mp4`` and cache key skips yt-dlp and Playwright.
    Audio-filter, HTML, or copy changes bump the cache key and force a fresh render.
    """
    settings = get_settings()
    language = normalize_outro_language(language)
    custom = resolve_custom_outro_mp4(language)
    if custom is not None and not force:
        logger.info("Using custom outro MP4 for {}: {}", language, custom)
        return custom
    destination = outro_video_path(language)
    html_path = settings.OUTRO_HTML_PATH
    music_path = settings.OUTRO_MUSIC_PATH
    if (
        not force
        and destination.is_file()
        and music_path.is_file()
        and html_path.is_file()
        and _is_cached(
            destination,
            _cache_key(
                html_path,
                music_path,
                settings.OUTRO_DURATION_SECONDS,
                settings.OUTRO_FPS,
                language,
            ),
        )
    ):
        logger.info("Using cached outro {} ({})", destination, language)
        return destination
    ensure_outro_music_file(
        destination=music_path,
        duration=settings.OUTRO_DURATION_SECONDS,
        force=force or not music_path.is_file(),
    )
    return render_html_outro_to_video(language=language, force=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Render the branded outro HTML card to pipeline/assets/outro_rendered.mp4.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-build outro audio and re-capture outro frames even if cached MP4s exist.",
    )
    parser.add_argument(
        "--language",
        choices=OUTRO_TARGET_LANGUAGES,
        default="en",
        help="Target locale for outro copy (default: en).",
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Render outro_rendered_<lang>.mp4 for en, es, and ru.",
    )
    args = parser.parse_args()
    if args.all_languages:
        for language in OUTRO_TARGET_LANGUAGES:
            path = ensure_outro_video(language=language, force=args.force)
            print(path.resolve())
        return
    path = ensure_outro_video(language=args.language, force=args.force)
    print(path.resolve())


if __name__ == "__main__":
    main()
