"""Runtime configuration loaded from the environment and an optional .env file."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLIP_STRATEGIES: list[str] = [
    "1_categorized",
    "2_viral_hooks",
    "3_controversy",
]

def _caption_font_name() -> str:
    """Heavy face for 1080x1920 captions.

    Prefer bundled Montserrat ExtraBold, then Arial Black, Impact, and Rubik Bold.
    libass only stores one Fontname, so the first face that is actually present wins.
    """
    fonts = PROJECT_ROOT / "assets" / "fonts"
    bundled = (
        ("Montserrat-ExtraBold.ttf", "Montserrat ExtraBold"),
        ("MontserratExtraBold.ttf", "Montserrat ExtraBold"),
        ("ArialBlack.ttf", "Arial Black"),
        ("ariblk.ttf", "Arial Black"),
        ("Impact.ttf", "Impact"),
        ("Rubik-Bold.ttf", "Rubik"),
    )
    for filename, family in bundled:
        if (fonts / filename).is_file():
            return family
    windows_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    system = (
        ("ariblk.ttf", "Arial Black"),
        ("impact.ttf", "Impact"),
        ("Rubik-Bold.ttf", "Rubik"),
    )
    for filename, family in system:
        if (windows_fonts / filename).is_file():
            return family
    return "Montserrat ExtraBold"


# Unified ASS caption style (V4+); colours use &HAABBGGRR. Shared by blur_bg and full_crop.
# Fontsize is in PlayRes pixels on the 1080x1920 canvas.
SUBTITLE_STYLE_ID = "high_contrast_unified"
ASS_CAPTION_FONT_NAME = _caption_font_name()
ASS_CAPTION_FONT_SIZE = 70
ASS_CAPTION_BOLD = 1
ASS_CAPTION_PRIMARY_COLOUR = "&H00FFFFFF"  # white
ASS_CAPTION_SECONDARY_COLOUR = "&H0000FFFF"  # bright yellow karaoke highlight
ASS_CAPTION_OUTLINE_COLOUR = "&H00000000"  # opaque black outline
ASS_CAPTION_BACK_COLOUR = "&H00000000"  # opaque black shadow
ASS_CAPTION_BORDER_STYLE = 1
ASS_CAPTION_OUTLINE = 6.0
ASS_CAPTION_SHADOW = 2
ASS_CAPTION_MARGIN_L = 140
ASS_CAPTION_MARGIN_R = 140
ASS_CAPTION_MARGIN_V = 380

_CHROME_USER_AGENTS: tuple[str, ...] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
)


class Settings(BaseSettings):
    """Phase 1 paths, model names, and acoustic thresholds."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    YTDLP_SLEEP_INTERVAL: int = 30
    COOKIES_FILE_PATH: Path = PROJECT_ROOT / "cookies.txt"
    USER_AGENT: str = _CHROME_USER_AGENTS[0]
    AUDIO_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "audio"
    VIDEO_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "video"
    JSON_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "transcripts"
    SHORTS_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "shorts"
    MODELS_DIR: Path = PROJECT_ROOT / "models"

    WHISPER_MODEL: str = "large-v3"
    WHISPER_LANGUAGE: str | None = None
    WHISPER_INITIAL_PROMPT: str = ""
    WHISPER_BATCH_SIZE: int = Field(default=16, ge=1, le=32)
    WHISPER_COMPUTE_TYPE: str = "float16"
    DEVICE: str = "cuda"

    PYANNOTE_MODEL: str = "pyannote/speaker-diarization-3.1"
    PYANNOTE_ONSET: float = 0.5
    PYANNOTE_OFFSET: float = 0.35
    PYANNOTE_SEGMENTATION_BATCH_SIZE: int = Field(default=8, ge=1, le=32)
    HF_TOKEN: str = Field(
        default="",
        validation_alias=AliasChoices("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    )

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"

    DEEPFILTER_CHUNK_SECONDS: float = 30.0
    DEEPFILTER_OVERLAP_SECONDS: float = 1.0
    TARGET_SAMPLE_RATE: int = 16000
    LOG_LEVEL: str = "INFO"
    MAX_SHORTS: int = Field(default=12, ge=12, le=12)
    VIDEO_LAYOUT_PRESET: str = "blur_bg"
    BRAND_NAME: str = "Example Channel"
    BRAND_WEBSITE_URL: str = "https://example.com"
    BRAND_SOCIAL_HANDLE: str = "@ExampleChannel"
    BRAND_OUTRO_SCHOLAR_BADGE: str = "Example Channel"
    BRAND_OUTRO_HEADLINE_EN: str = "Discover more from\nyour lectures"
    BRAND_OUTRO_TAGLINE_EN: str = (
        "Turn long-form talks into Shorts with captions, outros, and upload-ready metadata."
    )
    BRAND_OUTRO_SECONDARY_CTA_EN: str = "Link in description"
    # Optional extra paragraph appended to the clip-selection system prompt (niche / tone).
    CLIP_SELECTOR_CONTEXT: str = ""
    # JSON object: {"Term": {"es": "...", "ru": "..."}} for translation glossary overrides.
    TRANSLATION_GLOSSARY_JSON: str = ""
    # Comma-separated Latin tokens allowed untranslated in Russian output (e.g. product names).
    TRANSLATION_LATIN_ALLOWLIST: str = ""
    # Short description of content domain for translation prompts (not client-specific).
    TRANSLATION_CONTENT_DOMAIN: str = "long-form spoken video"
    BRAND_OUTRO_PATH: Path = PROJECT_ROOT / "assets" / "outro_rendered.mp4"
    BRAND_OUTRO_TRANSITION_SECONDS: float = Field(default=0.5, ge=0.5, le=2.5)
    # Optional user-supplied outro (skips HTML/Playwright render when the file exists).
    BRAND_OUTRO_CUSTOM_MP4: Path | None = None
    BRAND_OUTRO_CUSTOM_DIR: Path = PROJECT_ROOT / "assets" / "outro" / "custom"
    OUTRO_HTML_PATH: Path = (
        PROJECT_ROOT.parent / "poster" / "stitch_spiritual_video_outro_card_template" / "code.html"
    )
    OUTRO_MUSIC_URL: str = ""
    OUTRO_MUSIC_PATH: Path = PROJECT_ROOT / "assets" / "audio" / "outro_music.mp3"
    OUTRO_MUSIC_START_SECONDS: float = Field(default=0.0, ge=0.0)
    OUTRO_DURATION_SECONDS: float = Field(default=8.0, ge=1.0, le=15.0)
    OUTRO_FPS: int = Field(default=30, ge=12, le=60)

    @field_validator("BRAND_OUTRO_CUSTOM_MP4", mode="before")
    @classmethod
    def _optional_custom_outro_mp4(cls, value: object) -> Path | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return Path(str(value).strip())

    @field_validator("WHISPER_LANGUAGE", mode="before")
    @classmethod
    def _optional_whisper_language(cls, value: object) -> str | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return str(value).strip()

    @field_validator("VIDEO_LAYOUT_PRESET")
    @classmethod
    def _video_layout_preset(cls, value: str) -> str:
        preset = value.strip().lower()
        if preset not in {"blur_bg", "full_crop"}:
            raise ValueError("VIDEO_LAYOUT_PRESET must be 'blur_bg' or 'full_crop'")
        return preset

    @field_validator("DEVICE")
    @classmethod
    def _normalize_device(cls, value: str) -> str:
        device = value.strip().lower()
        if device not in {"cuda", "cpu"}:
            raise ValueError("DEVICE must be 'cuda' or 'cpu'")
        return device

    @field_validator("PYANNOTE_ONSET", "PYANNOTE_OFFSET")
    @classmethod
    def _threshold_range(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("onset and offset must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def _resolve_paths(self) -> "Settings":
        for name in (
            "COOKIES_FILE_PATH",
            "AUDIO_OUTPUT_DIR",
            "VIDEO_OUTPUT_DIR",
            "JSON_OUTPUT_DIR",
            "SHORTS_OUTPUT_DIR",
            "MODELS_DIR",
            "BRAND_OUTRO_PATH",
            "BRAND_OUTRO_CUSTOM_MP4",
            "BRAND_OUTRO_CUSTOM_DIR",
            "OUTRO_HTML_PATH",
            "OUTRO_MUSIC_PATH",
        ):
            path = getattr(self, name)
            if path is None:
                continue
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            setattr(self, name, path.resolve())
        if self.PYANNOTE_OFFSET > self.PYANNOTE_ONSET:
            raise ValueError("PYANNOTE_OFFSET must be less than or equal to PYANNOTE_ONSET")
        return self

    def ensure_output_dirs(self) -> None:
        """Create the directories the pipeline writes into."""
        for path in (
            self.AUDIO_OUTPUT_DIR,
            self.VIDEO_OUTPUT_DIR,
            self.JSON_OUTPUT_DIR,
            self.SHORTS_OUTPUT_DIR,
            self.MODELS_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings object."""
    return Settings()
