"""Select self-contained clip windows and render 9:16 captioned Shorts."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from config.settings import (
    ASS_CAPTION_BACK_COLOUR,
    ASS_CAPTION_BOLD,
    ASS_CAPTION_BORDER_STYLE,
    ASS_CAPTION_FONT_NAME,
    ASS_CAPTION_FONT_SIZE,
    ASS_CAPTION_MARGIN_L,
    ASS_CAPTION_MARGIN_R,
    ASS_CAPTION_MARGIN_V,
    ASS_CAPTION_OUTLINE,
    ASS_CAPTION_OUTLINE_COLOUR,
    ASS_CAPTION_PRIMARY_COLOUR,
    ASS_CAPTION_SECONDARY_COLOUR,
    ASS_CAPTION_SHADOW,
    PROJECT_ROOT,
    SUBTITLE_STYLE_ID,
    get_settings,
)
from modules.clip_selector import (
    SelectedClip,
    YoutubeSeo,
    select_clips_from_transcript,
    slugify_title,
)
from modules.outro import ensure_outro_video, normalize_outro_language, outro_clip_duration
from modules.translator import find_untranslated_english, translate_subtitle_lines
from utils.logger import logger

_SENTENCE_END = re.compile(r"[.!?]$")
_HOOK_START = re.compile(
    r"^(what|why|how|when|who|if|imagine|listen|here|this|the reason|so)\b",
    re.IGNORECASE,
)

_SUBTITLE_LINE_CHARS = 32
_SUBTITLE_MAX_CHARS = 58
_SUBTITLE_MAX_WORDS = 10
_SUBTITLE_MAX_SECONDS = 3.6
_SUBTITLE_EVENT_MAX_SECONDS = 2.5
_SUBTITLE_GAP_CLOSE_SECONDS = 0.6
_SILENCE_HIDE_SECONDS = 0.6
# Long recitation or pause shows up as a hole longer than this. Captions must
# clear for the whole hole; the next translated word must not be held across it.
_SPEECH_GAP_CLEAR_SECONDS = 1.2
_WORD_STRETCH_MAX_SECONDS = _SPEECH_GAP_CLEAR_SECONDS
_WORD_TIGHT_WINDOW_SECONDS = 0.35
_GHOST_ZERO_START_CLIP_OFFSET = 1.0
_CAPTION_FADE_IN_MS = 120
_CAPTION_FADE_OUT_MS = 150
_PLAYRES_Y = 1920
_CAPTION_BASELINE_Y = _PLAYRES_Y - ASS_CAPTION_MARGIN_V
_FONT_DIR = PROJECT_ROOT / "assets" / "fonts"
TARGET_LANGUAGES = ("en", "es", "ru")
# Bump when localized metadata generation changes so render-only refreshes it.
LOCALIZATION_VERSION = 2
# YouTube only shelves a video as a Short when this exact hashtag is present.
_PLATFORM_HASHTAGS = frozenset({"#shorts"})
DESCRIPTION_COPY: dict[str, dict[str, str]] = {
    "en": {
        "title": "[TITLE]",
        "description": "[DESCRIPTION]",
        "pinned_comment": "[PINNED COMMENT]",
        "tags": "[TAGS]",
        "watch_full": "🎬 Watch full video:",
        "website": "🌐 Website:",
    },
    "es": {
        "title": "[TÍTULO]",
        "description": "[DESCRIPCIÓN]",
        "pinned_comment": "[COMENTARIO FIJADO]",
        "tags": "[ETIQUETAS]",
        "watch_full": "🎬 Mira el video completo:",
        "website": "🌐 Sitio web:",
    },
    "ru": {
        "title": "[НАЗВАНИЕ]",
        "description": "[ОПИСАНИЕ]",
        "pinned_comment": "[ЗАКРЕПЛЁННЫЙ КОММЕНТАРИЙ]",
        "tags": "[ТЕГИ]",
        "watch_full": "🎬 Полное видео:",
        "website": "🌐 Сайт:",
    },
}
# The 12-clip allocation retained for callers that display the default matrix.
LANGUAGE_CLIP_RANKS: dict[str, tuple[int, ...]] = {
    "en": (1, 2, 3, 4, 7, 10),
    "es": (1, 2, 3, 5, 8, 11),
    "ru": (1, 2, 3, 6, 9, 12),
}


def allocate_clips_by_language(
    clips: list[SelectedClip],
) -> dict[str, list[SelectedClip]]:
    """Render the Golden Top in every language, then distribute the rest evenly."""
    if len(clips) < 6:
        raise ValueError(f"Language allocation requires at least 6 clips, got {len(clips)}")
    ranked = sorted(clips, key=lambda clip: int(clip.rank or 10_000))
    allocated = {language: list(ranked[:3]) for language in TARGET_LANGUAGES}
    for index, clip in enumerate(ranked[3:]):
        allocated[TARGET_LANGUAGES[index % len(TARGET_LANGUAGES)]].append(clip)
    return allocated


def _words(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in segments:
        for word in segment.get("words") or []:
            if "start" in word and "end" in word and word.get("word"):
                words.append(word)
    words.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return words


def _main_speaker(words: list[dict[str, Any]]) -> str:
    totals: dict[str, float] = {}
    for word in words:
        speaker = str(word.get("speaker") or "UNKNOWN")
        totals[speaker] = totals.get(speaker, 0.0) + max(float(word["end"]) - float(word["start"]), 0.0)
    if not totals:
        return "UNKNOWN"
    return max(totals, key=totals.get)


def _sentences(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split words on sentence punctuation or a pause longer than 1.2 seconds."""
    sentences: list[dict[str, Any]] = []
    bucket: list[dict[str, Any]] = []
    for word in words:
        if bucket:
            gap = float(word["start"]) - float(bucket[-1]["end"])
            if gap > 1.2:
                sentences.append(_pack_sentence(bucket))
                bucket = []
        bucket.append(word)
        if _SENTENCE_END.search(str(word["word"]).strip()):
            sentences.append(_pack_sentence(bucket))
            bucket = []
    if bucket:
        sentences.append(_pack_sentence(bucket))
    return [item for item in sentences if item["text"]]


def _pack_sentence(words: list[dict[str, Any]]) -> dict[str, Any]:
    text = " ".join(str(word["word"]).strip() for word in words).strip()
    return {
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "text": text,
        "words": words,
        "speaker": str(words[0].get("speaker") or "UNKNOWN"),
    }


def _window_from_sentences(sentences: list[dict[str, Any]], speaker: str) -> dict[str, Any]:
    text = " ".join(item["text"] for item in sentences).strip()
    hook = sentences[0]["text"]
    duration = float(sentences[-1]["end"]) - float(sentences[0]["start"])
    score = 0.0
    if _HOOK_START.search(hook):
        score += 2.0
    if "?" in hook:
        score += 1.5
    score += min(duration, 55.0) / 55.0
    score += min(len(text.split()), 120) / 120.0
    return {
        "start": round(float(sentences[0]["start"]), 3),
        "end": round(float(sentences[-1]["end"]), 3),
        "speaker": speaker,
        "hook": hook,
        "narration": text,
        "score": round(score, 4),
        "words": [word for sentence in sentences for word in sentence["words"]],
    }


def select_story_windows(
    segments: list[dict[str, Any]],
    *,
    min_seconds: float = 25.0,
    max_seconds: float = 55.0,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Pick non-overlapping sentence windows that can stand alone as a Short."""
    speaker = _main_speaker(_words(segments))
    speaker_words = [word for word in _words(segments) if str(word.get("speaker") or "UNKNOWN") == speaker]
    sentences = _sentences(speaker_words)
    candidates: list[dict[str, Any]] = []
    for start_index, _sentence in enumerate(sentences):
        chosen = [sentences[start_index]]
        if chosen[0]["end"] - chosen[0]["start"] > max_seconds:
            continue
        for follow in sentences[start_index + 1 :]:
            if follow["start"] - chosen[-1]["end"] > 2.0:
                break
            span = follow["end"] - chosen[0]["start"]
            if span > max_seconds:
                break
            chosen.append(follow)
        duration = chosen[-1]["end"] - chosen[0]["start"]
        if duration < min_seconds or duration > max_seconds:
            continue
        if len(chosen[0]["text"].split()) < 5:
            continue
        candidates.append(_window_from_sentences(chosen, speaker))
    if not candidates and sentences:
        whole = sentences[-1]["end"] - sentences[0]["start"]
        if 8.0 <= whole <= max_seconds:
            candidates.append(_window_from_sentences(sentences, speaker))
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(_overlaps(candidate, kept) for kept in selected):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: item["start"])
    for index, story in enumerate(selected, start=1):
        story["id"] = f"short_{index:02d}"
        story["title"] = _title_from_hook(story["hook"])
    return selected


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return float(left["start"]) < float(right["end"]) and float(right["start"]) < float(left["end"])


def _title_from_hook(hook: str) -> str:
    words = hook.replace("?", "").replace(".", "").split()
    return " ".join(words[:8]).strip() or "Short clip"


def _escape_ass_path(path: Path) -> str:
    """Escape a caption path for an ffmpeg filtergraph, including the Windows drive colon."""
    text = path.resolve().as_posix().replace("'", r"\'").replace(":", r"\:")
    return text


def _ass_timestamp(seconds: float) -> str:
    millis = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours}:{minutes:02d}:{secs:02d}.{ms // 10:02d}"


def _parse_ass_timestamp(value: str) -> float:
    hours, minutes, rest = value.strip().split(":")
    seconds, centiseconds = rest.split(".")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(centiseconds) / 100.0
    )


_ASS_KARAOKE_TOKEN = re.compile(
    r"\{\\k(\d+)(?:\\[^}]+)*\}([^\\{]+)",
    re.UNICODE,
)


def _words_from_ass_file(ass_path: Path, clip_start: float) -> list[dict[str, Any]]:
    """Rebuild caption word timings from an existing karaoke ASS file."""
    words: list[dict[str, Any]] = []
    for line in ass_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        event_start = _parse_ass_timestamp(parts[1])
        text = parts[9]
        cursor = event_start
        for match in _ASS_KARAOKE_TOKEN.finditer(text):
            duration = int(match.group(1)) / 100.0
            token = match.group(2).replace(r"\N", " ").strip()
            if not token:
                cursor += duration
                continue
            words.append(
                {
                    "word": token,
                    "start": clip_start + cursor,
                    "end": clip_start + cursor + duration,
                }
            )
            cursor += duration
    return words


def _story_media_paths(story: dict[str, Any], shorts_root: Path) -> tuple[Path, Path, Path]:
    """Return ``(ass_path, destination_mp4, temporary_mp4)`` for a saved story record."""
    file_stem = str(story["id"])
    language_dir = Path(story["output_dir"]) if story.get("output_dir") else shorts_root
    if story.get("video_path"):
        destination = Path(story["video_path"])
        language_dir = destination.parent
    else:
        destination = _short_mp4_path(language_dir, file_stem)
        if not destination.is_file():
            flat = language_dir / f"{file_stem}.mp4"
            if flat.is_file():
                destination = flat
            else:
                nested = _short_mp4_path(shorts_root, file_stem)
                if nested.is_file():
                    destination = nested
                    language_dir = nested.parent

    ass_path = language_dir / f"{file_stem}.ass"
    if not ass_path.is_file():
        ass_path = _short_ass_path(language_dir, file_stem)
    if not ass_path.is_file():
        ass_path = _short_ass_path(shorts_root, file_stem)

    temporary = language_dir / f".{file_stem}.brand.mp4"
    return ass_path, destination, temporary


def _caption_words_for_restyle(
    story: dict[str, Any],
    transcript: dict[str, Any] | None,
    ass_path: Path,
) -> list[dict[str, Any]]:
    """Words for ASS regeneration: transcript for English, existing ASS for localized clips."""
    clip_start, _ = _clip_bounds(story)
    language = str(story.get("language") or "en").lower()
    if language != "en" and ass_path.is_file():
        words = _words_from_ass_file(ass_path, clip_start)
        if words:
            return words

    if transcript is None:
        raise FileNotFoundError(
            f"Transcript required to rebuild captions for {story.get('id')}; missing {ass_path}"
        )
    return _transcript_clip_words(story, transcript)


def _transcript_clip_words(story: dict[str, Any], transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """English transcript words inside the story window, preferring the clip speaker."""
    clip_start, clip_end = _clip_bounds(story)
    speaker = str(story.get("speaker") or "UNKNOWN")
    all_words = _words(transcript.get("segments") or [])
    clip_words = [
        word
        for word in all_words
        if float(word["end"]) > clip_start
        and float(word["start"]) < clip_end
        and (speaker == "UNKNOWN" or str(word.get("speaker") or "UNKNOWN") == speaker)
    ]
    if not clip_words:
        clip_words = [
            word
            for word in all_words
            if float(word["end"]) > clip_start and float(word["start"]) < clip_end
        ]
    return clip_words


def _english_counterpart(story: dict[str, Any], document: dict[str, Any]) -> dict[str, Any] | None:
    clip_id = story.get("clip_id")
    for other in document.get("stories") or []:
        if other.get("language") == "en" and clip_id and other.get("clip_id") == clip_id:
            return other
    return None


def _saved_metadata_has_english(story: dict[str, Any], language: str) -> bool:
    seo = story.get("youtube_seo") or {}
    texts = [
        str(story.get("hook") or ""),
        str(seo.get("yt_title") or ""),
        str(seo.get("yt_description_summary") or ""),
        str(seo.get("yt_pinned_comment") or ""),
        *map(str, seo.get("yt_tags") or []),
        *(
            str(tag)
            for tag in seo.get("yt_hashtags") or []
            if str(tag).lower() not in _PLATFORM_HASHTAGS
        ),
    ]
    return any(find_untranslated_english(text, language) for text in texts)


def _english_hook_for_story(
    story: dict[str, Any],
    english: dict[str, Any] | None,
    transcript: dict[str, Any] | None,
) -> str:
    if english and english.get("hook"):
        return str(english["hook"])
    if transcript is not None:
        opening: list[str] = []
        for sentence in _sentences(_transcript_clip_words(story, transcript)):
            opening.append(str(sentence["text"]))
            if len(" ".join(opening).split()) >= 6:
                break
        if opening:
            return " ".join(opening)
    return str(story.get("hook") or "")


def _relocalize_saved_story(
    story: dict[str, Any],
    document: dict[str, Any],
    transcript: dict[str, Any] | None,
    ass_path: Path,
    *,
    video_id: str,
) -> None:
    """Bring a saved es/ru story to full localization before it is re-rendered.

    Captions come from the existing ASS unless it still contains English, in which
    case they are retranslated from the transcript. Hook, title, description,
    pinned comment, tags, and hashtags are retranslated whenever the saved record
    predates ``LOCALIZATION_VERSION``.
    """
    language = normalize_outro_language(str(story.get("language") or "en"))
    clip_start, _ = _clip_bounds(story)
    ass_words = _words_from_ass_file(ass_path, clip_start) if ass_path.is_file() else []
    caption_text = " ".join(str(word["word"]) for word in ass_words)
    leftover = find_untranslated_english(caption_text, language) if ass_words else ["<missing>"]
    metadata_current = story.get(
        "localization_version"
    ) == LOCALIZATION_VERSION and not _saved_metadata_has_english(story, language)
    if not leftover and metadata_current:
        story["words"] = ass_words
        return

    english = _english_counterpart(story, document)
    source_seo = YoutubeSeo.model_validate(
        (english or story).get("youtube_seo") or _resolve_story_seo(story).model_dump()
    )
    hook = _english_hook_for_story(story, english, transcript)
    if leftover:
        if transcript is None:
            raise FileNotFoundError(
                f"Transcript required to retranslate captions for {story.get('id')}"
            )
        logger.info(
            "Retranslating captions for {} (leftover English: {})",
            story.get("id"),
            leftover,
        )
        localized, seo = _localize_story(
            {**story, "words": _transcript_clip_words(story, transcript), "hook": hook},
            source_seo,
            language,
        )
        story["words"] = localized["words"]
        story["narration"] = localized["narration"]
        localized_hook = localized["hook"]
    else:
        logger.info("Relocalizing upload metadata for {}", story.get("id"))
        story["words"] = ass_words
        _, seo, localized_hook = _translate_story_copy(source_seo, hook, language)

    story["hook"] = localized_hook
    story["title"] = seo.yt_title
    story["localization_version"] = LOCALIZATION_VERSION
    output_dir = Path(story["output_dir"]) if story.get("output_dir") else ass_path.parent
    story.update(
        _write_youtube_seo_exports(
            output_dir,
            file_stem=str(story["id"]),
            video_id=video_id,
            story=story,
            seo=seo,
        )
    )


def _write_language_descriptions(document: dict[str, Any], language: str) -> None:
    """Rewrite ``<language>/description.txt`` from the saved story records."""
    blocks: list[str] = []
    language_dir: Path | None = None
    for story in document.get("stories") or []:
        if normalize_outro_language(str(story.get("language") or "en")) != language:
            continue
        if not story.get("youtube_seo") or not story.get("output_dir"):
            continue
        language_dir = Path(story["output_dir"])
        full_video_url = story.get("full_video_url") or _full_video_timestamp_url(
            str(document.get("video_id") or ""),
            float(story["start"]),
        )
        text = _format_youtube_description_text(
            YoutubeSeo.model_validate(story["youtube_seo"]),
            full_video_url,
            language,
        )
        blocks.append(f"===== {story['id']} =====\n{text.rstrip()}")
    if language_dir is not None and blocks:
        (language_dir / "description.txt").write_text(
            "\n\n".join(blocks) + "\n",
            encoding="utf-8",
        )


def publish_render_outputs(
    document: dict[str, Any],
    publish_dir: Path | None = None,
    *,
    languages: tuple[str, ...] | None = None,
) -> list[Path]:
    """Copy rendered shorts into ``out/<language>/`` for quick review and upload."""
    root = (publish_dir or (PROJECT_ROOT / "out")).resolve()
    selected_languages = set(languages or TARGET_LANGUAGES)
    copied: list[Path] = []
    for story in document.get("stories") or []:
        language = normalize_outro_language(str(story.get("language") or "en"))
        if language not in selected_languages:
            continue
        video_path = Path(str(story.get("video_path") or ""))
        if not video_path.is_file():
            continue
        target_dir = root / language
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / video_path.name
        shutil.copy2(video_path, target)
        copied.append(target)
    return copied


def _full_video_timestamp_url(video_id: str, start_time: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s"


def _fallback_youtube_seo(clip: SelectedClip, story: dict[str, Any]) -> YoutubeSeo:
    settings = get_settings()
    brand = settings.BRAND_NAME.strip() or "Channel"
    website = settings.BRAND_WEBSITE_URL.strip()
    title = (clip.title or story.get("title") or _title_from_hook(story.get("hook", ""))).strip()
    if len(title) > 65:
        title = title[:62].rstrip() + "..."
    hook = str(story.get("hook") or clip.hook or title).strip()
    summary = hook if hook.endswith(".") else f"{hook}."
    return YoutubeSeo(
        yt_title=title or "Short clip",
        yt_description_summary=(
            f"{summary} Watch this clip from {brand}. "
            "See the full long-form video for more context."
        ),
        yt_tags=[
            "shorts",
            "video",
            "education",
            brand.lower(),
            "clip",
            "talk",
        ],
        yt_hashtags=["#Shorts", "#Video", "#Education"],
        yt_pinned_comment=(
            "What stood out to you in this clip? Share below — "
            f"and explore more at {website}"
        ),
    )


def _resolve_youtube_seo(clip: SelectedClip, story: dict[str, Any]) -> YoutubeSeo:
    if clip.youtube_seo is not None:
        return clip.youtube_seo
    return _fallback_youtube_seo(clip, story)


def _format_youtube_description_text(
    seo: YoutubeSeo,
    full_video_url: str,
    language: str = "en",
) -> str:
    copy = DESCRIPTION_COPY[normalize_outro_language(language)]
    website = get_settings().BRAND_WEBSITE_URL.strip()
    hashtags_line = " ".join(seo.yt_hashtags)
    tags_block = ", ".join(seo.yt_tags)
    return (
        f"{copy['title']}\n"
        f"{seo.yt_title}\n\n"
        f"{copy['description']}\n"
        f"{seo.yt_description_summary}\n\n"
        f"{copy['watch_full']} {full_video_url}\n"
        f"{copy['website']} {website}\n\n"
        f"{hashtags_line}\n\n"
        f"{copy['pinned_comment']}\n"
        f"{seo.yt_pinned_comment}\n\n"
        f"{copy['tags']}\n"
        f"{tags_block}\n"
    )


def _hashtag_from_translation(text: str) -> str:
    """Collapse a translated hashtag body into a single ``#CamelCase`` token."""
    parts = re.findall(r"\w+", text.lstrip("#"), flags=re.UNICODE)
    body = "".join(part[:1].upper() + part[1:] for part in parts).replace("_", "")
    return f"#{body}" if body else ""


def _resolve_story_seo(story: dict[str, Any]) -> YoutubeSeo:
    clip_ref = story.get("_clip")
    if isinstance(clip_ref, SelectedClip):
        return _resolve_youtube_seo(clip_ref, story)
    if story.get("youtube_seo"):
        return YoutubeSeo.model_validate(story["youtube_seo"])
    return _fallback_youtube_seo(
        SelectedClip(
            clip_id=str(story.get("clip_id") or story["id"]),
            title=story["title"],
            hook=story["hook"],
            start_time=float(story["start"]),
            end_time=float(story["end"]),
            raw_text=story["narration"],
            reasoning=str(story.get("reasoning") or ""),
            speaker=story["speaker"],
            context_adjustments_made=str(story.get("context_adjustments_made") or ""),
        ),
        story,
    )


def _write_youtube_seo_exports(
    clip_dir: Path,
    *,
    file_stem: str,
    video_id: str,
    story: dict[str, Any],
    seo: YoutubeSeo,
) -> dict[str, Any]:
    full_video_url = _full_video_timestamp_url(video_id, float(story["start"]))
    metadata = {
        "short_id": story["id"],
        "video_id": video_id,
        "start_time": story["start"],
        "end_time": story["end"],
        "full_video_url": full_video_url,
        "website_url": get_settings().BRAND_WEBSITE_URL.strip(),
        "youtube_seo": seo.model_dump(),
    }
    clip_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = clip_dir / f"{file_stem}_youtube_metadata.json"
    description_path = clip_dir / f"{file_stem}_description.txt"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    description_path.write_text(
        _format_youtube_description_text(
            seo,
            full_video_url,
            str(story.get("language") or "en"),
        ),
        encoding="utf-8",
    )
    return {
        "full_video_url": full_video_url,
        "youtube_metadata_path": str(metadata_path.resolve()),
        "description_path": str(description_path.resolve()),
        "youtube_seo": seo.model_dump(),
    }


def _short_output_dir(shorts_dir: Path, file_stem: str) -> Path:
    return shorts_dir / file_stem


def _short_mp4_path(shorts_dir: Path, file_stem: str) -> Path:
    return _short_output_dir(shorts_dir, file_stem) / f"{file_stem}.mp4"


def _short_ass_path(shorts_dir: Path, file_stem: str) -> Path:
    return _short_output_dir(shorts_dir, file_stem) / f"{file_stem}.ass"


def _video_id_for_transcript(transcript: dict[str, Any]) -> str:
    stem = Path(transcript["audio_clean_path"]).stem
    if stem.endswith("_cleaned"):
        stem = stem[: -len("_cleaned")]
    if stem.endswith("_raw"):
        stem = stem[: -len("_raw")]
    return stem


def _shorts_output_dir(transcript: dict[str, Any], strategy_name: str | None = None) -> Path:
    settings = get_settings()
    base = settings.SHORTS_OUTPUT_DIR / _video_id_for_transcript(transcript)
    if strategy_name:
        return base / strategy_name
    return base


def _split_words_on_speech_gap(
    words: list[dict[str, Any]],
    gap_seconds: float,
) -> list[list[dict[str, Any]]]:
    """Split caption words wherever speech is silent longer than ``gap_seconds``."""
    if not words:
        return []
    groups: list[list[dict[str, Any]]] = [[words[0]]]
    for word in words[1:]:
        gap = float(word["start"]) - float(groups[-1][-1]["end"])
        if gap > gap_seconds:
            groups.append([word])
        else:
            groups[-1].append(word)
    return groups


def _split_words_on_silence(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return _split_words_on_speech_gap(words, _SILENCE_HIDE_SECONDS)


def _subtitle_phrases(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Build readable phrase-level cues with natural punctuation and pause breaks."""
    phrases: list[list[dict[str, Any]]] = []
    phrase: list[dict[str, Any]] = []
    for word in words:
        token = str(word["word"]).strip()
        candidate = [*phrase, word]
        candidate_text = " ".join(str(item["word"]).strip() for item in candidate)
        candidate_duration = float(candidate[-1]["end"]) - float(candidate[0]["start"])
        pause = float(word["start"]) - float(phrase[-1]["end"]) if phrase else 0.0
        should_break = bool(
            phrase
            and (
                word.get("_caption_timing_recalculated", False)
                or pause > _SPEECH_GAP_CLEAR_SECONDS
                or pause > _SILENCE_HIDE_SECONDS
                or len(candidate_text) > _SUBTITLE_MAX_CHARS
                or len(candidate) > _SUBTITLE_MAX_WORDS
                or candidate_duration > _SUBTITLE_MAX_SECONDS
            )
        )
        if should_break:
            phrases.append(phrase)
            phrase = [word]
        else:
            phrase = candidate

        if _SENTENCE_END.search(token) or (token.endswith((",", ";", ":")) and len(phrase) >= 4):
            phrases.append(phrase)
            phrase = []
    if phrase:
        phrases.append(phrase)
    return phrases


def _distribute_tokens_over_span(
    tokens: list[str],
    *,
    start: float,
    end: float,
    speaker: str,
) -> list[dict[str, Any]]:
    """Place tokens inside one continuous speech span. Never past ``end``."""
    if not tokens:
        return []
    duration = max(end - start, 0.04)
    weights = [max(len(re.sub(r"\W+", "", token, flags=re.UNICODE)), 1) for token in tokens]
    total_weight = sum(weights)
    cursor = start
    translated: list[dict[str, Any]] = []
    for index, (token, weight) in enumerate(zip(tokens, weights)):
        token_end = end if index == len(tokens) - 1 else cursor + duration * weight / total_weight
        token_end = min(end, max(token_end, cursor + 0.04))
        translated.append(
            {
                "word": token,
                "start": round(cursor, 3),
                "end": round(token_end, 3),
                "speaker": speaker,
                "score": 1.0,
            }
        )
        cursor = token_end
    return translated


def _allocate_token_counts(total: int, weights: list[int]) -> list[int]:
    """Split ``total`` tokens across runs in proportion to source letter weight."""
    if total <= 0 or not weights:
        return [0 for _ in weights]
    weight_sum = sum(weights) or 1
    raw = [total * weight / weight_sum for weight in weights]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (raw[index] - counts[index], weights[index]),
        reverse=True,
    )
    for index in order:
        if remainder <= 0:
            break
        counts[index] += 1
        remainder -= 1
    return counts


def _translated_words_for_phrase(
    text: str,
    *,
    start: float,
    end: float,
    speaker: str,
    source_words: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Time translated tokens on speech only.

    A hole longer than 1.2s (a Sanskrit shloka between English lines) is left
    empty. Tokens that belong after the hole are not stretched backward into it.
    """
    tokens = re.findall(r"\S+", text.strip())
    if not tokens:
        return []
    if source_words:
        prepared = _sanitize_caption_words(source_words, 0.0)
        runs = _split_words_on_speech_gap(prepared, _SPEECH_GAP_CLEAR_SECONDS)
        if len(runs) > 1:
            weights = []
            for run in runs:
                letters = sum(
                    max(len(re.sub(r"\W+", "", str(word["word"]), flags=re.UNICODE)), 1)
                    for word in run
                )
                weights.append(letters)
            placed: list[dict[str, Any]] = []
            cursor = 0
            for run, count in zip(runs, _allocate_token_counts(len(tokens), weights)):
                if count <= 0:
                    continue
                placed.extend(
                    _distribute_tokens_over_span(
                        tokens[cursor : cursor + count],
                        start=float(run[0]["start"]),
                        end=float(run[-1]["end"]),
                        speaker=speaker,
                    )
                )
                cursor += count
            return placed
        if prepared:
            start = float(prepared[0]["start"])
            end = float(prepared[-1]["end"])
    return _distribute_tokens_over_span(tokens, start=start, end=end, speaker=speaker)


def _localize_story(
    story: dict[str, Any],
    seo: YoutubeSeo,
    language: str,
) -> tuple[dict[str, Any], YoutubeSeo]:
    """Translate one story's subtitle phrases and upload copy in one API request."""
    localized = deepcopy(story)
    localized["language"] = language
    if language == "en":
        return localized, seo

    clip_start, clip_end = _clip_bounds(story)
    source_words = _sanitize_caption_words(
        _words_in_clip(story, clip_start, clip_end),
        clip_start,
    )
    phrases = _subtitle_phrases(source_words)
    phrase_texts = [
        " ".join(str(word["word"]).strip() for word in phrase).strip()
        for phrase in phrases
    ]
    translated_phrases, localized_seo, localized_hook = _translate_story_copy(
        seo,
        str(story.get("hook") or ""),
        language,
        caption_lines=phrase_texts,
    )

    localized_words: list[dict[str, Any]] = []
    for phrase, text in zip(phrases, translated_phrases):
        localized_words.extend(
            _translated_words_for_phrase(
                text,
                start=float(phrase[0]["start"]),
                end=float(phrase[-1]["end"]),
                speaker=str(phrase[0].get("speaker") or story.get("speaker") or "UNKNOWN"),
                source_words=phrase,
            )
        )
    localized["words"] = localized_words
    localized["narration"] = " ".join(translated_phrases)
    localized["hook"] = localized_hook
    localized["title"] = localized_seo.yt_title
    localized["youtube_seo"] = localized_seo.model_dump()
    localized["localization_version"] = LOCALIZATION_VERSION
    return localized, localized_seo


def _translate_story_copy(
    seo: YoutubeSeo,
    hook: str,
    language: str,
    *,
    caption_lines: list[str] | tuple[str, ...] = (),
) -> tuple[list[str], YoutubeSeo, str]:
    """Translate captions, hook, and every upload field in one request.

    Returns ``(translated_captions, localized_seo, localized_hook)``.
    """
    platform_hashtags = [tag for tag in seo.yt_hashtags if tag.lower() in _PLATFORM_HASHTAGS]
    content_hashtags = [tag for tag in seo.yt_hashtags if tag.lower() not in _PLATFORM_HASHTAGS]
    entries: list[tuple[str, str]] = [
        *((line, "subtitle") for line in caption_lines),
        (hook or seo.yt_title, "hook"),
        (seo.yt_title, "title"),
        (seo.yt_description_summary, "summary"),
        (seo.yt_pinned_comment, "pinned_comment"),
        *((tag, "tag") for tag in seo.yt_tags),
        *((tag.lstrip("#"), "hashtag") for tag in content_hashtags),
    ]
    translated = translate_subtitle_lines(
        [text for text, _ in entries],
        language,
        kinds=[kind for _, kind in entries],
    )
    cursor = len(caption_lines)
    captions = translated[:cursor]
    hook_text, title, summary, pinned = translated[cursor : cursor + 4]
    cursor += 4
    tags = translated[cursor : cursor + len(seo.yt_tags)]
    cursor += len(seo.yt_tags)
    hashtags = [_hashtag_from_translation(text) for text in translated[cursor:]]
    localized_seo = YoutubeSeo.model_validate(
        {
            **seo.model_dump(),
            "yt_title": title,
            "yt_description_summary": summary,
            "yt_pinned_comment": pinned,
            "yt_tags": list(dict.fromkeys(tag for tag in tags if tag)),
            "yt_hashtags": list(
                dict.fromkeys([*platform_hashtags, *(tag for tag in hashtags if tag)])
            ),
        }
    )
    return captions, localized_seo, hook_text


def _wrap_subtitle(text: str) -> str:
    """Balance a caption into no more than two lines."""
    if len(text) <= _SUBTITLE_LINE_CHARS:
        return text
    words = text.split()
    candidates: list[tuple[int, str, str]] = []
    for index in range(1, len(words)):
        first = " ".join(words[:index])
        second = " ".join(words[index:])
        if len(first) <= _SUBTITLE_LINE_CHARS and len(second) <= _SUBTITLE_LINE_CHARS:
            candidates.append((abs(len(first) - len(second)), first, second))
    if not candidates:
        return text
    _, first, second = min(candidates, key=lambda item: item[0])
    return f"{first}\n{second}"


def _phrase_wrap_break(phrase_words: list[dict[str, Any]]) -> int | None:
    """Return the word index where a two-line caption should break, if any."""
    wrapped = _wrap_subtitle(
        " ".join(str(item["word"]).strip() for item in phrase_words)
    )
    if "\n" not in wrapped:
        return None
    first_line, _ = wrapped.split("\n", 1)
    break_at = len(first_line.split())
    if 0 < break_at < len(phrase_words):
        return break_at
    return None


def _ass_escape_caption_text(token: str) -> str:
    return token.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def _ass_ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000.0)))


def _ass_karaoke_phrase(
    phrase_words: list[dict[str, Any]],
    *,
    event_start: float,
    clip_start: float,
) -> str:
    """Word-timed karaoke. Highlight stays inside each word's speech interval."""
    break_at = _phrase_wrap_break(phrase_words)
    parts: list[str] = []
    for index, word in enumerate(phrase_words):
        token = _ass_escape_caption_text(str(word["word"]).strip())
        rel_start = max(0.0, float(word["start"]) - clip_start)
        rel_end = max(rel_start, float(word["end"]) - clip_start)
        if index + 1 < len(phrase_words):
            next_start = max(0.0, float(phrase_words[index + 1]["start"]) - clip_start)
            rel_end = min(rel_end, next_start)
        highlight_on = _ass_ms(max(0.0, rel_start - event_start))
        highlight_off = max(highlight_on + 40, _ass_ms(max(rel_start, rel_end) - event_start))
        # Pin both fill colours to white so default ASS karaoke (Secondary → Primary)
        # cannot keep every syllable yellow-then-white. Drive the accent with \t.
        parts.append(
            rf"{{\k{max(1, (highlight_off - highlight_on) // 10)}"
            rf"\1c{ASS_CAPTION_PRIMARY_COLOUR}&\2c{ASS_CAPTION_PRIMARY_COLOUR}&"
            rf"\3c{ASS_CAPTION_OUTLINE_COLOUR}&\bord{ASS_CAPTION_OUTLINE}"
            rf"\t({highlight_on},{highlight_on},\1c{ASS_CAPTION_SECONDARY_COLOUR}&)"
            rf"\t({highlight_off},{highlight_off},\1c{ASS_CAPTION_PRIMARY_COLOUR}&)}}"
            f"{token}"
        )
        if break_at is not None and index + 1 == break_at:
            parts.append(r"\N")
        elif index + 1 < len(phrase_words):
            parts.append(" ")
    return "".join(parts)


def _clip_bounds(story: dict[str, Any]) -> tuple[float, float]:
    """Clip window shared by FFmpeg trim and subtitle re-basing (t=0 at clip_start)."""
    clip_start = max(float(story["start"]), 0.0)
    clip_end = float(story["end"])
    if clip_end <= clip_start:
        clip_end = clip_start + 0.5
    return clip_start, clip_end


def _words_in_clip(story: dict[str, Any], clip_start: float, clip_end: float) -> list[dict[str, Any]]:
    included: list[dict[str, Any]] = []
    for word in story.get("words") or []:
        word_start = float(word["start"])
        word_end = float(word["end"])
        if word_end <= clip_start or word_start >= clip_end:
            continue
        included.append(word)
    included.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return included


def _word_alignment_unreliable(word: dict[str, Any], start: float, end: float) -> bool:
    """True when WhisperX likely assigned segment-level timing instead of a real word span."""
    if "end" not in word:
        return True
    if end <= start + 0.01:
        return True
    score = word.get("score")
    if score is None:
        return (end - start) > _WORD_STRETCH_MAX_SECONDS
    return float(score) <= 0.0 and (end - start) > 0.5


def _sanitize_caption_word(word: dict[str, Any], *, clip_start: float) -> dict[str, Any] | None:
    """Fix stretched WhisperX timings and drop or tighten ghost alignments."""
    token = str(word.get("word") or "").strip()
    if not token or "start" not in word:
        return None

    start = float(word["start"])
    end = float(word.get("end", start))
    if end < start:
        end = start

    ghost_zero_start = start == 0.0 and clip_start >= _GHOST_ZERO_START_CLIP_OFFSET
    unreliable = _word_alignment_unreliable(word, start, end)
    stretched = (end - start) > _WORD_STRETCH_MAX_SECONDS

    if unreliable and end <= start + 0.01:
        return None

    if stretched:
        # WhisperX can stretch the next spoken word backwards across a shloka
        # or silent section. Anchor that word to its actual end instead.
        start = max(start, end - _WORD_TIGHT_WINDOW_SECONDS)
    elif ghost_zero_start or unreliable:
        start = max(start, end - _WORD_TIGHT_WINDOW_SECONDS)

    if end <= start:
        start = max(0.0, end - 0.1)

    sanitized = dict(word)
    sanitized["start"] = start
    sanitized["end"] = end
    if stretched:
        sanitized["_caption_timing_recalculated"] = True
    return sanitized


def _sanitize_caption_words(words: list[dict[str, Any]], clip_start: float) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for word in words:
        item = _sanitize_caption_word(word, clip_start=clip_start)
        if item is not None:
            cleaned.append(item)
    cleaned.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return cleaned


def _relative_word_times(
    word_start: float,
    word_end: float,
    clip_start: float,
) -> tuple[float, float]:
    rel_start = max(0.0, word_start - clip_start)
    rel_end = max(rel_start + 0.1, word_end - clip_start)
    return rel_start, rel_end


def _ass_subtitle_events(story: dict[str, Any], clip_start: float, clip_end: float) -> list[str]:
    """Render complete phrases so captions support speech instead of competing with it."""
    clip_words = _sanitize_caption_words(
        _words_in_clip(story, clip_start, clip_end),
        clip_start,
    )
    timed_groups: list[tuple[float, str]] = []
    speech_runs: list[list[dict[str, Any]]] = []
    for phrase_words in _subtitle_phrases(clip_words):
        speech_runs.extend(_split_words_on_speech_gap(phrase_words, _SPEECH_GAP_CLEAR_SECONDS))
    for index, phrase_words in enumerate(speech_runs):
        phrase_abs_end = float(phrase_words[-1]["end"])
        rel_phrase_end = max(0.0, phrase_abs_end - clip_start)
        start, end = _relative_word_times(
            float(phrase_words[0]["start"]),
            phrase_abs_end,
            clip_start,
        )
        end = max(end + 0.08, start + 0.35)
        end = min(end, start + _SUBTITLE_EVENT_MAX_SECONDS)
        if index + 1 < len(speech_runs):
            next_abs_start = float(speech_runs[index + 1][0]["start"])
            gap_to_next = next_abs_start - phrase_abs_end
            next_start, _ = _relative_word_times(
                next_abs_start,
                float(speech_runs[index + 1][0]["end"]),
                clip_start,
            )
            if gap_to_next > _SPEECH_GAP_CLEAR_SECONDS:
                # Shloka / long recitation: clear the line. Do not hold this
                # cue, and do not let the next translated word start early.
                end = min(max(rel_phrase_end, start + 0.04), next_start)
            elif gap_to_next <= _SUBTITLE_GAP_CLOSE_SECONDS:
                end = min(end, next_start)
            else:
                end = min(end, rel_phrase_end)
        else:
            end = min(end, rel_phrase_end + 0.08)
        phrase = _ass_karaoke_phrase(
            phrase_words,
            event_start=start,
            clip_start=clip_start,
        )
        timed_groups.append((start, _ass_dialogue(start, end, phrase)))
    timed_groups.sort(key=lambda item: item[0])
    return [event for _, event in timed_groups]


def _ass_caption_style_line() -> str:
    """Return the unified ASS ``Style:`` row (identical for every layout preset)."""
    return (
        f"Style: Caption,{ASS_CAPTION_FONT_NAME},{ASS_CAPTION_FONT_SIZE},"
        f"{ASS_CAPTION_PRIMARY_COLOUR},{ASS_CAPTION_SECONDARY_COLOUR},"
        f"{ASS_CAPTION_OUTLINE_COLOUR},{ASS_CAPTION_BACK_COLOUR},"
        # Bold, Italic, Underline, StrikeOut — heavy face, bold on.
        f"{ASS_CAPTION_BOLD},0,0,0,100,100,0.4,0,{ASS_CAPTION_BORDER_STYLE},"
        f"{ASS_CAPTION_OUTLINE},{ASS_CAPTION_SHADOW},2,"
        f"{ASS_CAPTION_MARGIN_L},{ASS_CAPTION_MARGIN_R},{ASS_CAPTION_MARGIN_V},1"
    )


def _write_ass(story: dict[str, Any], path: Path) -> None:
    clip_start, clip_end = _clip_bounds(story)
    events = _ass_subtitle_events(story, clip_start, clip_end)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{_ass_caption_style_line()}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def _ass_dialogue(start: float, end: float, text: str) -> str:
    motion = (
        rf"{{\fad({_CAPTION_FADE_IN_MS},{_CAPTION_FADE_OUT_MS})"
        rf"\move(540,{_CAPTION_BASELINE_Y + 8},540,{_CAPTION_BASELINE_Y})}}"
    )
    return (
        f"Dialogue: 0,{_ass_timestamp(start)},{_ass_timestamp(end)},Caption,,"
        f"{ASS_CAPTION_MARGIN_L},{ASS_CAPTION_MARGIN_R},{ASS_CAPTION_MARGIN_V},,{motion}{text}"
    )


def _render_filter_complex(
    ass_filter: str,
    fonts_filter: str,
    preset: str,
    trim_start: float,
    trim_end: float,
    *,
    outro_enabled: bool,
    outro_has_audio: bool,
    transition_seconds: float,
    outro_duration_seconds: float,
) -> str:
    duration = trim_end - trim_start
    video_trim = f"trim=start={trim_start:.6f}:end={trim_end:.6f},setpts=PTS-STARTPTS"
    audio_trim = f"atrim=start={trim_start:.6f}:end={trim_end:.6f},asetpts=PTS-STARTPTS"
    ass = f"ass=filename='{ass_filter}':fontsdir='{fonts_filter}'"
    if preset == "full_crop":
        video_tail = (
            f"scale=-1:1920,crop=1080:1920:(iw-1080)/2:0,"
            f"setsar=1,fps=30,{ass}[lecture]"
        )
    else:
        video_tail = (
            "split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:2[bgb];"
            "[fg]scale=1080:-2:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1,fps=30,{ass}[lecture]"
        )
    graph = f"[0:v]{video_trim},{video_tail};[0:a]{audio_trim}[lecture_a]"
    if not outro_enabled:
        return f"{graph};[lecture]null[v];[lecture_a]anull[aout]"

    transition = min(transition_seconds, max(duration / 2.0, 0.25))
    transition_start = max(duration - transition, 0.0)
    outro_end = f"{outro_duration_seconds:.6f}"
    graph += (
        f";[lecture]fade=t=out:st={transition_start:.3f}:d={transition:.3f}:"
        "color=0x120B2D,format=yuv420p[lecture_fade]"
        f";[1:v]trim=end={outro_end},setpts=PTS-STARTPTS,"
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,setsar=1,fps=30,format=yuv420p[outro_v]"
        f";[lecture_fade][outro_v]xfade=transition=fade:duration={transition:.3f}:"
        f"offset={transition_start:.3f}[v]"
    )
    if outro_has_audio:
        graph += (
            ";[lecture_a]aresample=48000,aformat=channel_layouts=stereo[lecture_a48]"
            f";[1:a]atrim=end={outro_end},asetpts=PTS-STARTPTS,"
            "aresample=48000,aformat=channel_layouts=stereo[outro_a]"
            f";[lecture_a48][outro_a]acrossfade=d={transition:.3f}:c1=tri:c2=tri[aout]"
        )
    else:
        graph += (
            f";[lecture_a]afade=t=out:st={transition_start:.3f}:d={transition:.3f},"
            "apad[aout]"
        )
    return graph


def _media_has_audio(ffprobe: str | None, path: Path) -> bool:
    if not ffprobe:
        return False
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and bool(probe.stdout.strip())


def _render_short(
    ffmpeg: str,
    source_video: Path,
    story: dict[str, Any],
    destination: Path,
    ass_path: Path,
    *,
    layout_preset: str,
    outro_path: Path | None,
    outro_transition_seconds: float,
    outro_duration_seconds: float,
) -> None:
    clip_start, clip_end = _clip_bounds(story)
    duration = clip_end - clip_start
    preroll = min(2.0, clip_start)
    fast_seek = max(0.0, clip_start - preroll)
    trim_start = clip_start - fast_seek
    trim_end = trim_start + duration
    ass_filter = _escape_ass_path(ass_path)
    fonts_filter = _escape_ass_path(_FONT_DIR)
    outro_enabled = bool(outro_path and outro_path.is_file())
    ffprobe = shutil.which("ffprobe")
    outro_has_audio = bool(
        outro_enabled and outro_path and _media_has_audio(ffprobe, outro_path)
    )
    filter_complex = _render_filter_complex(
        ass_filter,
        fonts_filter,
        layout_preset,
        trim_start,
        trim_end,
        outro_enabled=outro_enabled,
        outro_has_audio=outro_has_audio,
        transition_seconds=outro_transition_seconds,
        outro_duration_seconds=outro_duration_seconds,
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{fast_seek:.3f}",
        "-i",
        str(source_video),
    ]
    if outro_enabled and outro_path:
        command.extend(["-i", str(outro_path)])
    command.extend(
        [
        "-avoid_negative_ts",
        "make_zero",
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(destination),
        ]
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not destination.is_file():
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed to render {destination.name}: {detail}")

def _stories_from_clips(
    clips: list[SelectedClip],
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map DeepSeek clip records into the render pipeline story shape."""
    all_words = _words(segments)
    stories: list[dict[str, Any]] = []
    for index, clip in enumerate(clips, start=1):
        start = float(clip.start_time)
        end = float(clip.end_time)
        speaker = str(clip.speaker or "UNKNOWN")
        clip_words = [
            word
            for word in all_words
            if str(word.get("speaker") or "UNKNOWN") == speaker
            and float(word["end"]) > start
            and float(word["start"]) < end
        ]
        if not clip_words:
            clip_words = [
                word for word in all_words if float(word["end"]) > start and float(word["start"]) < end
            ]
        narration = clip.raw_text.strip() or " ".join(str(word["word"]).strip() for word in clip_words).strip()
        story_id = f"short_{index:02d}"
        stories.append(
            {
                "id": story_id,
                "clip_id": clip.clip_id,
                "title": clip.title.strip() or _title_from_hook(clip.hook),
                "hook": clip.hook.strip() or narration.split(".")[0][:120],
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker,
                "narration": narration,
                "reasoning": clip.reasoning,
                "context_adjustments_made": clip.context_adjustments_made,
                "youtube_seo": clip.youtube_seo.model_dump() if clip.youtube_seo else None,
                "words": clip_words,
                "_clip": clip,
            }
        )
    return stories


def source_video_for_audio(cleaned_or_raw_wav: str) -> Path:
    """Find ``{id}_source.mp4`` from a raw or cleaned wav path."""
    settings = get_settings()
    stem = Path(cleaned_or_raw_wav).stem
    if stem.endswith("_cleaned"):
        stem = stem[: -len("_cleaned")]
    if stem.endswith("_raw"):
        stem = stem[: -len("_raw")]
    video = settings.VIDEO_OUTPUT_DIR / f"{stem}_source.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"Source video not found: {video}")
    return video


def build_shorts(
    transcript: dict[str, Any],
    *,
    source_video: Path | None = None,
    clips: list[SelectedClip] | None = None,
    strategy_name: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Render Golden Top clips in every language and distribute the remainder."""
    settings = get_settings()
    settings.ensure_output_dirs()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH")
    video = source_video or source_video_for_audio(str(transcript["audio_clean_path"]))
    outro_by_language = {
        language: ensure_outro_video(language=language) for language in TARGET_LANGUAGES
    }
    outro_duration_by_language = {
        language: outro_clip_duration(path) for language, path in outro_by_language.items()
    }
    if clips is None:
        selection = select_clips_from_transcript(
            transcript,
            limit=12,
            min_seconds=30.0,
            max_seconds=60.0,
            model=settings.DEEPSEEK_MODEL,
        )
        clip_list = selection.clips
    else:
        clip_list = clips
    allocations = allocate_clips_by_language(clip_list)
    ranked_clips = sorted(clip_list, key=lambda clip: int(clip.rank or 10_000))
    stories = _stories_from_clips(ranked_clips, transcript.get("segments") or [])
    stories_by_clip = {id(story["_clip"]): story for story in stories}
    ranks_by_clip = {id(clip): rank for rank, clip in enumerate(ranked_clips, start=1)}
    shorts_dir = output_dir or _shorts_output_dir(transcript, strategy_name)
    shorts_dir.mkdir(parents=True, exist_ok=True)
    video_id = _video_id_for_transcript(transcript)
    rendered: list[dict[str, Any]] = []
    for language, language_clips in allocations.items():
        language_dir = shorts_dir / language
        language_dir.mkdir(parents=True, exist_ok=True)
        used_file_stems: set[str] = set()
        language_descriptions: list[str] = []
        for clip in language_clips:
            rank = ranks_by_clip[id(clip)]
            source_story = stories_by_clip[id(clip)]
            source_seo = _resolve_story_seo(source_story)
            story, seo = _localize_story(source_story, source_seo, language)
            slug = slugify_title(seo.yt_title)
            file_stem = f"short_{rank:02d}_{language}_{slug}"
            if file_stem in used_file_stems:
                file_stem = f"{file_stem}_{rank:02d}"
            used_file_stems.add(file_stem)
            story["id"] = file_stem

            ass_path = language_dir / f"{file_stem}.ass"
            mp4_path = language_dir / f"{file_stem}.mp4"
            _write_ass(story, ass_path)
            logger.info(
                "Rendering {} ({:.1f}s) {}",
                file_stem,
                story["end"] - story["start"],
                story["title"],
            )
            outro_path = outro_by_language[language]
            _render_short(
                ffmpeg,
                video,
                story,
                mp4_path,
                ass_path,
                layout_preset=settings.VIDEO_LAYOUT_PRESET,
                outro_path=outro_path,
                outro_transition_seconds=settings.BRAND_OUTRO_TRANSITION_SECONDS,
                outro_duration_seconds=outro_duration_by_language[language],
            )
            story.pop("_clip", None)
            seo_exports = _write_youtube_seo_exports(
                language_dir,
                file_stem=file_stem,
                video_id=video_id,
                story=story,
                seo=seo,
            )
            language_descriptions.append(
                f"===== {file_stem} =====\n"
                f"{_format_youtube_description_text(seo, seo_exports['full_video_url'], language).rstrip()}"
            )
            rendered.append(
                {
                    "id": story["id"],
                    "clip_id": story.get("clip_id", story["id"]),
                    "rank": rank,
                    "language": language,
                    "title": story["title"],
                    "hook": story["hook"],
                    "start": story["start"],
                    "end": story["end"],
                    "speaker": story["speaker"],
                    "narration": story["narration"],
                    "reasoning": story.get("reasoning", ""),
                    "context_adjustments_made": story.get("context_adjustments_made", ""),
                    "full_video_url": seo_exports["full_video_url"],
                    "youtube_seo": seo_exports["youtube_seo"],
                    "youtube_metadata_path": seo_exports["youtube_metadata_path"],
                    "description_path": seo_exports["description_path"],
                    "output_dir": str(language_dir.resolve()),
                    "video_path": str(mp4_path.resolve()),
                    "localization_version": LOCALIZATION_VERSION,
                }
            )
        (language_dir / "description.txt").write_text(
            "\n\n".join(language_descriptions) + "\n",
            encoding="utf-8",
        )
    expected_render_count = len(clip_list) + 6
    if len(rendered) != expected_render_count:
        raise RuntimeError(
            f"Expected {expected_render_count} rendered shorts, got {len(rendered)}"
        )
    document = {
        "source_url": transcript["source_url"],
        "source_video": str(video.resolve()),
        "video_id": _video_id_for_transcript(transcript),
        "output_dir": str(shorts_dir.resolve()),
        "layout_preset": settings.VIDEO_LAYOUT_PRESET,
        "subtitle_style": SUBTITLE_STYLE_ID,
        "language_matrix": {
            language: [ranks_by_clip[id(clip)] for clip in language_clips]
            for language, language_clips in allocations.items()
        },
        "outro_paths": {
            language: str(path.resolve()) for language, path in outro_by_language.items()
        },
        "outro_path": str(outro_by_language["en"].resolve()),
        "stories": rendered,
    }
    if strategy_name:
        document["strategy"] = strategy_name
    stories_path = shorts_dir / "stories.json"
    stories_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote {} stories to {}", len(rendered), stories_path)
    return document


def restyle_existing_shorts(
    video_id: str,
    *,
    publish_dir: Path | None = None,
    languages: tuple[str, ...] = TARGET_LANGUAGES,
) -> dict[str, Any]:
    """Re-render saved Shorts with the current brand style without reselecting clips."""
    settings = get_settings()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH")

    transcript_path = settings.JSON_OUTPUT_DIR / f"{video_id}.json"
    stories_path = settings.SHORTS_OUTPUT_DIR / video_id / "stories.json"
    if not stories_path.is_file():
        raise FileNotFoundError(f"Stories file not found: {stories_path}")

    transcript: dict[str, Any] | None = None
    if transcript_path.is_file():
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))

    document = json.loads(stories_path.read_text(encoding="utf-8"))
    source_video = Path(document["source_video"])
    if not source_video.is_file():
        if transcript is None:
            raise FileNotFoundError(f"Source video not found: {source_video}")
        source_video = source_video_for_audio(str(transcript["audio_clean_path"]))

    outro_cache: dict[str, Path] = {}
    outro_duration_cache: dict[str, float] = {}
    shorts_root = stories_path.parent
    for story in document.get("stories") or []:
        language = normalize_outro_language(str(story.get("language") or "en"))
        if language not in languages:
            continue
        if language not in outro_cache:
            outro_cache[language] = ensure_outro_video(language=language)
            outro_duration_cache[language] = outro_clip_duration(outro_cache[language])
        ass_path, destination, temporary = _story_media_paths(story, shorts_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if language == "en":
            story["words"] = _caption_words_for_restyle(story, transcript, ass_path)
        else:
            _relocalize_saved_story(
                story,
                document,
                transcript,
                ass_path,
                video_id=str(document.get("video_id") or video_id),
            )
        _write_ass(story, ass_path)
        _render_short(
            ffmpeg,
            source_video,
            story,
            temporary,
            ass_path,
            layout_preset=str(document.get("layout_preset") or settings.VIDEO_LAYOUT_PRESET),
            outro_path=outro_cache[language],
            outro_transition_seconds=settings.BRAND_OUTRO_TRANSITION_SECONDS,
            outro_duration_seconds=outro_duration_cache[language],
        )
        temporary.replace(destination)
        story["video_path"] = str(destination.resolve())
        story.pop("words", None)
        # Persist per story so paid translations survive a later failure.
        stories_path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    for language in outro_cache:
        if language != "en":
            _write_language_descriptions(document, language)
    document["subtitle_style"] = SUBTITLE_STYLE_ID
    document["outro_paths"] = {
        **(document.get("outro_paths") or {}),
        **{
            language: str(path.resolve())
            for language, path in outro_cache.items()
        },
    }
    if "en" in outro_cache:
        document["outro_path"] = str(outro_cache["en"].resolve())
    stories_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if publish_dir is not None:
        publish_render_outputs(document, publish_dir, languages=languages)
    return document
