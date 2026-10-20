"""Semantic clip selection via DeepSeek API with context-integrity rules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from config.settings import CLIP_STRATEGIES, get_settings
from utils.logger import logger

MIN_CLIP_SECONDS = 30.0
MAX_CLIP_SECONDS = 60.0
DEFAULT_CLIP_COUNT = 12
MIN_FALLBACK_CLIP_COUNT = 6
MIN_CANDIDATE_COUNT = 28
MAX_CANDIDATE_COUNT = 30
OVERLAP_THRESHOLD = 0.25
END_PAUSE_SECONDS = 0.2
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")
_NON_SLUG_CHARS = re.compile(r"[^\w]+", re.UNICODE)


def slugify_title(title: str, max_words: int = 6) -> str:
    """Turn a YouTube title into a short snake_case slug (about ``max_words`` tokens)."""
    tokens: list[str] = []
    for raw in str(title or "").lower().split():
        token = _NON_SLUG_CHARS.sub("", raw)
        if token:
            tokens.append(token)
    if not tokens:
        return "clip_short"
    return "_".join(tokens[:max_words])

_PROMPT_PREAMBLE = """You are a YouTube Shorts editor for long-form spoken video (lectures, talks, podcasts, interviews).

Preserve the speaker's meaning, tone, and any quoted material that is central to a clip. Match clip titles to the ideas in the transcript rather than inventing a niche theme.

Return a ranked candidate pool of 28–30 clips of 30–60 seconds each. Rank the
strongest clip first. Candidates must cover distinct ideas and should not overlap.
Distribute candidate clips evenly across the early, middle, and late sections of
the transcript (e.g., 0–25m, 25–50m, 50m+) to prevent timestamp clustering."""

_PROMPT_CRITERIA_AND_SCHEMA = """
Within each selected clip, apply these five criteria:

1. Hook — The opening line must grab attention (question, bold claim, tension, or vivid setup).
2. Conflict — There is stakes, disagreement, paradox, or a problem the speaker resolves or explores.
3. Standalone — A viewer with zero prior context understands the clip without the rest of the video.
4. Novelty — Insight, story, or teaching that feels fresh rather than generic filler.
5. Context Integrity — Critical for pronouns and references:
   - If the clip opens with pronouns or dangling references (he, she, they, it, this, that, these, those, here, there), expand the start backward until the referent is named (roughly 100–200 spoken words when needed).
   - Expand the end forward until the thought finishes at a natural sentence boundary; never cut mid-sentence.
   - After expansion, the clip MUST still be 30–60 seconds. If longer, trim from the end at a sentence boundary; if shorter, extend forward to complete the thought or reject the clip.

For EVERY clip, also return a complete ``youtube_seo`` object tuned for YouTube Shorts upload (search keywords for the topic; mention __BRAND_NAME__ and __BRAND_WEBSITE__ when natural).

Return JSON only, matching this schema:
{
  "clips": [
    {
      "clip_id": "clip_01",
      "rank": 1,
      "title": "short title",
      "hook": "first sentence or hook line",
      "hook_score": 9.2,
      "viral_score": 8.8,
      "start_time": 123.45,
      "end_time": 178.90,
      "raw_text": "full transcript text for the window",
      "reasoning": "brief note on hook/conflict/standalone/novelty/context",
      "speaker": "SPEAKER_00",
      "context_adjustments_made": "describe backward/forward expansions or none",
      "youtube_seo": {
        "yt_title": "High-CTR viral title, target 40–60 characters (hard max 65), include 1-2 relevant emojis",
        "yt_description_summary": "Exactly 2 concise sentences rich in SEO keywords summarizing the clip.",
        "yt_tags": ["tag1", "tag2", "... 8-12 relevant SEO tags total"],
        "yt_hashtags": ["#Shorts", "#tag2", "#tag3", "#tag4", "4-5 hashtags total, first must be #Shorts"],
        "yt_pinned_comment": "Engaging pinned comment draft: invite discussion, mention watching the full long-form video, and __BRAND_WEBSITE__ when appropriate."
      }
    }
  ]
}

Use segment timestamps as bounds. Prefer the main speaker's voice when multiple speakers appear."""


def _prompt_criteria_and_schema() -> str:
    settings = get_settings()
    brand_name = settings.BRAND_NAME.strip() or "your channel"
    brand_website = settings.BRAND_WEBSITE_URL.strip() or "https://example.com"
    return (
        _PROMPT_CRITERIA_AND_SCHEMA.replace("__BRAND_NAME__", brand_name).replace(
            "__BRAND_WEBSITE__", brand_website
        )
    )


def _strategy_prompt_body(strategy_suffix: str) -> str:
    settings = get_settings()
    extra = (getattr(settings, "CLIP_SELECTOR_CONTEXT", "") or "").strip()
    preamble = _PROMPT_PREAMBLE
    if extra:
        preamble = preamble.rstrip() + "\n\n" + extra + "\n"
    return preamble + strategy_suffix + _prompt_criteria_and_schema()


_STRATEGY_1_SUFFIX = """

Strategy: balanced category mix. Choose clips with this distribution:

- Category A (Humor & Playful Wit, 1–2 clips): Moments where the speaker jokes, uses playful analogies, ironizes, or makes the audience laugh.
- Category B (Paradox & Mind-Bending Hooks, 2 clips): Provocative statements, counter-intuitive claims, or strong intellectual hooks.
- Category C (Core ideas & teaching, 2–3 clips): Foundational explanations, memorable quotes, and high-value teaching moments.

Do not select only serious academic segments. Humor, jokes, and witty stories are top priority for engagement and retention."""


_STRATEGY_2_SUFFIX = """

Strategy: retention science and viral hooks. Prioritize clips engineered for scroll-stopping retention:

- Open with a 3-second curiosity gap: the first line must create immediate tension, mystery, or "wait, what?" energy.
- Favor intense emotional or intellectual hooks: awe, surprise, righteous anger, laughter, or a mind-flip revelation.
- End on a punchline, payoff, or mic-drop line when possible—closure that rewards the watch.
- Prefer bold claims, vivid stories, and quotable one-liners over slow setup or dry exposition.
- Still keep each clip substantively grounded; do not sacrifice meaning for clickbait."""


_STRATEGY_3_SUFFIX = """

Strategy: myth-busting and constructive controversy. Prioritize clips where the speaker:

- Refutes common misconceptions, popular myths, or lazy assumptions about the topic.
- Engages in sharp but respectful debate: pushback, counter-arguments, and clear rebuttals.
- Uses bold analogies or striking comparisons that challenge conventional thinking.
- Names what "people wrongly believe" before dismantling it—make the tension explicit.
- Avoid pure insult or outrage bait; the controversy must serve clarity and teaching."""


def strategy_system_prompt(strategy_id: str) -> str:
    suffixes = {
        "1_categorized": _STRATEGY_1_SUFFIX,
        "2_viral_hooks": _STRATEGY_2_SUFFIX,
        "3_controversy": _STRATEGY_3_SUFFIX,
    }
    return _strategy_prompt_body(suffixes[strategy_id])


def strategy_prompt_for(strategy_id: str) -> str:
    return strategy_system_prompt(strategy_id)


SYSTEM_PROMPT = ""  # Use strategy_prompt_for() or strategy_system_prompt() at runtime.


class YoutubeSeo(BaseModel):
    yt_title: str = Field(..., max_length=65)
    yt_description_summary: str
    yt_tags: list[str] = Field(..., min_length=1)
    yt_hashtags: list[str] = Field(..., min_length=1)
    yt_pinned_comment: str

    @field_validator("yt_title", mode="before")
    @classmethod
    def truncate_title(cls, v: str) -> str:
        if isinstance(v, str) and len(v) > 65:
            return v[:62].strip() + "..."
        return v

    @field_validator("yt_tags")
    @classmethod
    def _normalize_tags(cls, value: list[str]) -> list[str]:
        tags = [str(item).strip() for item in value if str(item).strip()]
        if not tags:
            raise ValueError("yt_tags must contain at least one tag")
        return tags

    @field_validator("yt_hashtags")
    @classmethod
    def _normalize_hashtags(cls, value: list[str]) -> list[str]:
        hashtags = [str(item).strip() for item in value if str(item).strip()]
        if not hashtags:
            raise ValueError("yt_hashtags must contain at least one hashtag")
        if not hashtags[0].startswith("#"):
            hashtags[0] = f"#{hashtags[0].lstrip('#')}"
        if hashtags[0].lower() != "#shorts":
            hashtags.insert(0, "#Shorts")
        return hashtags


class SelectedClip(BaseModel):
    clip_id: str
    rank: int | None = Field(default=None, ge=1)
    title: str
    hook: str
    hook_score: float = Field(default=0.0, ge=0.0, le=10.0)
    viral_score: float = Field(default=0.0, ge=0.0, le=10.0)
    start_time: float
    end_time: float
    raw_text: str
    reasoning: str
    speaker: str
    context_adjustments_made: str = ""
    youtube_seo: YoutubeSeo | None = None

    @field_validator("start_time", "end_time")
    @classmethod
    def _round_times(cls, value: float) -> float:
        return round(float(value), 3)


class ClipSelectionResult(BaseModel):
    clips: list[SelectedClip] = Field(default_factory=list)


def _clip_quality(clip: SelectedClip) -> tuple[float, float, float]:
    """Return a deterministic quality key, with model rank as the final tie-break."""
    rank_score = -float(clip.rank or 10_000)
    return (float(clip.hook_score) + float(clip.viral_score), float(clip.hook_score), rank_score)


def _overlap_ratio(left: SelectedClip, right: SelectedClip) -> float:
    """Measure shared time against the shorter clip."""
    overlap = max(
        0.0,
        min(float(left.end_time), float(right.end_time))
        - max(float(left.start_time), float(right.start_time)),
    )
    shorter = min(
        _duration(left.start_time, left.end_time),
        _duration(right.start_time, right.end_time),
    )
    return overlap / shorter if shorter > 0 else 0.0


def deduplicate_ranked_clips(
    clips: list[SelectedClip],
    *,
    limit: int = DEFAULT_CLIP_COUNT,
    overlap_threshold: float = OVERLAP_THRESHOLD,
) -> list[SelectedClip]:
    """Keep the highest-scoring unique windows and return a clean 1-based ranking."""
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be between 0 and 1")

    ranked = sorted(clips, key=_clip_quality, reverse=True)
    kept: list[SelectedClip] = []
    seen_content: set[tuple[str, str]] = set()
    for clip in ranked:
        content_key = (
            re.sub(r"\W+", "", clip.title.casefold()),
            re.sub(r"\W+", "", clip.raw_text.casefold()),
        )
        if content_key in seen_content:
            continue
        if any(_overlap_ratio(clip, other) > overlap_threshold for other in kept):
            continue
        # The 25% rule removes duplicate concepts; the final contract is stricter:
        # no rendered windows may share source time at all.
        if any(_overlap_ratio(clip, other) > 0.0 for other in kept):
            continue
        kept.append(clip)
        seen_content.add(content_key)
        if len(kept) >= limit:
            break

    return [
        clip.model_copy(update={"rank": index, "clip_id": f"clip_{index:02d}"})
        for index, clip in enumerate(kept[:limit], start=1)
    ]


def _load_transcript(transcript_path_or_dict: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(transcript_path_or_dict, dict):
        return transcript_path_or_dict
    path = Path(transcript_path_or_dict)
    if not path.is_file():
        raise FileNotFoundError(f"Transcript not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _segments_text(segments: list[dict[str, Any]], *, max_chars: int = 450_000) -> str:
    lines: list[str] = []
    total = 0
    for segment in segments:
        start = segment.get("start", 0.0)
        end = segment.get("end", start)
        speaker = segment.get("speaker") or "UNKNOWN"
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        line = f"[{float(start):.2f}-{float(end):.2f}] {speaker}: {text}"
        if total + len(line) + 1 > max_chars:
            lines.append("... [transcript truncated for model context] ...")
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _all_words(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in segments:
        for word in segment.get("words") or []:
            if "start" in word and "end" in word and word.get("word"):
                words.append(word)
    words.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return words


def _words_for_speaker(words: list[dict[str, Any]], speaker: str) -> list[dict[str, Any]]:
    return [word for word in words if str(word.get("speaker") or "UNKNOWN") == speaker]


def _duration(start: float, end: float) -> float:
    return max(float(end) - float(start), 0.0)


def _trim_to_max_duration(
    ordered_words: list[dict[str, Any]],
    start: float,
    end: float,
    max_seconds: float,
) -> tuple[float, float, str]:
    if _duration(start, end) <= max_seconds:
        return start, end, ""
    in_window = [word for word in ordered_words if float(word["end"]) > start and float(word["start"]) < end]
    if not in_window:
        return start, min(start + max_seconds, end), "trimmed end to max duration"
    kept: list[dict[str, Any]] = []
    for word in in_window:
        candidate_end = float(word["end"]) + END_PAUSE_SECONDS
        if candidate_end - start > max_seconds and kept:
            break
        kept.append(word)
    if not kept:
        kept = [in_window[0]]
    new_end = float(kept[-1]["end"]) + END_PAUSE_SECONDS
    note = "trimmed end to fit 60s at word boundary"
    for index in range(len(kept) - 1, -1, -1):
        if _SENTENCE_END.search(str(kept[index]["word"]).strip()):
            new_end = float(kept[index]["end"]) + END_PAUSE_SECONDS
            note = "trimmed end to sentence boundary within 60s"
            break
    return start, new_end, note


def _expand_to_min_duration(
    all_words: list[dict[str, Any]],
    start: float,
    end: float,
    min_seconds: float,
) -> tuple[float, float, str]:
    if _duration(start, end) >= min_seconds:
        return start, end, ""
    start_index = next(
        (index for index, word in enumerate(all_words) if float(word["end"]) > start),
        0,
    )
    end_index = next(
        (index for index in range(len(all_words) - 1, -1, -1) if float(all_words[index]["start"]) < end),
        len(all_words) - 1,
    )
    new_start = start
    new_end = end
    while _duration(new_start, new_end) < min_seconds and (start_index > 0 or end_index < len(all_words) - 1):
        grew = False
        if end_index < len(all_words) - 1:
            end_index += 1
            new_end = float(all_words[end_index]["end"]) + END_PAUSE_SECONDS
            grew = True
        if _duration(new_start, new_end) >= min_seconds:
            break
        if start_index > 0:
            start_index -= 1
            new_start = float(all_words[start_index]["start"])
            grew = True
        if not grew:
            break
    note = ""
    if _duration(new_start, new_end) >= min_seconds and (new_start != start or new_end != end):
        note = "expanded window to reach minimum 30s"
    return new_start, new_end, note


def refine_clip_timestamps(
    clip: SelectedClip | dict[str, Any],
    segments: list[dict[str, Any]],
    *,
    min_seconds: float = MIN_CLIP_SECONDS,
    max_seconds: float = MAX_CLIP_SECONDS,
) -> dict[str, Any]:
    """Snap clip bounds to word timestamps and enforce duration with a short end pause."""
    if isinstance(clip, SelectedClip):
        payload = clip.model_dump()
    else:
        payload = dict(clip)

    start = float(payload["start_time"])
    end = float(payload["end_time"])
    speaker = str(payload.get("speaker") or "UNKNOWN")
    adjustments: list[str] = []
    existing = str(payload.get("context_adjustments_made") or "").strip()
    if existing:
        adjustments.append(existing)

    all_words = _all_words(segments)
    speaker_words = _words_for_speaker(all_words, speaker) or all_words
    in_range = [
        word
        for word in speaker_words
        if float(word["end"]) > start - 0.05 and float(word["start"]) < end + 0.05
    ]
    if in_range:
        start = float(in_range[0]["start"])
        end = float(in_range[-1]["end"]) + END_PAUSE_SECONDS
        adjustments.append("snapped to word boundaries")

    trim_note = ""
    if _duration(start, end) > max_seconds:
        start, end, trim_note = _trim_to_max_duration(speaker_words or all_words, start, end, max_seconds)
        if trim_note:
            adjustments.append(trim_note)

    expand_note = ""
    if _duration(start, end) < min_seconds:
        start, end, expand_note = _expand_to_min_duration(all_words, start, end, min_seconds)
        if expand_note:
            adjustments.append(expand_note)

    if end <= start:
        end = start + min(min_seconds, max_seconds)

    payload["start_time"] = round(start, 3)
    payload["end_time"] = round(end, 3)
    payload["context_adjustments_made"] = "; ".join(item for item in adjustments if item)
    return payload


def _parse_selection(content: str) -> ClipSelectionResult:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as decode_error:
        data = _repair_or_recover_selection(text, decode_error)
    if isinstance(data, list):
        data = {"clips": data}
    return ClipSelectionResult.model_validate(data)


def _repair_or_recover_selection(
    text: str,
    decode_error: json.JSONDecodeError,
) -> dict[str, Any] | list[Any]:
    """Repair truncated model JSON, or retain its complete clip objects."""
    try:
        from json_repair import repair_json
    except ImportError:
        repair_json = None

    if repair_json is not None:
        try:
            repaired = repair_json(text, return_objects=True)
            if isinstance(repaired, (dict, list)):
                # Validate here so a repaired, incomplete final clip does not
                # prevent the complete preceding clips from being recovered.
                candidate = {"clips": repaired} if isinstance(repaired, list) else repaired
                ClipSelectionResult.model_validate(candidate)
                return repaired
        except Exception:
            logger.debug("json_repair could not produce a valid clip selection")

    clips = _recover_complete_clip_objects(text)
    if clips is None:
        raise decode_error
    return {"clips": clips}


def _recover_complete_clip_objects(text: str) -> list[dict[str, Any]] | None:
    """Decode complete objects from a cut-off top-level list or ``clips`` array."""
    stripped = text.lstrip()
    if stripped.startswith("["):
        array_start = len(text) - len(stripped)
    else:
        match = re.search(r'"clips"\s*:\s*\[', text)
        if match is None:
            return None
        array_start = match.end() - 1

    decoder = json.JSONDecoder()
    clips: list[dict[str, Any]] = []
    position = array_start + 1
    while position < len(text):
        while position < len(text) and (text[position].isspace() or text[position] == ","):
            position += 1
        if position >= len(text) or text[position] == "]":
            break
        try:
            item, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        if not isinstance(item, dict):
            return None
        clips.append(item)
    return clips


def _deepseek_client() -> OpenAI:
    settings = get_settings()
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is empty. Set DEEPSEEK_API_KEY in .env or the environment."
        )
    return OpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )


def select_clips_from_transcript(
    transcript_path_or_dict: str | Path | dict[str, Any],
    *,
    limit: int | None = None,
    min_seconds: float = MIN_CLIP_SECONDS,
    max_seconds: float = MAX_CLIP_SECONDS,
    model: str | None = None,
    system_prompt: str | None = None,
) -> ClipSelectionResult:
    """Call DeepSeek to pick semantic Short windows, then refine timestamps on words."""
    settings = get_settings()
    transcript = _load_transcript(transcript_path_or_dict)
    segments = transcript.get("segments") or []
    if not segments:
        logger.warning("Transcript has no segments; returning empty clip list")
        return ClipSelectionResult(clips=[])

    clip_limit = limit if limit is not None else DEFAULT_CLIP_COUNT
    candidate_limit = max(MAX_CANDIDATE_COUNT, clip_limit)
    transcript_body = _segments_text(segments)
    user_message = (
        f"Return {MIN_CANDIDATE_COUNT} to {candidate_limit} ranked candidates between "
        f"{min_seconds:.0f} and {max_seconds:.0f} seconds. Include numeric "
        f"hook_score and viral_score values from 0 to 10 for every candidate.\n\n"
        f"Transcript segments:\n{transcript_body}"
    )

    client = _deepseek_client()
    chosen_model = model or settings.DEEPSEEK_MODEL
    prompt = (
        system_prompt
        if system_prompt is not None
        else strategy_system_prompt("1_categorized")
    )
    logger.info("Requesting up to {} clip candidates from DeepSeek model {}", candidate_limit, chosen_model)
    response = client.chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
        max_tokens=8192,
    )
    content = response.choices[0].message.content or "{}"
    result = _parse_selection(content)

    refined: list[SelectedClip] = []
    for index, clip in enumerate(result.clips[:candidate_limit], start=1):
        if not clip.clip_id.strip():
            clip = clip.model_copy(update={"clip_id": f"clip_{index:02d}"})
        adjusted = refine_clip_timestamps(
            clip,
            segments,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
        )
        duration = _duration(adjusted["start_time"], adjusted["end_time"])
        if duration < min_seconds - 0.5 or duration > max_seconds + 0.5:
            logger.warning(
                "Skipping {} after refine ({:.1f}s outside {:.0f}-{:.0f}s)",
                adjusted.get("clip_id"),
                duration,
                min_seconds,
                max_seconds,
            )
            continue
        refined.append(SelectedClip.model_validate(adjusted))

    selected = deduplicate_ranked_clips(refined, limit=clip_limit)
    if len(selected) < MIN_FALLBACK_CLIP_COUNT:
        raise RuntimeError(
            f"DeepSeek produced only {len(selected)} unique non-overlapping clips; "
            f"at least {MIN_FALLBACK_CLIP_COUNT} are required. "
            "The source video may be too short or the candidate "
            "pool may need regeneration."
        )
    if len(selected) < clip_limit:
        logger.warning(
            "DeepSeek produced {} of {} requested unique clips; "
            "continuing with graceful language allocation",
            len(selected),
            clip_limit,
        )
    logger.info("Selected {} ranked, unique, non-overlapping clips", len(selected))
    return ClipSelectionResult(clips=selected)


def select_clips_for_all_strategies(
    transcript_path_or_dict: str | Path | dict[str, Any],
    *,
    limit: int | None = None,
    min_seconds: float = MIN_CLIP_SECONDS,
    max_seconds: float = MAX_CLIP_SECONDS,
    model: str | None = None,
    strategies: list[str] | None = None,
) -> dict[str, list[SelectedClip]]:
    """Run DeepSeek clip selection once per strategy folder name in CLIP_STRATEGIES."""
    strategy_ids = strategies if strategies is not None else CLIP_STRATEGIES
    results: dict[str, list[SelectedClip]] = {}
    for strategy_id in strategy_ids:
        if strategy_id not in CLIP_STRATEGIES:
            raise ValueError(
                f"Unknown clip strategy '{strategy_id}'. "
                f"Expected one of: {', '.join(CLIP_STRATEGIES)}"
            )
        prompt = strategy_prompt_for(strategy_id)
        logger.info("Clip selection strategy: {}", strategy_id)
        selection = select_clips_from_transcript(
            transcript_path_or_dict,
            limit=limit,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            model=model,
            system_prompt=prompt,
        )
        results[strategy_id] = selection.clips
    return results
