"""Context-aware subtitle translation through the DeepSeek-V3 API."""

from __future__ import annotations

import json
import re
from typing import Iterable, Sequence

from openai import OpenAI

from config.settings import get_settings
from utils.logger import logger

SUPPORTED_LANGUAGES = {"es": "Spanish", "ru": "Russian"}


def _translation_glossary() -> dict[str, dict[str, str]]:
    """Optional per-channel glossary from ``TRANSLATION_GLOSSARY_JSON``."""
    raw = get_settings().TRANSLATION_GLOSSARY_JSON.strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("TRANSLATION_GLOSSARY_JSON must be valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("TRANSLATION_GLOSSARY_JSON must be a JSON object")
    glossary: dict[str, dict[str, str]] = {}
    for term, forms in data.items():
        if not isinstance(forms, dict):
            continue
        entry = {
            code: str(forms[code]).strip()
            for code in SUPPORTED_LANGUAGES
            if code in forms and str(forms[code]).strip()
        }
        if entry:
            glossary[str(term).strip()] = entry
    return glossary

TEXT_KINDS = {
    "subtitle": "a spoken subtitle line; it may be a sentence fragment, translate it as a fragment",
    "hook": "the on-screen hook line that opens the Short",
    "title": "a YouTube Shorts title; keep any emoji, stay under 65 characters",
    "summary": "a YouTube description paragraph",
    "pinned_comment": "a pinned YouTube comment",
    "tag": "a short YouTube search tag (1-4 words, no '#')",
    "hashtag": "a hashtag body: ONE word in CamelCase, no spaces, no '#', letters/digits only",
}

# Common English words that never occur as standalone words in Spanish, Russian,
# or IAST-style Sanskrit transliteration. Any hit marks a line as untranslated.
_ENGLISH_MARKERS = frozenset(
    """
    the and you your you're yours is are was were this that these those what with for
    have it it's its they them their we our be been being will would can can't could
    should just like but because when where which who whom how why there here from
    about all not do does don't did if my his her she him into out up very really than
    then also only some any every know think people something nothing everything
    always never of to in on or us watch full lecture website subscribe support
    wisdom philosophy teaching soul anger self mind
    life love god truth peace happiness desire attachment devotion renunciation ethics
    psychology growth faith death fear purpose consciousness enlightenment mindfulness
    """.split()
)
def _latin_allowed_in_ru() -> frozenset[str]:
    settings = get_settings()
    allowed = {"shorts", "youtube", "instagram", "example"}
    extra = (settings.TRANSLATION_LATIN_ALLOWLIST or "").replace(",", " ").split()
    for token in extra:
        token = token.strip().lower()
        if token:
            allowed.add(token)
    brand = settings.BRAND_NAME.strip().lower()
    if brand:
        allowed.add(brand)
        for token in re.findall(r"[a-z0-9]+", brand):
            allowed.add(token)
    host = (
        settings.BRAND_WEBSITE_URL.strip()
        .lower()
        .replace("https://", "")
        .replace("http://", "")
        .split("/")[0]
    )
    if host:
        allowed.add(host)
    return frozenset(allowed)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)
_URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:org|com|net)\b", re.IGNORECASE)
_MAX_RETRY_ROUNDS = 2

_SYSTEM_PROMPT = """You are an expert subtitle and YouTube-copy translator for {content_domain}.
Translate every entry from English into {language}. Preserve the speaker's meaning, warmth,
humor, and conversational rhythm. Subtitles must be natural, concise, and easy to read quickly.
Never add explanations, omit ideas, merge entries, or change the number/order of entries.

ZERO ENGLISH RULE — the output must contain no English at all:
- Translate every entry completely, including short fragments, single words, entries that
  start or end mid-sentence, interjections ("So", "Yeah", "Okay"), and idioms.
- Never copy an English entry through unchanged and never leave English words mid-line.
- Some entries may already be partly in {language}; return them entirely in {language}.
- The ONLY text allowed to stay untranslated: proper nouns and quoted terms from the source,
  configured glossary entries below, brand name ({brand_name}), platform names (YouTube), and URLs.
{script_rule}
{glossary_block}
Each entry has a "kind":
{kinds}

Return JSON only, with exactly this shape: {{"translations": [{{"id": 0, "text": "..."}}]}}."""

_SCRIPT_RULES = {
    "ru": (
        "- Write body text in Cyrillic. Transliterate foreign proper nouns from the source into "
        "Cyrillic when they appear in the line, including inside tags and hashtags. Only URLs, "
        "the channel brand ({brand_name}), configured allowlisted Latin tokens, and YouTube may "
        "remain in Latin script."
    ),
    "es": (
        "- Keep recognizable proper nouns from the source in their usual form; everything else "
        "must be natural Spanish."
    ),
}


def _system_prompt(language_code: str) -> str:
    settings = get_settings()
    brand_name = settings.BRAND_NAME.strip() or "Example Channel"
    domain = (settings.TRANSLATION_CONTENT_DOMAIN or "long-form spoken video").strip()
    glossary_table = _translation_glossary()
    if glossary_table:
        glossary_lines = "\n".join(
            f'- {term}: {SUPPORTED_LANGUAGES[language_code]} "{forms[language_code]}"'
            for term, forms in glossary_table.items()
            if language_code in forms
        )
        glossary_block = "Glossary (always use exactly these forms):\n" + glossary_lines
    else:
        glossary_block = ""
    kinds = "\n".join(f'- "{kind}": {description}' for kind, description in TEXT_KINDS.items())
    script_rule = _SCRIPT_RULES[language_code].format(brand_name=brand_name)
    return _SYSTEM_PROMPT.format(
        content_domain=domain,
        language=SUPPORTED_LANGUAGES[language_code],
        brand_name=brand_name,
        script_rule=script_rule,
        glossary_block=glossary_block,
        kinds=kinds,
    )


def find_untranslated_english(text: str, target_language: str) -> list[str]:
    """Return English words left in ``text`` that should have been translated."""
    language_code = target_language.strip().lower()
    if language_code not in SUPPORTED_LANGUAGES:
        return []
    cleaned = _CAMEL_BOUNDARY.sub(" ", _URL.sub(" ", text))
    latin_words = [word for word in _WORD.findall(cleaned) if word.isascii()]
    if language_code == "ru":
        hits = [word for word in latin_words if word.lower() not in _latin_allowed_in_ru()]
    else:
        hits = [word for word in latin_words if word.lower() in _ENGLISH_MARKERS]
    return sorted(set(hits), key=hits.index)


def _client() -> OpenAI:
    settings = get_settings()
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is empty. Set it before translating subtitles."
        )
    return OpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )


def _request_translations(
    client: OpenAI,
    entries: list[dict[str, object]],
    language_code: str,
    model: str,
) -> dict[int, str]:
    payload = {
        "source_language": "English",
        "target_language": SUPPORTED_LANGUAGES[language_code],
        "lines": entries,
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _system_prompt(language_code)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    translated = data.get("translations")
    if not isinstance(translated, list):
        raise RuntimeError("DeepSeek translation response has no translations list")
    return {
        int(item["id"]): str(item["text"]).strip()
        for item in translated
        if isinstance(item, dict) and "id" in item and "text" in item
    }


def translate_subtitle_lines(
    lines: Iterable[str],
    target_language: str,
    *,
    kinds: Sequence[str] | None = None,
    model: str | None = None,
) -> list[str]:
    """Translate ordered lines while preserving one output per input.

    Lines that still contain English after translation are re-requested; if any
    survive ``_MAX_RETRY_ROUNDS`` retries a ``RuntimeError`` is raised.
    """
    source = [str(line).strip() for line in lines]
    language_code = target_language.strip().lower()
    if language_code == "en":
        return source
    if language_code not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported target language: {target_language}")
    if not source:
        return []
    line_kinds = list(kinds) if kinds is not None else ["subtitle"] * len(source)
    if len(line_kinds) != len(source):
        raise ValueError("kinds must have one entry per line")
    unknown = sorted(set(line_kinds) - set(TEXT_KINDS))
    if unknown:
        raise ValueError(f"Unknown text kind(s): {', '.join(unknown)}")

    settings = get_settings()
    client = _client()
    chosen_model = model or settings.DEEPSEEK_MODEL
    by_id = _request_translations(
        client,
        [
            {"id": index, "kind": kind, "text": text}
            for index, (text, kind) in enumerate(zip(source, line_kinds))
        ],
        language_code,
        chosen_model,
    )
    if set(by_id) != set(range(len(source))):
        raise RuntimeError(
            "DeepSeek translation response did not preserve all subtitle line IDs"
        )

    for attempt in range(1, _MAX_RETRY_ROUNDS + 1):
        leftovers = {
            index: find_untranslated_english(by_id[index], language_code)
            for index in range(len(source))
            if source[index]
        }
        leftovers = {index: words for index, words in leftovers.items() if words}
        if not leftovers:
            break
        logger.warning(
            "Retranslating {} {} line(s) with leftover English (attempt {}): {}",
            len(leftovers),
            language_code,
            attempt,
            {index: words for index, words in leftovers.items()},
        )
        retry = _request_translations(
            client,
            [
                {
                    "id": index,
                    "kind": line_kinds[index],
                    "text": source[index],
                    "rejected_translation": by_id[index],
                    "problem": (
                        "The rejected translation still contains English: "
                        + ", ".join(words)
                        + f". Translate it fully into {SUPPORTED_LANGUAGES[language_code]}."
                    ),
                }
                for index, words in leftovers.items()
            ],
            language_code,
            chosen_model,
        )
        for index, text in retry.items():
            if index in leftovers and text:
                by_id[index] = text
    else:
        remaining = {
            index: (source[index], by_id[index])
            for index in range(len(source))
            if source[index] and find_untranslated_english(by_id[index], language_code)
        }
        if remaining:
            raise RuntimeError(
                f"DeepSeek left English in {len(remaining)} {language_code} line(s): {remaining}"
            )
    return [by_id[index] for index in range(len(source))]
