from __future__ import annotations

import re
import unittest
from pathlib import Path

from config.settings import PROJECT_ROOT, Settings, get_settings
from modules.clip_selector import strategy_system_prompt
from modules.outro import download_outro_music, get_outro_language_copy

_REPO_ROOT = PROJECT_ROOT.parent
_OUTRO_HTML = (
    _REPO_ROOT
    / "poster"
    / "outro_card_template"
    / "code.html"
)
_BANNED_TEXT_PATTERNS = (
    r"jiva",
    r"jivainstitute",
    r"satyanarayana",
    r"gnppigfquyc",
    r"hlDq6MZR4ws",
    r"A_Babaji",
    r"youtubeShorts",
)
_BANNED_PROMPT_SUBSTRINGS = (
    "Vedic Wisdom",
    "shloka",
    "Bhagavad Gita",
    "Chaitanya Mahaprabhu",
    "Support on Patreon",
)


class PublicationDefaultTests(unittest.TestCase):
    def setUp(self) -> None:
        get_settings.cache_clear()

    def test_default_outro_music_url_is_empty(self) -> None:
        self.assertEqual(Settings().OUTRO_MUSIC_URL.strip(), "")

    def test_download_outro_music_requires_url(self) -> None:
        with self.assertRaises(RuntimeError):
            download_outro_music(url="")

    def test_default_outro_html_has_no_banned_identity_strings(self) -> None:
        self.assertTrue(_OUTRO_HTML.is_file(), msg=str(_OUTRO_HTML))
        html = _OUTRO_HTML.read_text(encoding="utf-8").lower()
        for pattern in _BANNED_TEXT_PATTERNS:
            self.assertIsNone(re.search(pattern, html, re.IGNORECASE), pattern)

    def test_clip_selector_prompts_are_generic_by_default(self) -> None:
        prompt = strategy_system_prompt("1_categorized")
        lowered = prompt.lower()
        for snippet in _BANNED_PROMPT_SUBSTRINGS:
            self.assertNotIn(snippet.lower(), lowered, snippet)

    def test_outro_copy_uses_secondary_cta_not_patreon_default(self) -> None:
        copy = get_outro_language_copy()["en"]
        self.assertIn("secondary_cta", copy)
        self.assertNotIn("patreon", copy["secondary_cta"].lower())

    def test_no_client_png_assets_in_poster_tree(self) -> None:
        poster = _REPO_ROOT / "poster"
        images = list(poster.rglob("*.png")) + list(poster.rglob("*.jpg"))
        self.assertEqual(images, [])


if __name__ == "__main__":
    unittest.main()
