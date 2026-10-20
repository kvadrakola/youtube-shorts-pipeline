from __future__ import annotations

import unittest

from config.settings import get_settings
from modules.outro import (
    get_outro_language_copy,
    normalize_outro_language,
    outro_video_path,
)
from modules.translator import find_untranslated_english


class OutroLocalizationTests(unittest.TestCase):
    def test_normalize_outro_language(self) -> None:
        self.assertEqual(normalize_outro_language("ES"), "es")
        self.assertEqual(normalize_outro_language("ru"), "ru")
        self.assertEqual(normalize_outro_language("de"), "en")

    def test_copy_templates_match_pipeline_languages(self) -> None:
        copy_table = get_outro_language_copy()
        self.assertEqual(set(copy_table), {"en", "es", "ru"})
        host = (
            get_settings()
            .BRAND_WEBSITE_URL.strip()
            .replace("https://", "")
            .replace("http://", "")
            .rstrip("/")
        )
        ctas = {
            language: (copy["lecture_cta"], copy["secondary_cta"])
            for language, copy in copy_table.items()
        }
        self.assertEqual(
            ctas,
            {
                "en": (f"Full video & subscribe at {host}", "Link in description"),
                "es": (f"Video completo y suscripción en {host}", "Enlace en la descripción"),
                "ru": (f"Полное видео и подписка на {host}", "Ссылка в описании"),
            },
        )

    def test_every_language_covers_every_visible_outro_string(self) -> None:
        copy_table = get_outro_language_copy()
        keys = set(copy_table["en"])
        self.assertTrue(
            {"scholar_badge", "headline", "tagline", "website_label", "social_handle", "footer_line"}
            <= keys
        )
        for language, copy in copy_table.items():
            self.assertEqual(set(copy), keys, language)

    def test_localized_outro_copy_has_no_english(self) -> None:
        copy_table = get_outro_language_copy()
        for language in ("es", "ru"):
            for key, text in copy_table[language].items():
                if key in {"website_label", "social_handle", "footer_line", "scholar_badge"}:
                    continue
                self.assertEqual(
                    find_untranslated_english(text, language),
                    [],
                    f"{language}.{key}: {text}",
                )
                self.assertNotEqual(text, copy_table["en"][key], f"{language}.{key}")

    def test_outro_video_path_is_per_language(self) -> None:
        self.assertTrue(outro_video_path("en").name.endswith("_en.mp4"))
        self.assertTrue(outro_video_path("es").name.endswith("_es.mp4"))
        self.assertTrue(outro_video_path("ru").name.endswith("_ru.mp4"))


if __name__ == "__main__":
    unittest.main()
