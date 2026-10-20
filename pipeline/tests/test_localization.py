from __future__ import annotations

import unittest
from unittest.mock import patch

import modules.translator as translator
from modules.clip_selector import YoutubeSeo
from modules.shorts import (
    _format_youtube_description_text,
    _hashtag_from_translation,
    _saved_metadata_has_english,
    _translate_story_copy,
)
from modules.translator import find_untranslated_english, translate_subtitle_lines


def _seo() -> YoutubeSeo:
    return YoutubeSeo(
        yt_title="Already Fallen? You Can't Fall Further!",
        yt_description_summary="Discover the core idea from this talk.",
        yt_tags=["education", "video", "Example Channel"],
        yt_hashtags=["#Shorts", "#Education", "#Video"],
        yt_pinned_comment="What stood out to you? Visit example.com",
    )


class UntranslatedEnglishDetectionTests(unittest.TestCase):
    def test_flags_english_fragment_inside_spanish_line(self) -> None:
        self.assertEqual(
            find_untranslated_english("Él dice que the soul is eterna", "es"),
            ["the", "soul", "is"],
        )

    @patch.object(
        translator,
        "_translation_glossary",
        return_value={"Product": {"es": "Producto", "ru": "Продукт"}},
    )
    def test_allows_configured_glossary_terms_in_spanish(self, _glossary) -> None:
        line = "Product example.com"
        self.assertEqual(find_untranslated_english(line, "es"), [])

    def test_accented_spanish_words_are_not_split_into_english(self) -> None:
        self.assertEqual(find_untranslated_english("¿Ya caído? Él tenía razón, sí.", "es"), [])

    def test_flags_camel_case_english_hashtag(self) -> None:
        self.assertEqual(find_untranslated_english("TheAnswer", "es"), ["The"])

    def test_flags_mostly_latin_russian_line(self) -> None:
        self.assertTrue(find_untranslated_english("Somebody sleeping ground falls", "ru"))
        self.assertEqual(find_untranslated_english("Гнев и example.com", "ru"), [])

    def test_russian_requires_cyrillic_names_and_hashtags(self) -> None:
        self.assertEqual(find_untranslated_english("#Anger", "ru"), ["Anger"])
        self.assertEqual(find_untranslated_english("#SelfControl", "ru"), ["Self", "Control"])
        self.assertEqual(find_untranslated_english("ProductName", "ru"), ["Product", "Name"])
        self.assertEqual(find_untranslated_english("Example Channel", "ru"), [])
        self.assertEqual(find_untranslated_english("ProductName", "es"), [])

    def test_saved_metadata_with_english_hashtag_is_detected(self) -> None:
        story = {
            "hook": "Гнев — как пример",
            "youtube_seo": {
                "yt_title": "Гнев",
                "yt_description_summary": "Описание.",
                "yt_pinned_comment": "Комментарий",
                "yt_tags": ["гнев"],
                "yt_hashtags": ["#Shorts", "#Гнев"],
            },
        }
        self.assertFalse(_saved_metadata_has_english(story, "ru"))
        story["youtube_seo"]["yt_hashtags"].append("#Anger")
        self.assertTrue(_saved_metadata_has_english(story, "ru"))

    def test_english_target_is_never_flagged(self) -> None:
        self.assertEqual(find_untranslated_english("the soul", "en"), [])


class TranslateRetryTests(unittest.TestCase):
    @patch.object(translator, "_client", return_value=object())
    def test_retries_only_lines_with_leftover_english(self, _client) -> None:
        responses = [
            {0: "Hola amigos", 1: "Esto es the end"},
            {1: "Este es el final"},
        ]
        with patch.object(translator, "_request_translations", side_effect=responses) as request:
            result = translate_subtitle_lines(["Hello friends", "This is the end"], "es")

        self.assertEqual(result, ["Hola amigos", "Este es el final"])
        retry_entries = request.call_args_list[1].args[1]
        self.assertEqual([entry["id"] for entry in retry_entries], [1])
        self.assertIn("the", retry_entries[0]["problem"])

    @patch.object(translator, "_client", return_value=object())
    def test_raises_when_english_survives_all_retries(self, _client) -> None:
        responses = [{0: "the end"}, {0: "the end"}, {0: "the end"}]
        with patch.object(translator, "_request_translations", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "left English"):
                translate_subtitle_lines(["the end"], "es")

    def test_rejects_mismatched_kinds(self) -> None:
        with self.assertRaises(ValueError):
            translate_subtitle_lines(["a", "b"], "es", kinds=["subtitle"])


class StoryCopyLocalizationTests(unittest.TestCase):
    def test_translates_hook_title_summary_comment_tags_and_hashtags(self) -> None:
        def fake_translate(lines, language, *, kinds):
            self.assertEqual(language, "es")
            self.assertEqual(
                kinds,
                ["subtitle", "hook", "title", "summary", "pinned_comment",
                 "tag", "tag", "tag", "hashtag", "hashtag"],
            )
            self.assertNotIn("Shorts", lines)
            return [
                "Hola", "¿Ya caído?", "¿Ya caído? ¡No puedes caer más!",
                "Resumen breve del clip.", "¿Qué te llamó la atención? Visita example.com",
                "educación", "ideas", "Example Channel",
                "Educación", "Ideas",
            ]

        with patch("modules.shorts.translate_subtitle_lines", side_effect=fake_translate):
            captions, seo, hook = _translate_story_copy(
                _seo(), "Somebody is sleeping on the ground", "es", caption_lines=["Hello"]
            )

        self.assertEqual(captions, ["Hola"])
        self.assertEqual(hook, "¿Ya caído?")
        self.assertEqual(seo.yt_title, "¿Ya caído? ¡No puedes caer más!")
        self.assertEqual(seo.yt_tags, ["educación", "ideas", "Example Channel"])
        self.assertEqual(seo.yt_hashtags, ["#Shorts", "#Educación", "#Ideas"])

    def test_hashtag_normalization(self) -> None:
        self.assertEqual(_hashtag_from_translation("#ведическая мудрость"), "#ВедическаяМудрость")
        self.assertEqual(_hashtag_from_translation("  "), "")

    def test_localized_description_labels_have_no_english(self) -> None:
        fields = {
            "es": ("Título", "Resumen.", "etiqueta", "Comentario"),
            "ru": ("Название", "Описание.", "тег", "Комментарий"),
        }
        for language, (title, summary, tag, comment) in fields.items():
            seo = YoutubeSeo(
                yt_title=title,
                yt_description_summary=summary,
                yt_tags=[tag],
                yt_hashtags=["#Shorts"],
                yt_pinned_comment=comment,
            )
            text = _format_youtube_description_text(seo, "https://www.youtube.com/watch?v=x", language)
            body = text.replace("#Shorts", "")
            self.assertEqual(find_untranslated_english(body, language), [], language)
            self.assertNotIn("[TITLE]", text)


if __name__ == "__main__":
    unittest.main()
