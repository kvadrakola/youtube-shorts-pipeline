from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config.settings import (
    ASS_CAPTION_BACK_COLOUR,
    ASS_CAPTION_BOLD,
    ASS_CAPTION_FONT_NAME,
    ASS_CAPTION_FONT_SIZE,
    ASS_CAPTION_MARGIN_V,
    ASS_CAPTION_OUTLINE,
    ASS_CAPTION_OUTLINE_COLOUR,
    ASS_CAPTION_PRIMARY_COLOUR,
    ASS_CAPTION_SECONDARY_COLOUR,
    ASS_CAPTION_SHADOW,
)
from modules.shorts import (
    _ass_caption_style_line,
    _ass_subtitle_events,
    _render_filter_complex,
    _translated_words_for_phrase,
    _write_ass,
    _wrap_subtitle,
)


class BrandSubtitleStyleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.story = {
            "id": "short_01",
            "start": 10.0,
            "end": 15.0,
            "words": [
                {"word": "Wisdom", "start": 10.2, "end": 10.6},
                {"word": "begins", "start": 10.7, "end": 11.1},
                {"word": "when", "start": 11.2, "end": 11.5},
                {"word": "we", "start": 11.6, "end": 11.8},
                {"word": "understand", "start": 11.9, "end": 12.5},
                {"word": "ourselves.", "start": 12.6, "end": 13.3},
            ],
        }

    def test_caption_is_phrase_level_with_two_line_limit(self) -> None:
        events = _ass_subtitle_events(self.story, 10.0, 15.0)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertRegex(event, r"Wisdom.*begins.*when.*we\\N.*understand.*ourselves\.")
        self.assertEqual(event.count(r"\N"), 1)
        self.assertIn(r"\fad(120,150)", event)
        baseline_y = 1920 - ASS_CAPTION_MARGIN_V
        self.assertIn(rf"\move(540,{baseline_y + 8},540,{baseline_y})", event)
        self.assertNotIn(r"\bord0", event)
        self.assertNotIn(r"\blur", event)
        self.assertNotIn(r"\shad", event)

    def test_active_word_karaoke_uses_gold_accent(self) -> None:
        events = _ass_subtitle_events(self.story, 10.0, 15.0)
        event = events[0]
        self.assertIn(rf"\1c{ASS_CAPTION_PRIMARY_COLOUR}&", event)
        self.assertIn(rf"\2c{ASS_CAPTION_PRIMARY_COLOUR}&", event)
        self.assertIn(rf"\1c{ASS_CAPTION_SECONDARY_COLOUR}&", event)
        self.assertIn(rf"\3c{ASS_CAPTION_OUTLINE_COLOUR}&", event)
        self.assertIn(rf"\bord{ASS_CAPTION_OUTLINE}", event)
        self.assertGreaterEqual(event.count(r"\k"), 6)
        self.assertGreaterEqual(event.count(r"\t("), 12)

    def test_wrap_balances_long_caption(self) -> None:
        wrapped = _wrap_subtitle("The deepest knowledge begins with understanding yourself")
        lines = wrapped.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(len(line) <= 32 for line in lines))

    def test_ass_style_uses_unified_high_contrast_template(self) -> None:
        style = _ass_caption_style_line()
        self.assertEqual(ASS_CAPTION_FONT_NAME, "Montserrat ExtraBold")
        self.assertEqual(ASS_CAPTION_FONT_SIZE, 70)
        self.assertEqual(ASS_CAPTION_BOLD, 1)
        self.assertEqual(ASS_CAPTION_OUTLINE, 6.0)
        self.assertEqual(ASS_CAPTION_SHADOW, 2)
        self.assertEqual(ASS_CAPTION_MARGIN_V, 380)
        self.assertEqual(ASS_CAPTION_PRIMARY_COLOUR, "&H00FFFFFF")
        self.assertEqual(ASS_CAPTION_SECONDARY_COLOUR, "&H0000FFFF")
        self.assertEqual(ASS_CAPTION_OUTLINE_COLOUR, "&H00000000")
        self.assertEqual(ASS_CAPTION_BACK_COLOUR, "&H00000000")
        self.assertIn("Montserrat ExtraBold", style)
        self.assertNotIn("SemiBold", style)
        self.assertIn(",1,0,0,0,100,100,", style)
        self.assertIn(ASS_CAPTION_PRIMARY_COLOUR, style)
        self.assertIn(ASS_CAPTION_SECONDARY_COLOUR, style)
        self.assertIn(f",1,{ASS_CAPTION_OUTLINE},{ASS_CAPTION_SHADOW},2,", style)
        self.assertIn(f",140,140,{ASS_CAPTION_MARGIN_V},1", style)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "caption.ass"
            _write_ass(self.story, output)
            document = output.read_text(encoding="utf-8")
        self.assertIn(style, document)
        self.assertNotIn("Arial", document)
        self.assertNotIn("Cormorant", document)

    def test_both_layout_presets_burn_the_same_captions(self) -> None:
        for preset in ("blur_bg", "full_crop"):
            graph = _render_filter_complex(
                "captions.ass",
                "fonts",
                preset,
                0.0,
                20.0,
                outro_enabled=False,
                outro_has_audio=False,
                transition_seconds=0.5,
                outro_duration_seconds=8.0,
            )
            self.assertIn("ass=filename='captions.ass':fontsdir='fonts'", graph)

    def test_outro_graph_darkens_and_crossfades(self) -> None:
        graph = _render_filter_complex(
            "captions.ass",
            "fonts",
            "full_crop",
            0.0,
            20.0,
            outro_enabled=True,
            outro_has_audio=False,
            transition_seconds=1.25,
            outro_duration_seconds=8.0,
        )
        self.assertIn("color=0x120B2D", graph)
        self.assertIn("xfade=transition=fade:duration=1.250:offset=18.750", graph)
        self.assertIn("afade=t=out:st=18.750:d=1.250", graph)

    def test_stretched_whisper_word_appears_at_speech_not_gap(self) -> None:
        story = {
            "id": "short_stretch",
            "start": 10.0,
            "end": 35.0,
            "words": [
                {"word": "Earlier", "start": 10.2, "end": 10.8, "score": 0.9},
                {"word": "speech.", "start": 10.9, "end": 11.4, "score": 0.9},
                {"word": "After", "start": 31.4, "end": 31.9, "score": 0.9},
                {
                    "word": "shloka",
                    "start": 11.5,
                    "end": 32.0,
                    "score": 0.2,
                },
            ],
        }
        events = _ass_subtitle_events(story, 10.0, 35.0)
        self.assertGreaterEqual(len(events), 2)
        shloka_event = next(event for event in events if "shloka" in event)
        start_ts = shloka_event.split(",", 3)[1]
        self.assertEqual(start_ts, "0:00:21.65")

    def test_long_gap_hides_captions_between_phrases(self) -> None:
        story = {
            "id": "short_gap",
            "start": 0.0,
            "end": 20.0,
            "words": [
                {"word": "Line", "start": 1.0, "end": 1.4, "score": 0.9},
                {"word": "one.", "start": 1.5, "end": 2.0, "score": 0.9},
                {"word": "Line", "start": 12.0, "end": 12.4, "score": 0.9},
                {"word": "two.", "start": 12.5, "end": 13.0, "score": 0.9},
            ],
        }
        events = _ass_subtitle_events(story, 0.0, 20.0)
        self.assertEqual(len(events), 2)
        first_end = events[0].split(",", 3)[2]
        second_start = events[1].split(",", 3)[1]
        self.assertEqual(first_end, "0:00:02.00")
        self.assertGreaterEqual(second_start, "0:00:11.90")

    def test_sanskrit_gap_over_1_2s_hides_upcoming_word(self) -> None:
        story = {
            "id": "short_shloka_gap",
            "start": 0.0,
            "end": 20.0,
            "words": [
                {"word": "Listen.", "start": 1.0, "end": 1.5, "score": 0.9},
                {"word": "Afterwards", "start": 1.55, "end": 7.2, "score": 0.4},
            ],
        }
        events = _ass_subtitle_events(story, 0.0, 20.0)
        self.assertEqual(len(events), 2)
        self.assertNotIn("Afterwards", events[0])
        self.assertEqual(events[0].split(",", 3)[2], "0:00:01.50")
        self.assertEqual(events[1].split(",", 3)[1], "0:00:06.85")
        highlight = events[1].split(r"\t(", 1)[1]
        self.assertTrue(highlight.startswith("0,0,"))

    def test_translation_skips_gap_longer_than_1_2_seconds(self) -> None:
        source = [
            {"word": "Listen.", "start": 1.0, "end": 1.5, "score": 0.9, "speaker": "S"},
            {"word": "Afterwards", "start": 1.55, "end": 7.2, "score": 0.4, "speaker": "S"},
        ]
        translated = _translated_words_for_phrase(
            "Escucha Después",
            start=1.0,
            end=7.2,
            speaker="S",
            source_words=source,
        )
        self.assertEqual([word["word"] for word in translated], ["Escucha", "Después"])
        snapped_start = round(7.2 - 0.35, 3)
        for word in translated:
            overlaps_gap = word["start"] < snapped_start and word["end"] > 1.5
            self.assertFalse(overlaps_gap, word)
        self.assertLessEqual(translated[0]["end"], 1.5)
        self.assertGreaterEqual(translated[1]["start"], snapped_start)

    def test_gap_over_point_six_seconds_closes_current_line(self) -> None:
        story = {
            "id": "short_gap_threshold",
            "start": 0.0,
            "end": 5.0,
            "words": [
                {"word": "Before", "start": 1.0, "end": 1.4, "score": 0.9},
                {"word": "after.", "start": 2.1, "end": 2.5, "score": 0.9},
            ],
        }
        events = _ass_subtitle_events(story, 0.0, 5.0)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].split(",", 3)[2], "0:00:01.40")
        self.assertEqual(events[1].split(",", 3)[1], "0:00:02.10")

    def test_outro_graph_crossfades_outro_audio(self) -> None:
        graph = _render_filter_complex(
            "captions.ass",
            "fonts",
            "full_crop",
            2.0,
            32.0,
            outro_enabled=True,
            outro_has_audio=True,
            transition_seconds=0.5,
            outro_duration_seconds=8.0,
        )
        self.assertIn("crop=1080:1920:(iw-1080)/2:0", graph)
        self.assertLess(graph.index("ass=filename="), graph.index("xfade="))
        self.assertIn("xfade=transition=fade:duration=0.500:offset=29.500[v]", graph)
        self.assertIn("[lecture_a48][outro_a]acrossfade=d=0.500", graph)


if __name__ == "__main__":
    unittest.main()
