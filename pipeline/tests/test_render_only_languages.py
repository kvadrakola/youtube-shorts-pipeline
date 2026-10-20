from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import run_render_only
from modules.shorts import publish_render_outputs


class RenderOnlyLanguageTests(unittest.TestCase):
    def test_default_languages_preserve_existing_behavior(self) -> None:
        args = run_render_only._parse_args(["video-id"])

        self.assertEqual(args.lang, ("en", "es", "ru"))

    def test_comma_separated_languages_are_normalized_and_deduplicated(self) -> None:
        args = run_render_only._parse_args(["video-id", "--lang", " RU,es,ru "])

        self.assertEqual(args.lang, ("ru", "es"))

    def test_powershell_language_alias_is_accepted(self) -> None:
        args = run_render_only._parse_args(["video-id", "-Lang", "en,ru"])

        self.assertEqual(args.lang, ("en", "ru"))

    def test_invalid_language_is_rejected_clearly(self) -> None:
        with self.assertRaisesRegex(SystemExit, "2"):
            run_render_only._parse_args(["--lang", "en,de"])

    @patch("run_render_only.restyle_existing_shorts")
    def test_selected_languages_are_forwarded_to_restyle(self, restyle) -> None:
        restyle.return_value = {
            "stories": [
                {"language": "en"},
                {"language": "es"},
                {"language": "ru"},
            ]
        }

        run_render_only.run_render_only(
            ["video-id"],
            publish=False,
            languages=("es",),
        )

        restyle.assert_called_once_with(
            "video-id",
            publish_dir=None,
            languages=("es",),
        )

    def test_publish_leaves_unselected_language_output_untouched(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_en = root / "source-en.mp4"
            source_es = root / "source-es.mp4"
            source_en.write_bytes(b"new-en")
            source_es.write_bytes(b"new-es")
            existing_en = root / "out" / "en" / source_en.name
            existing_en.parent.mkdir(parents=True)
            existing_en.write_bytes(b"existing-en")

            copied = publish_render_outputs(
                {
                    "stories": [
                        {"language": "en", "video_path": str(source_en)},
                        {"language": "es", "video_path": str(source_es)},
                    ]
                },
                root / "out",
                languages=("es",),
            )

            self.assertEqual(copied, [root / "out" / "es" / source_es.name])
            self.assertEqual(existing_en.read_bytes(), b"existing-en")


if __name__ == "__main__":
    unittest.main()
