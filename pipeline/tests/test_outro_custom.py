from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config.settings import get_settings
from modules.outro import (
    ensure_outro_video,
    outro_video_path,
    resolve_custom_outro_mp4,
    resolve_outro_video_path,
)


class CustomOutroResolutionTests(unittest.TestCase):
    def test_default_uses_rendered_cache_path(self) -> None:
        get_settings.cache_clear()
        self.assertIsNone(resolve_custom_outro_mp4("en"))
        self.assertEqual(resolve_outro_video_path("en"), outro_video_path("en"))

    def test_single_custom_mp4_takes_priority(self) -> None:
        get_settings.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "channel_outro.mp4"
            custom.write_bytes(b"\x00\x00\x00\x18ftypmp42")
            with patch.dict(
                "os.environ",
                {"BRAND_OUTRO_CUSTOM_MP4": str(custom)},
                clear=False,
            ):
                get_settings.cache_clear()
                resolved = resolve_custom_outro_mp4("es")
                self.assertEqual(resolved, custom.resolve())
                self.assertEqual(resolve_outro_video_path("ru"), custom.resolve())

    def test_per_language_custom_dir(self) -> None:
        get_settings.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom"
            custom_dir.mkdir()
            ru_file = custom_dir / "outro_ru.mp4"
            ru_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")
            with patch.dict(
                "os.environ",
                {"BRAND_OUTRO_CUSTOM_DIR": str(custom_dir)},
                clear=False,
            ):
                get_settings.cache_clear()
                self.assertIsNone(resolve_custom_outro_mp4("en"))
                self.assertEqual(resolve_custom_outro_mp4("ru"), ru_file.resolve())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required for silent outro audio test")
    def test_ensure_outro_music_file_is_silent_without_url(self) -> None:
        get_settings.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "silent.mp3"
            with patch.dict("os.environ", {"OUTRO_MUSIC_URL": "", "OUTRO_MUSIC_PATH": str(destination)}, clear=False):
                get_settings.cache_clear()
                from modules.outro import ensure_outro_music_file

                path = ensure_outro_music_file(destination=destination, duration=2.0)
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_ensure_outro_skips_html_render_when_custom_exists(self) -> None:
        get_settings.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "my_outro.mp4"
            custom.write_bytes(b"\x00\x00\x00\x18ftypmp42")
            with patch.dict("os.environ", {"BRAND_OUTRO_CUSTOM_MP4": str(custom)}, clear=False):
                get_settings.cache_clear()
                with patch("modules.outro.render_html_outro_to_video") as render:
                    path = ensure_outro_video(language="en")
                render.assert_not_called()
                self.assertEqual(path, custom.resolve())


if __name__ == "__main__":
    unittest.main()
