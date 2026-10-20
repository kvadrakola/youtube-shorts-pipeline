from __future__ import annotations

import unittest

from modules.outro import outro_music_audio_filter_chain


class OutroMusicFilterTests(unittest.TestCase):
    def test_eight_second_clip_matches_ffmpeg_af_chain(self) -> None:
        self.assertEqual(
            outro_music_audio_filter_chain(8.0),
            "volume=0.4,afade=t=in:st=0:d=0.8,afade=t=out:st=7.2:d=0.8",
        )


if __name__ == "__main__":
    unittest.main()
