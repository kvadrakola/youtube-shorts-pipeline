from __future__ import annotations

import unittest

from modules.clip_selector import SelectedClip, deduplicate_ranked_clips
from modules.shorts import LANGUAGE_CLIP_RANKS, allocate_clips_by_language


def _clip(index: int, start: float, end: float, score: float = 8.0) -> SelectedClip:
    return SelectedClip(
        clip_id=f"candidate_{index}",
        rank=index,
        title=f"Title {index}",
        hook=f"Hook {index}",
        hook_score=score,
        viral_score=score,
        start_time=start,
        end_time=end,
        raw_text=f"Unique text {index}",
        reasoning="test",
        speaker="SPEAKER_00",
    )


class ClipMatrixTests(unittest.TestCase):
    def test_overlap_filter_keeps_higher_scored_clip(self) -> None:
        lower = _clip(1, 0.0, 40.0, 6.0)
        higher = _clip(2, 20.0, 60.0, 9.0)
        separate = _clip(3, 70.0, 110.0, 7.0)

        result = deduplicate_ranked_clips([lower, higher, separate], limit=2)

        self.assertEqual([clip.title for clip in result], ["Title 2", "Title 3"])
        self.assertEqual([clip.rank for clip in result], [1, 2])
        self.assertEqual([clip.clip_id for clip in result], ["clip_01", "clip_02"])

    def test_language_matrix_is_six_per_language_and_eighteen_total(self) -> None:
        clips = [_clip(index, index * 70.0, index * 70.0 + 40.0) for index in range(1, 13)]
        allocated = allocate_clips_by_language(clips)

        self.assertEqual({key: len(value) for key, value in allocated.items()}, {
            "en": 6,
            "es": 6,
            "ru": 6,
        })
        self.assertEqual(sum(map(len, allocated.values())), 18)
        for language, ranks in LANGUAGE_CLIP_RANKS.items():
            self.assertEqual([clip.rank for clip in allocated[language]], list(ranks))

    def test_fallback_distributes_six_to_eleven_clips_round_robin(self) -> None:
        clips = [_clip(index, index * 70.0, index * 70.0 + 40.0) for index in range(1, 9)]

        allocated = allocate_clips_by_language(clips)

        self.assertEqual([clip.rank for clip in allocated["en"]], [1, 2, 3, 4, 7])
        self.assertEqual([clip.rank for clip in allocated["es"]], [1, 2, 3, 5, 8])
        self.assertEqual([clip.rank for clip in allocated["ru"]], [1, 2, 3, 6])

    def test_language_allocation_rejects_fewer_than_six_clips(self) -> None:
        clips = [_clip(index, index * 70.0, index * 70.0 + 40.0) for index in range(1, 6)]

        with self.assertRaisesRegex(ValueError, "at least 6"):
            allocate_clips_by_language(clips)


if __name__ == "__main__":
    unittest.main()
