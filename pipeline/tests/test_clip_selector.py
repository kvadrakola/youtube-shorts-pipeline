from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from modules.clip_selector import (
    _parse_selection,
    select_clips_from_transcript,
    strategy_system_prompt,
)


def _clip(index: int) -> dict[str, object]:
    return {
        "clip_id": f"clip_{index:02d}",
        "rank": index,
        "title": f"Title {index}",
        "hook": f"Hook {index}",
        "hook_score": 8.5,
        "viral_score": 8.0,
        "start_time": (index - 1) * 50.0,
        "end_time": (index - 1) * 50.0 + 40.0,
        "raw_text": f"Complete transcript text {index}",
        "reasoning": "Strong standalone idea",
        "speaker": "SPEAKER_00",
        "context_adjustments_made": "none",
    }


class ClipSelectionParsingTests(unittest.TestCase):
    def test_recovers_all_complete_objects_from_truncated_fenced_json(self) -> None:
        complete = [_clip(index) for index in range(1, 13)]
        content = (
            "```json\n"
            + '{"clips":['
            + ",".join(json.dumps(clip) for clip in complete)
            + ',{"clip_id":"clip_13","rank":13,"title":"cut off'
        )

        result = _parse_selection(content)

        self.assertEqual(len(result.clips), 12)
        self.assertEqual(result.clips[-1].clip_id, "clip_12")

    def test_preserves_top_level_list_and_schema_validation(self) -> None:
        result = _parse_selection(json.dumps([_clip(1)]))
        self.assertEqual(result.clips[0].title, "Title 1")

        invalid = _clip(2)
        del invalid["speaker"]
        with self.assertRaises(ValidationError):
            _parse_selection(json.dumps({"clips": [invalid]}))

    def test_unrelated_malformed_json_still_raises_decode_error(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            _parse_selection('{"unexpected": ')


class ClipSelectionRequestTests(unittest.TestCase):
    def test_prompt_requests_broad_timeline_spread_and_concise_summary(self) -> None:
        prompt = strategy_system_prompt("1_categorized")
        self.assertIn("28–30 clips", prompt)
        self.assertIn("0–25m, 25–50m, 50m+", prompt)
        self.assertIn("prevent timestamp clustering", prompt)
        self.assertIn("Exactly 2 concise sentences", prompt)

    @patch("modules.clip_selector.get_settings")
    @patch("modules.clip_selector._deepseek_client")
    def test_deepseek_request_sets_max_tokens(
        self,
        deepseek_client: MagicMock,
        get_settings: MagicMock,
    ) -> None:
        get_settings.return_value = SimpleNamespace(
            DEEPSEEK_MODEL="deepseek-chat",
            BRAND_NAME="Example Channel",
            BRAND_WEBSITE_URL="https://example.com",
            CLIP_SELECTOR_CONTEXT="",
        )
        client = deepseek_client.return_value
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"clips": [_clip(index) for index in range(1, 7)]}),
                    )
                )
            ]
        )
        transcript = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 40.0,
                    "speaker": "SPEAKER_00",
                    "text": "A complete standalone thought.",
                }
            ]
        }

        result = select_clips_from_transcript(transcript)

        self.assertEqual(len(result.clips), 6)
        self.assertEqual(client.chat.completions.create.call_args.kwargs["max_tokens"], 8192)


if __name__ == "__main__":
    unittest.main()
