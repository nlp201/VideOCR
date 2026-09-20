"""Tests for the optional bounding-box metadata JSON output.

These exercise the real serializer and the real dual-zone merge: OCR results
are injected as representative model records at the executable/hardware lookup
boundary, but every coordinate mapping, cue assembly and file-staging path runs
unmodified.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import unittest
from collections.abc import Iterator
from typing import Any, cast
from unittest import mock

from videocr import utils
from videocr.api import save_subtitles_to_file
from videocr.models import PredictedFrames
from videocr.video import Video


def _box(x: float, y: float, w: float, h: float) -> list[list[float]]:
    """A clockwise quadrilateral in OCR-image pixel space."""
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _make_frame(
    index: int, zone_index: int, words: list[tuple[str, list[list[float]], float]], lang: str = "en"
) -> PredictedFrames:
    """Build a real PredictedFrames from representative OCR records."""
    records: list[list[Any]] = [[bbox, [text, conf]] for text, bbox, conf in words]
    pred_data: list[list[Any]] = [records]
    return PredictedFrames("paddleocr", index, pred_data, 0.0, zone_index, lang, False)


def _make_zone(
    crop_x: int, crop_y: int, crop_w: int, crop_h: int, target_w: int, target_h: int, midpoint_y: float
) -> dict[str, Any]:
    """A validated zone carrying the exact geometry handed to FFmpeg."""
    return {
        "x_start": crop_x,
        "y_start": crop_y,
        "x_end": crop_x + crop_w,
        "y_end": crop_y + crop_h,
        "midpoint_y": midpoint_y,
        "w": target_w,
        "h": target_h,
        "ocr_crop_x": crop_x,
        "ocr_crop_y": crop_y,
        "ocr_crop_w": crop_w,
        "ocr_crop_h": crop_h,
        "ocr_target_w": target_w,
        "ocr_target_h": target_h,
        "crop_str": f"{crop_w}:{crop_h}:{crop_x}:{crop_y}",
        "scale_str": f"{target_w}:{target_h}:flags=area:threads=1",
    }


def _build_video(
    zones: list[dict[str, Any]],
    frames_zone1: list[PredictedFrames],
    frames_zone2: list[PredictedFrames],
    timestamps: dict[int, float],
    *,
    width: int = 1280,
    height: int = 720,
) -> Video:
    """Construct a real Video without a media file or OCR engine."""
    props: dict[str, Any] = {
        "width": width,
        "height": height,
        "duration_ms": 900_000,
        "start_time_offset_ms": 0.0,
    }
    with mock.patch("videocr.video.get_video_properties", return_value=props):
        video = Video("dummy.mp4", "/bin/true", "/m/det", "/m/rec", "/m/cls", "/bin/true")
    video.validated_zones = zones
    video.pred_frames_zone1 = frames_zone1
    video.pred_frames_zone2 = frames_zone2
    video.frame_timestamps = timestamps
    video.start_time_offset_ms = 0.0
    return video


# Shared single-cue fixtures: crop offset plus deliberately unequal axis factors
# (404/200 = 2.02 horizontally vs 300/150 = 2.0 vertically).
_ZONE = _make_zone(100, 50, 404, 300, 200, 150, 360)
_FRAME = _make_frame(
    0,
    0,
    [
        ("Hello", _box(0.0, 10.0, 50.0, 20.0), 0.95),
        ("안녕", _box(60.0, 10.0, 50.0, 20.0), 0.92),
    ],
)
_TIMESTAMPS = {0: 1234.0, 1: 5678.0}


class JsonMetadataTests(unittest.TestCase):
    def test_cue_text_and_fractional_timestamps(self) -> None:
        video = _build_video([_ZONE], [_FRAME], [], _TIMESTAMPS)
        srt = video.get_subtitles(80, 0.1, "en", False, 0.2, [None, None])
        meta = video.get_boxes_metadata([None, None])

        self.assertEqual(meta["schema_version"], 2)
        cues = cast("list[dict[str, Any]]", meta["cues"])
        self.assertEqual(len(cues), 1)

        cue = cues[0]
        self.assertEqual(cue["text"], "Hello 안녕")
        self.assertIn("Hello 안녕", srt)

        # The millisecond fields mirror the SRT quantization exactly.
        self.assertEqual(cue["start_ms"], utils.quantize_timestamp_ms(1234.0))
        self.assertEqual(cue["end_ms"], utils.quantize_timestamp_ms(5678.0))
        self.assertIn("00:00:01,234", srt)
        self.assertIn("00:00:05,678", srt)

    def test_full_frame_mapping_uses_offset_and_unequal_factors(self) -> None:
        video = _build_video([_ZONE], [_FRAME], [], _TIMESTAMPS)
        video.get_subtitles(80, 0.1, "en", False, 0.2, [None, None])
        meta = video.get_boxes_metadata([None, None])
        cues = cast("list[dict[str, Any]]", meta["cues"])
        region = cues[0]["regions"][0]
        line = region["lines"][0]

        # First word "Hello" sits at OCR (0, 10); mapped back the crop offset
        # (100, 50) and the separate scale factors apply.
        self.assertEqual(line["words"][0]["text"], "Hello")
        self.assertEqual(line["words"][0]["bounding_box"][0], [100.0, 70.0])
        self.assertEqual(line["words"][0]["bounding_box"][1], [201.0, 70.0])

        zone = video.validated_zones[0]
        x_factor = zone["ocr_crop_w"] / zone["ocr_target_w"]
        y_factor = zone["ocr_crop_h"] / zone["ocr_target_h"]
        self.assertNotEqual(x_factor, y_factor)
        self.assertEqual(zone["ocr_crop_x"], 100)
        self.assertEqual(zone["ocr_crop_y"], 50)

    def test_dual_zone_merge_keeps_both_regions(self) -> None:
        top_zone = _make_zone(0, 0, 640, 360, 320, 180, 180)
        bottom_zone = _make_zone(0, 360, 640, 360, 320, 180, 540)
        top_frame = _make_frame(0, 0, [("LineA", _box(0.0, 0.0, 40.0, 20.0), 0.9)], lang="en")
        bottom_frame = _make_frame(0, 1, [("LineB", _box(0.0, 0.0, 40.0, 20.0), 0.9)], lang="en")

        video = _build_video(
            [top_zone, bottom_zone], [top_frame], [bottom_frame], {0: 0.0, 1: 1000.0}
        )
        # Equal alignments force the dual-zone merge path rather than a sort.
        video.get_subtitles(80, 0.1, "en", False, 0.2, [None, None])
        meta = video.get_boxes_metadata([None, None])
        cues = cast("list[dict[str, Any]]", meta["cues"])
        self.assertEqual(len(cues), 1)

        regions = cues[0]["regions"]
        self.assertEqual(len(regions), 2)
        zone_indexes = {region["zone_index"] for region in regions}
        self.assertEqual(zone_indexes, {0, 1})
        output_texts = {region["output_text"] for region in regions}
        self.assertEqual(output_texts, {"LineA", "LineB"})


_PROPS: dict[str, Any] = {
    "width": 1280,
    "height": 720,
    "duration_ms": 900_000,
    "start_time_offset_ms": 0.0,
}

_SAVE_ZONE = _make_zone(0, 540, 1280, 180, 640, 90, 600)
_SAVE_FRAME = _make_frame(0, 0, [("Subtitle", _box(10.0, 5.0, 80.0, 20.0), 0.9)])


def _fake_run_ocr(self: Video, *args: Any, **kwargs: Any) -> None:
    """Stand in for the OCR engine, injecting representative records only."""
    self.validated_zones = [_SAVE_ZONE]
    self.pred_frames_zone1 = [_SAVE_FRAME]
    self.pred_frames_zone2 = []
    self.frame_timestamps = {0: 1000.0, 1: 4000.0}
    self.start_time_offset_ms = 0.0


@contextlib.contextmanager
def _patched_save_api() -> Iterator[None]:
    """Apply the minimal boundary mocks the save path needs under test.

    Only the OCR engine, video property lookup, hardware check, executable
    discovery and model-dir resolution are stubbed; everything else runs the
    real production code.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(Video, "run_ocr", _fake_run_ocr))
        stack.enter_context(mock.patch("videocr.video.get_video_properties", return_value=_PROPS))
        stack.enter_context(mock.patch("videocr.utils.perform_hardware_check", return_value=None))
        stack.enter_context(mock.patch("videocr.utils.find_executable", return_value="/usr/bin/paddleocr"))
        stack.enter_context(mock.patch("videocr.utils.resolve_model_dirs", return_value=("/det", "/rec", "/cls")))
        yield


class SaveApiTests(unittest.TestCase):
    def test_save_writes_opt_in_json_and_srt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            srt_path = os.path.join(tmp, "out.srt")
            json_path = os.path.join(tmp, "out.boxes.json")

            with _patched_save_api():
                save_subtitles_to_file(
                    "dummy.mp4", srt_path, lang="en", boxes_path=json_path
                )

            self.assertTrue(os.path.exists(srt_path))
            self.assertTrue(os.path.exists(json_path))
            with open(srt_path, encoding="utf-8") as f:
                self.assertIn("Subtitle", f.read())
            with open(json_path, encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["coordinate_space"], "full-frame-pixels")

    def test_save_without_boxes_path_is_srt_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            srt_path = os.path.join(tmp, "out.srt")
            json_path = os.path.join(tmp, "out.boxes.json")

            with _patched_save_api():
                save_subtitles_to_file("dummy.mp4", srt_path, lang="en")

            self.assertTrue(os.path.exists(srt_path))
            self.assertFalse(os.path.exists(json_path))

    def test_save_rejects_equal_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            same = os.path.join(tmp, "out.srt")
            with self.assertRaises(SystemExit):
                save_subtitles_to_file("dummy.mp4", same, lang="en", boxes_path=same)

    def test_serialization_failure_leaves_prior_outputs_intact(self) -> None:
        broken: dict[str, Any] = {"bad": set([1, 2, 3])}

        def _broken(_self: Video, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return broken

        with tempfile.TemporaryDirectory() as tmp:
            srt_path = os.path.join(tmp, "out.srt")
            json_path = os.path.join(tmp, "out.boxes.json")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write("PRIOR_SRT_CONTENT")

            with (
                _patched_save_api(),
                mock.patch.object(Video, "get_boxes_metadata", _broken),
                self.assertRaises(TypeError),
            ):
                save_subtitles_to_file(
                    "dummy.mp4", srt_path, lang="en", boxes_path=json_path
                )

            with open(srt_path, encoding="utf-8") as f:
                self.assertEqual(f.read(), "PRIOR_SRT_CONTENT")
            self.assertFalse(os.path.exists(json_path))


if __name__ == "__main__":
    unittest.main()
