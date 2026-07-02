from __future__ import annotations

import json

from granite_speech.writers import format_timestamp, write_result


def result() -> dict:
    return {
        "text": "hello world",
        "segments": [
            {"start": 0.0, "end": 1.25, "text": "hello"},
            {"start": 1.25, "end": 2.5, "text": "", "error": "boom"},
            {"start": 2.5, "end": 3.0, "text": "world"},
        ],
        "language": "en",
        "target_language": None,
        "warnings": [
            {"type": "window_error", "start": 1.25, "end": 2.5, "text": "", "error": "boom"}
        ],
    }


def test_format_timestamp():
    assert format_timestamp(3661.2345) == "01:01:01.234"
    assert format_timestamp(1.2, decimal_marker=",") == "00:00:01,200"


def test_write_all_formats(tmp_path):
    paths = write_result(result(), "clip.wav", output_dir=tmp_path, output_format="all")

    assert {path.suffix for path in paths} == {".txt", ".srt", ".vtt", ".tsv", ".json"}
    assert (tmp_path / "clip.txt").read_text() == "hello world\n"
    assert "00:00:00,000 --> 00:00:01,250" in (tmp_path / "clip.srt").read_text()
    assert "boom" not in (tmp_path / "clip.srt").read_text()
    assert (tmp_path / "clip.vtt").read_text().startswith("WEBVTT")
    assert (tmp_path / "clip.tsv").read_text().splitlines()[0] == "start\tend\ttext"
    assert json.loads((tmp_path / "clip.json").read_text())["warnings"][0]["error"] == "boom"
