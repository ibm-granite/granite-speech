# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from granite_speech.errors import InvalidArgumentError
from granite_speech.writers import format_timestamp, get_writer, write_result


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


def test_subtitle_layout_wraps_srt_and_vtt_without_changing_text_or_json(tmp_path):
    data = result()
    data["text"] = "hello brave new world"
    data["segments"][0]["text"] = "hello brave new world"

    write_result(
        data,
        "clip.wav",
        output_dir=tmp_path,
        output_format="all",
        max_line_width=11,
        max_line_count=2,
    )

    assert "hello brave\nnew world" in (tmp_path / "clip.srt").read_text()
    assert "hello brave\nnew world" in (tmp_path / "clip.vtt").read_text()
    assert (tmp_path / "clip.txt").read_text() == "hello brave new world\n"
    assert json.loads((tmp_path / "clip.json").read_text())["text"] == "hello brave new world"


def test_subtitle_layout_can_cap_line_count(tmp_path):
    data = result()
    data["segments"][0]["text"] = "one two three four five six"

    write_result(
        data,
        "clip.wav",
        output_dir=tmp_path,
        output_format="srt",
        max_line_width=9,
        max_line_count=1,
    )

    assert "one two three four five six" in (tmp_path / "clip.srt").read_text()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_line_width": 0},
        {"max_line_width": -1},
        {"max_line_count": 0},
        {"max_line_count": -1},
    ],
)
def test_subtitle_layout_rejects_invalid_limits(tmp_path, kwargs):
    with pytest.raises(InvalidArgumentError):
        write_result(result(), "clip.wav", output_dir=tmp_path, output_format="srt", **kwargs)


def test_get_writer_accepts_subtitle_layout_options(tmp_path):
    writer = get_writer("srt", tmp_path, max_line_width=11)

    writer(result(), "clip.wav")

    assert (tmp_path / "clip.srt").exists()
