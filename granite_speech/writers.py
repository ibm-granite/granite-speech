from __future__ import annotations

import json
import textwrap
from collections.abc import Callable
from pathlib import Path

from .errors import InvalidArgumentError

# Concrete formats in the order "all" emits them; OUTPUT_FORMATS adds the "all"
# meta-format that expands to every entry here.
_CONCRETE_FORMATS = ("txt", "srt", "vtt", "tsv", "json")
OUTPUT_FORMATS = {*_CONCRETE_FORMATS, "all"}


def get_writer(
    output_format: str,
    output_dir: str | Path,
    *,
    max_line_width: int | None = None,
    max_line_count: int | None = None,
) -> Callable[[dict, str | Path], list[Path]]:
    """Return a writer that serializes a result dict to ``output_dir``.

    The output request is validated up front (so an invalid ``output_format`` or
    subtitle-layout option raises immediately, not when the writer is called).
    The returned callable takes ``(result, audio_path)``, writes one file per
    concrete format (``"all"`` expands to every format), and returns the paths
    written. ``max_line_width`` / ``max_line_count`` apply only to SRT/VTT cues.
    """
    _validate_output_request(
        output_format, max_line_width=max_line_width, max_line_count=max_line_count
    )

    def writer(result: dict, audio_path: str | Path) -> list[Path]:
        return write_result(
            result,
            audio_path,
            output_dir=output_dir,
            output_format=output_format,
            max_line_width=max_line_width,
            max_line_count=max_line_count,
        )

    return writer


def write_result(
    result: dict,
    audio_path: str | Path,
    *,
    output_dir: str | Path,
    output_format: str,
    max_line_width: int | None = None,
    max_line_count: int | None = None,
) -> list[Path]:
    _validate_output_request(
        output_format, max_line_width=max_line_width, max_line_count=max_line_count
    )
    formats = list(_CONCRETE_FORMATS) if output_format == "all" else [output_format]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem
    paths: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        _write_one(
            result,
            path,
            fmt,
            max_line_width=max_line_width,
            max_line_count=max_line_count,
        )
        paths.append(path)
    return paths


def _write_one(
    result: dict,
    path: Path,
    fmt: str,
    *,
    max_line_width: int | None,
    max_line_count: int | None,
) -> None:
    if fmt == "txt":
        path.write_text(_format_txt(result), encoding="utf-8")
    elif fmt == "srt":
        path.write_text(
            _format_srt(
                result,
                max_line_width=max_line_width,
                max_line_count=max_line_count,
            ),
            encoding="utf-8",
        )
    elif fmt == "vtt":
        path.write_text(
            _format_vtt(
                result,
                max_line_width=max_line_width,
                max_line_count=max_line_count,
            ),
            encoding="utf-8",
        )
    elif fmt == "tsv":
        path.write_text(_format_tsv(result), encoding="utf-8")
    elif fmt == "json":
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:  # pragma: no cover - guarded by caller
        raise InvalidArgumentError(f"unsupported output format {fmt!r}")


def _format_txt(result: dict) -> str:
    text = result.get("text", "")
    return text + ("\n" if text else "")


def _format_srt(
    result: dict,
    *,
    max_line_width: int | None = None,
    max_line_count: int | None = None,
) -> str:
    return _format_cues(
        result,
        decimal_marker=",",
        numbered=True,
        max_line_width=max_line_width,
        max_line_count=max_line_count,
    )


def _format_vtt(
    result: dict,
    *,
    max_line_width: int | None = None,
    max_line_count: int | None = None,
) -> str:
    return _format_cues(
        result,
        header="WEBVTT\n",
        decimal_marker=".",
        numbered=False,
        max_line_width=max_line_width,
        max_line_count=max_line_count,
    )


def _format_cues(
    result: dict,
    *,
    header: str | None = None,
    decimal_marker: str,
    numbered: bool,
    max_line_width: int | None,
    max_line_count: int | None,
) -> str:
    cues = [header] if header is not None else []
    for idx, segment in enumerate(_successful_text_segments(result), start=1):
        text = _format_subtitle_text(
            segment["text"],
            max_line_width=max_line_width,
            max_line_count=max_line_count,
        )
        start = format_timestamp(segment["start"], decimal_marker=decimal_marker)
        end = format_timestamp(segment["end"], decimal_marker=decimal_marker)
        index = f"{idx}\n" if numbered else ""
        cues.append(f"{index}{start} --> {end}\n{text}\n")
    return "\n".join(cues)


def _format_tsv(result: dict) -> str:
    rows = ["start\tend\ttext"]
    for segment in _successful_text_segments(result):
        rows.append(f"{segment['start']:.3f}\t{segment['end']:.3f}\t{segment['text'].strip()}")
    return "\n".join(rows) + "\n"


def _successful_text_segments(result: dict):
    for segment in result.get("segments", []):
        if "error" in segment:
            continue
        if not segment.get("text", "").strip():
            continue
        yield segment


def _format_subtitle_text(
    text: str,
    *,
    max_line_width: int | None,
    max_line_count: int | None,
) -> str:
    """Wrap one cue's text into subtitle lines.

    With neither limit set, the text is returned stripped and otherwise
    untouched. ``max_line_width`` wraps on word boundaries (never breaking words
    or hyphens, so a single word longer than the width overflows rather than
    splitting). ``max_line_count`` then caps the number of lines: any overflow
    is folded back into the last allowed line so no text is dropped — with
    ``max_line_count=1`` the whole cue collapses onto one line.
    """
    stripped = text.strip()
    if max_line_width is None and max_line_count is None:
        return stripped

    normalized = " ".join(stripped.split())
    if not normalized:
        return ""

    if max_line_width is None:
        lines = [normalized]
    else:
        lines = textwrap.wrap(
            normalized,
            width=max_line_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if not lines:
            lines = [normalized]

    if max_line_count is not None and len(lines) > max_line_count:
        if max_line_count == 1:
            lines = [" ".join(lines)]
        else:
            lines = lines[: max_line_count - 1] + [" ".join(lines[max_line_count - 1 :])]
    return "\n".join(lines)


def _validate_output_request(
    output_format: str,
    *,
    max_line_width: int | None,
    max_line_count: int | None,
) -> None:
    if output_format not in OUTPUT_FORMATS:
        raise InvalidArgumentError(
            f"output_format must be one of {', '.join(sorted(OUTPUT_FORMATS))}"
        )
    _validate_subtitle_layout(max_line_width=max_line_width, max_line_count=max_line_count)


def _validate_subtitle_layout(
    *,
    max_line_width: int | None,
    max_line_count: int | None,
) -> None:
    if max_line_width is not None and (
        isinstance(max_line_width, bool) or max_line_width <= 0
    ):
        raise InvalidArgumentError("max_line_width must be greater than 0")
    if max_line_count is not None and (
        isinstance(max_line_count, bool) or max_line_count <= 0
    ):
        raise InvalidArgumentError("max_line_count must be greater than 0")


def format_timestamp(seconds: float, *, decimal_marker: str = ".") -> str:
    milliseconds = round(max(seconds, 0.0) * 1000)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    millis = milliseconds % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{millis:03d}"
