from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._models import DEFAULT_MODEL
from .errors import GraniteSpeechError
from .loader import DEFAULT_LLAMA_CPP_QUANT, download_llama_cpp_model, load_model
from .prompts import PROMPT_MODES
from .writers import OUTPUT_FORMATS, write_result


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "download":
        return _download_main(argv[1:])
    return _transcribe_main(argv)


def _transcribe_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="granite-speech",
        epilog="Use `granite-speech download <model>` to prefetch model weights.",
    )
    parser.add_argument("audio", nargs="+", help="audio file path(s) to transcribe")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name or local path")
    parser.add_argument("--device", default=None, help="device override, e.g. cuda, mps, cpu")
    parser.add_argument("--download_root", "--download-root", default=None)
    parser.add_argument("--model_dir", "--model-dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local_files_only", "--local-files-only", action="store_true")
    parser.add_argument("--llama_cpp_binary", "--llama-cpp-binary", default=None)
    parser.add_argument("--llama_cpp_quant", "--llama-cpp-quant", default=DEFAULT_LLAMA_CPP_QUANT)
    parser.add_argument("--llama_cpp_mmproj", "--llama-cpp-mmproj", default=None)
    parser.add_argument(
        "--llama_cpp_arg",
        "--llama-cpp-arg",
        action="append",
        default=None,
        help="extra argument to pass through to llama-cli; repeat for multiple arguments",
    )
    parser.add_argument("--llama_cpp_timeout", "--llama-cpp-timeout", type=float, default=None)
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--language", default=None)
    parser.add_argument("--target_language", "--target-language", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--initial_prompt", "--initial-prompt", default=None)
    parser.add_argument(
        "--prompt_mode",
        "--prompt-mode",
        default="default",
        choices=sorted(PROMPT_MODES),
    )
    parser.add_argument("--prefix_text", "--prefix-text", default=None)
    parser.add_argument(
        "--keyword",
        "--keyword_bias",
        "--keyword-bias",
        dest="keyword_biases",
        action="append",
        default=None,
        help="keyword or phrase to bias recognition toward; repeat for multiple hints",
    )
    parser.add_argument(
        "--max_new_tokens",
        "--max-new-tokens",
        type=int,
        default=None,
        help="maximum generated tokens per audio window; omitted means auto-size from chunk length",
    )
    parser.add_argument("--num_beams", "--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--word_timestamps", "--word-timestamps", action="store_true")
    parser.add_argument("--chunk_length", "--chunk-length", type=float, default=30.0)
    parser.add_argument("--chunk_overlap", "--chunk-overlap", type=float, default=0.0)
    parser.add_argument("--segmentation", choices=["fixed", "vad"], default="fixed")
    parser.add_argument("--vad_threshold", "--vad-threshold", type=float, default=0.2)
    parser.add_argument(
        "--vad_min_speech_duration",
        "--vad-min-speech-duration",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--vad_min_silence_duration",
        "--vad-min-silence-duration",
        type=float,
        default=0.5,
    )
    parser.add_argument("--vad_speech_pad", "--vad-speech-pad", type=float, default=0.2)
    parser.add_argument(
        "--output_format",
        "--output-format",
        default="txt",
        choices=sorted(OUTPUT_FORMATS),
    )
    parser.add_argument("--output_dir", "--output-dir", default=".")
    args = parser.parse_args(argv)
    download_root = _resolve_download_root(args, parser)

    try:
        model = load_model(
            args.model,
            device=args.device,
            download_root=download_root,
            revision=args.revision,
            local_files_only=args.local_files_only,
            llama_cpp_binary=args.llama_cpp_binary,
            llama_cpp_quant=args.llama_cpp_quant,
            llama_cpp_mmproj=args.llama_cpp_mmproj,
            llama_cpp_extra_args=args.llama_cpp_arg,
            llama_cpp_timeout=args.llama_cpp_timeout,
        )
    except GraniteSpeechError as exc:
        print(f"granite-speech: {exc}", file=sys.stderr)
        return 2

    exit_code = 0
    for audio_path in args.audio:
        per_file_code = 0
        try:
            result = model.transcribe(
                audio_path,
                task=args.task,
                language=args.language,
                target_language=args.target_language,
                prompt=args.prompt,
                prompt_mode=args.prompt_mode,
                initial_prompt=args.initial_prompt,
                prefix_text=args.prefix_text,
                keyword_biases=args.keyword_biases,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                temperature=args.temperature,
                verbose=args.verbose,
                word_timestamps=args.word_timestamps,
                chunk_length=args.chunk_length,
                chunk_overlap=args.chunk_overlap,
                segmentation=args.segmentation,
                vad_threshold=args.vad_threshold,
                vad_min_speech_duration=args.vad_min_speech_duration,
                vad_min_silence_duration=args.vad_min_silence_duration,
                vad_speech_pad=args.vad_speech_pad,
            )
            write_result(
                result,
                audio_path,
                output_dir=args.output_dir,
                output_format=args.output_format,
            )
            if _has_window_error(result):
                per_file_code = 1
            _print_warning_summary(audio_path, result)
        except GraniteSpeechError as exc:
            per_file_code = 2
            print(f"granite-speech: {audio_path}: {exc}", file=sys.stderr)
        except Exception as exc:
            per_file_code = 2
            print(f"granite-speech: {audio_path}: unexpected error: {exc}", file=sys.stderr)
        exit_code = max(exit_code, per_file_code)
    return exit_code


def _download_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="granite-speech download")
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="model name or local path")
    parser.add_argument("--llama_cpp_quant", "--llama-cpp-quant", default=DEFAULT_LLAMA_CPP_QUANT)
    parser.add_argument("--llama_cpp_mmproj", "--llama-cpp-mmproj", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--download_root", "--download-root", default=None)
    parser.add_argument("--model_dir", "--model-dir", default=None)
    parser.add_argument("--local_files_only", "--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    download_root = _resolve_download_root(args, parser)

    try:
        model_path, mmproj_path = download_llama_cpp_model(
            args.model,
            quant=args.llama_cpp_quant,
            mmproj_path=args.llama_cpp_mmproj,
            revision=args.revision,
            download_root=download_root,
            local_files_only=args.local_files_only,
        )
        print(model_path, file=sys.stderr)
        print(mmproj_path, file=sys.stderr)
        return 0
    except GraniteSpeechError as exc:
        print(f"granite-speech download: {exc}", file=sys.stderr)
        return 2


def _has_window_error(result: dict) -> bool:
    return any(
        warning.get("type") == "window_error" or bool(warning.get("error"))
        for warning in result.get("warnings", [])
    )


def _resolve_download_root(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str | None:
    if args.model_dir is None:
        return args.download_root
    if args.download_root is not None and args.download_root != args.model_dir:
        parser.error("pass either --download_root or --model_dir, not both")
    return args.model_dir


def _print_warning_summary(audio_path: str | Path, result: dict) -> None:
    warnings = result.get("warnings", [])
    if not warnings:
        return
    window_errors = sum(1 for warning in warnings if warning.get("type") == "window_error")
    advisories = len(warnings) - window_errors
    pieces = []
    if window_errors:
        pieces.append(f"{window_errors} failed window(s)")
    if advisories:
        pieces.append(f"{advisories} advisory warning(s)")
    print(f"granite-speech: {audio_path}: {', '.join(pieces)}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
