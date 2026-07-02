from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections.abc import Sequence

import numpy as np

from . import BackendCapabilities, GenerateRequest, GenerateResult, require_sample_rate


class LlamaCppBackend:
    name = "llama.cpp"

    def __init__(
        self,
        *,
        binary: str,
        model_path: str,
        mmproj_path: str,
        capabilities: BackendCapabilities,
        extra_args: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> None:
        self.binary = binary
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.capabilities = capabilities
        self.extra_args = tuple(extra_args or ())
        self.timeout = timeout

    def generate(self, req: GenerateRequest) -> GenerateResult:
        require_sample_rate(req.sample_rate)
        if req.num_beams != 1:
            raise ValueError("llama.cpp backend does not support beam search; use num_beams=1")

        prompt = req.instruction or _strip_audio_token(req.prompt)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="granite-speech-", delete=False
        ) as tmp:
            audio_path = tmp.name
        try:
            _write_wav(audio_path, req.wav, req.sample_rate)
            cmd = self._command(req=req, audio_path=audio_path, prompt=prompt)
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        output = completed.stdout or ""
        if completed.returncode != 0:
            detail = _last_nonempty_line(output) or f"exit code {completed.returncode}"
            raise RuntimeError(f"llama.cpp generation failed: {detail}")

        return GenerateResult(text=_extract_transcript(output, prompt=prompt))

    def _command(self, *, req: GenerateRequest, audio_path: str, prompt: str) -> list[str]:
        cmd = [
            self.binary,
            "--single-turn",
            "--model",
            self.model_path,
            "--mmproj",
            self.mmproj_path,
            "--audio",
            audio_path,
            "--prompt",
            prompt,
            "--predict",
            str(req.max_new_tokens),
            "--temperature",
            str(req.temperature),
            "--no-warmup",
            "--log-disable",
            "--no-display-prompt",
            "--no-show-timings",
            "--simple-io",
        ]
        cmd.extend(self.extra_args)
        return cmd


def _write_wav(path: str, wav: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    audio = np.asarray(wav, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.T
    sf.write(path, audio, sample_rate)


def _strip_audio_token(prompt: str) -> str:
    return prompt.replace("<|audio|>", "", 1).strip()


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _clean_line(line: str) -> str:
    line = _ANSI_RE.sub("", line)
    while "\b" in line:
        line = re.sub(r".?\b", "", line, count=1)
    return line.strip()


def _extract_transcript(output: str, *, prompt: str) -> str:
    lines = [_clean_line(line) for line in output.replace("\r", "\n").splitlines()]

    transcript: list[str] = []
    collecting = False
    prompt_markers = {f"> {prompt}", prompt}
    for line in lines:
        if not collecting:
            if line in prompt_markers:
                collecting = True
            continue

        if line == "Exiting..." or line.startswith("[ Prompt:"):
            break
        if line:
            transcript.append(line)

    if transcript:
        return "\n".join(transcript).strip()

    # Fallback for llama.cpp builds that eventually honor --no-display-prompt in single-turn mode.
    boilerplate_prefixes = (
        "Loading model",
        "build",
        "model",
        "modalities",
        "available commands",
        "/exit",
        "/regen",
        "/clear",
        "/read",
        "/glob",
        "/audio",
        "Loaded media from",
        ">",
    )
    kept = [
        line
        for line in lines
        if line
        and line != "Exiting..."
        and not line.startswith("[ Prompt:")
        and not any(line.startswith(prefix) for prefix in boilerplate_prefixes)
        and not set(line) <= {"▄", "█", "▀", " "}
    ]
    return "\n".join(kept).strip()


def _last_nonempty_line(output: str) -> str:
    for line in reversed(output.splitlines()):
        cleaned = _clean_line(line)
        if cleaned:
            return cleaned
    return ""


__all__ = ["LlamaCppBackend"]
