# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ._models import ModelSpec, language_label
from .errors import InvalidArgumentError

AUDIO_TOKEN = "<|audio|>"
DEFAULT_ASR_INSTRUCTION = "transcribe the speech with proper punctuation and capitalization."
KEYWORD_BIAS_ASR_INSTRUCTION = "transcribe the speech to text."
# Reproduces the plus model's training-time system prompt verbatim. The dates are
# fixed on purpose to match what the model saw during training — they are NOT the
# current date and must not be templated to "today", or output drifts from the
# trained distribution.
PLUS_SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\n"
    "Today's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)
PLUS_ASR_INSTRUCTION = "can you transcribe the speech into a written format?"
PLUS_SPEAKER_ATTRIBUTION_INSTRUCTION = (
    "Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: "
    "and [Speaker 2]: tags before speaker turns."
)
PLUS_WORD_TIMESTAMPS_INSTRUCTION = (
    "Timestamps: Transcribe the speech. After each word, add a timestamp tag showing "
    "the end time in centiseconds, e.g. hello [T:45] world [T:82]"
)
PROMPT_MODES = frozenset({"default", "speaker_attributed", "word_timestamps"})


@dataclass(frozen=True)
class PromptParts:
    prompt: str
    instruction: str


def build_prompt(
    tokenizer_or_processor: Any,
    *,
    task: str,
    language: str | None,
    target_language: str | None,
    prompt: str | None,
    keyword_biases: Iterable[str] | str | None = None,
    model_spec: ModelSpec | None = None,
    prompt_mode: str = "default",
    prefix_text: str | None = None,
) -> PromptParts:
    """Build the instruction and the full templated prompt for one request.

    Resolves the instruction text — an explicit ``prompt`` (with any audio token
    stripped) overrides the task/language/``prompt_mode`` default — then appends
    keyword biases and the ``<|audio|>`` token. When the tokenizer exposes
    ``apply_chat_template`` (the plus models), the instruction is rendered
    through the chat template, including the plus system prompt and any
    ``prefix_text`` decoding hook; otherwise the raw content is used. Returns
    :class:`PromptParts` with both the templated ``prompt`` and the bare
    ``instruction`` (backends that take an instruction directly use the latter).
    """
    keywords = normalize_keyword_biases(keyword_biases)
    instruction = (
        _strip_audio_token(prompt)
        if prompt is not None
        else default_instruction(
            task,
            language,
            target_language,
            keyword_biased=bool(keywords),
            model_spec=model_spec,
            prompt_mode=prompt_mode,
        )
    )
    instruction = add_keyword_bias_instruction(instruction, keywords)
    content = add_audio_token(instruction, model_spec=model_spec)
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)

    if hasattr(tokenizer, "apply_chat_template"):
        chat = chat_messages(content, model_spec=model_spec)
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if prefix_text is not None:
            template_kwargs["prefix_text"] = prefix_text
        templated = tokenizer.apply_chat_template(chat, **template_kwargs)
    else:
        templated = content
    return PromptParts(prompt=templated, instruction=instruction)


def chat_messages(content: str, *, model_spec: ModelSpec | None = None) -> list[dict[str, str]]:
    if _prompt_profile(model_spec) == "plus":
        return [
            {"role": "system", "content": PLUS_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
    return [{"role": "user", "content": content}]


def add_audio_token(instruction: str, *, model_spec: ModelSpec | None = None) -> str:
    separator = " " if _prompt_profile(model_spec) == "plus" else ""
    return f"{AUDIO_TOKEN}{separator}{instruction}"


def _strip_audio_token(prompt: str) -> str:
    if prompt.startswith(AUDIO_TOKEN):
        return prompt[len(AUDIO_TOKEN) :].strip()
    return prompt


def normalize_keyword_biases(keyword_biases: Iterable[str] | str | None) -> tuple[str, ...]:
    if keyword_biases is None:
        return ()
    if isinstance(keyword_biases, str):
        raw_keywords = (keyword_biases,)
    else:
        try:
            raw_keywords = tuple(keyword_biases)
        except TypeError as exc:
            raise InvalidArgumentError(
                "keyword_biases must be a string or an iterable of strings"
            ) from exc

    normalized: list[str] = []
    seen: set[str] = set()
    for keyword in raw_keywords:
        if not isinstance(keyword, str):
            raise InvalidArgumentError("keyword_biases entries must be strings")
        cleaned = " ".join(keyword.split())
        if not cleaned:
            raise InvalidArgumentError("keyword_biases entries must be non-empty strings")
        if cleaned not in seen:
            normalized.append(cleaned)
            seen.add(cleaned)
    return tuple(normalized)


def add_keyword_bias_instruction(
    instruction: str,
    keyword_biases: Iterable[str] | str | None,
) -> str:
    keywords = normalize_keyword_biases(keyword_biases)
    if not keywords:
        return instruction
    suffix = f"Keywords: {_format_keyword_biases(keywords)}"
    base = instruction.rstrip()
    if not base:
        return suffix
    if base.endswith((".", "!", "?")):
        return f"{base} {suffix}"
    return f"{base}. {suffix}"


def _format_keyword_biases(keyword_biases: tuple[str, ...]) -> str:
    return ", ".join(keyword_biases)


def default_instruction(
    task: str,
    language: str | None,
    target_language: str | None,
    *,
    keyword_biased: bool = False,
    model_spec: ModelSpec | None = None,
    prompt_mode: str = "default",
) -> str:
    if task == "transcribe":
        if _prompt_profile(model_spec) == "plus":
            if prompt_mode == "speaker_attributed":
                return PLUS_SPEAKER_ATTRIBUTION_INSTRUCTION
            if prompt_mode == "word_timestamps":
                return PLUS_WORD_TIMESTAMPS_INSTRUCTION
            if language is None:
                return PLUS_ASR_INSTRUCTION
            return (
                f"can you transcribe the {language_label(language)} speech "
                "into a written format?"
            )
        if keyword_biased:
            return KEYWORD_BIAS_ASR_INSTRUCTION
        if language is None:
            return DEFAULT_ASR_INSTRUCTION
        return (
            f"transcribe the {language_label(language)} speech with proper punctuation "
            "and capitalization."
        )

    if keyword_biased:
        return f"translate the speech to {language_label(target_language)}."
    return (
        f"translate the speech to {language_label(target_language)} "
        "with proper punctuation and capitalization."
    )


def _prompt_profile(model_spec: ModelSpec | None) -> str:
    return getattr(model_spec, "prompt_profile", "base")
