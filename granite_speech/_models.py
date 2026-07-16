# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .errors import InvalidArgumentError

DEFAULT_MODEL = "granite-speech-4.1-2b"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    llama_cpp_repo_id: str | None
    llama_cpp_model_file_template: str | None
    llama_cpp_mmproj_file: str | None
    source_languages: frozenset[str]
    translation_pairs: frozenset[tuple[str, str]]
    supports_asr: bool
    supports_translation: bool
    supports_word_timing_output: bool
    supports_speaker_attribution_output: bool
    prompt_profile: str = "base"


LANGUAGE_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "fr": "fr",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "de": "de",
    "deu": "de",
    "ger": "de",
    "german": "de",
    "es": "es",
    "spa": "es",
    "spanish": "es",
    "pt": "pt",
    "por": "pt",
    "portuguese": "pt",
    "ja": "ja",
    "jpn": "ja",
    "japanese": "ja",
    "it": "it",
    "ita": "it",
    "italian": "it",
    "zh": "zh",
    "zho": "zh",
    "chi": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "mandarin chinese": "zh",
}

LANGUAGE_LABELS = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ja": "Japanese",
    "it": "Italian",
    "zh": "Mandarin",
}

_BASE_SOURCE_LANGUAGES = frozenset({"en", "fr", "de", "es", "pt", "ja"})

# The base model card advertises speech translation to and from English for the ASR languages,
# plus English-to-Italian and English-to-Mandarin prompts. Keep this registry conservative and
# explicit; prompt overrides do not bypass it.
_BASE_TRANSLATION_PAIRS = frozenset(
    {
        ("fr", "en"),
        ("de", "en"),
        ("es", "en"),
        ("pt", "en"),
        ("ja", "en"),
        ("en", "fr"),
        ("en", "de"),
        ("en", "es"),
        ("en", "pt"),
        ("en", "ja"),
        ("en", "it"),
        ("en", "zh"),
    }
)

MODELS = {
    "granite-speech-4.1-2b": ModelSpec(
        name="granite-speech-4.1-2b",
        repo_id="ibm-granite/granite-speech-4.1-2b",
        llama_cpp_repo_id="ibm-granite/granite-speech-4.1-2b-GGUF",
        llama_cpp_model_file_template="granite-speech-4.1-2b-{quant}.gguf",
        llama_cpp_mmproj_file="mmproj-model-f16.gguf",
        source_languages=_BASE_SOURCE_LANGUAGES,
        translation_pairs=_BASE_TRANSLATION_PAIRS,
        supports_asr=True,
        supports_translation=True,
        supports_word_timing_output=False,
        supports_speaker_attribution_output=False,
    ),
    "granite-speech-4.1-2b-plus": ModelSpec(
        name="granite-speech-4.1-2b-plus",
        repo_id="ibm-granite/granite-speech-4.1-2b-plus",
        llama_cpp_repo_id="ibm-granite/granite-speech-4.1-2b-plus-GGUF",
        llama_cpp_model_file_template="granite-speech-4.1-2b-plus-{quant}.gguf",
        llama_cpp_mmproj_file="mmproj-model-f16.gguf",
        source_languages=frozenset({"en", "fr", "de", "es", "pt"}),
        translation_pairs=frozenset(),
        supports_asr=True,
        supports_translation=False,
        supports_word_timing_output=True,
        supports_speaker_attribution_output=True,
        prompt_profile="plus",
    ),
}

_ALIASES = {
    **{spec.repo_id: name for name, spec in MODELS.items()},
    **{Path(spec.repo_id).name: name for name, spec in MODELS.items()},
    **{
        spec.llama_cpp_repo_id: name
        for name, spec in MODELS.items()
        if spec.llama_cpp_repo_id is not None
    },
    **{
        Path(spec.llama_cpp_repo_id).name: name
        for name, spec in MODELS.items()
        if spec.llama_cpp_repo_id is not None
    },
}


def available_models() -> list[str]:
    return list(MODELS)


def canonical_language(value: str | None) -> str | None:
    if value is None:
        return None
    return LANGUAGE_ALIASES.get(value.strip().lower())


def language_label(value: str | None) -> str:
    canonical = canonical_language(value)
    if canonical is None:
        return value or "the requested language"
    return LANGUAGE_LABELS[canonical]


def resolve_model_spec(name: str | Path) -> ModelSpec:
    raw_name = str(name)
    if raw_name in MODELS:
        return MODELS[raw_name]
    alias = _ALIASES.get(raw_name)
    if alias is not None:
        return MODELS[alias]

    path = Path(raw_name).expanduser()
    if path.exists():
        inferred = (
            MODELS["granite-speech-4.1-2b-plus"]
            if "plus" in path.name
            else MODELS["granite-speech-4.1-2b"]
        )
        has_llama_cpp_files = _looks_like_llama_cpp_path(path)
        # Same capabilities as the inferred base spec, but pointed at the local
        # path; drop the llama.cpp references when the path has no GGUF files.
        return replace(
            inferred,
            name=raw_name,
            repo_id=str(path),
            llama_cpp_repo_id=str(path) if has_llama_cpp_files else None,
            llama_cpp_model_file_template=(
                inferred.llama_cpp_model_file_template if has_llama_cpp_files else None
            ),
            llama_cpp_mmproj_file=inferred.llama_cpp_mmproj_file if has_llama_cpp_files else None,
        )

    choices = ", ".join(available_models())
    raise InvalidArgumentError(f"unknown model {raw_name!r}; available models: {choices}")


def _looks_like_llama_cpp_path(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() == ".gguf"
    return (path / "mmproj-model-f16.gguf").exists() and any(path.glob("*.gguf"))


def validate_source_language(spec: ModelSpec, language: str | None) -> None:
    if language is None:
        return
    canonical = canonical_language(language)
    if canonical is None or canonical not in spec.source_languages:
        supported = ", ".join(sorted(spec.source_languages))
        raise InvalidArgumentError(
            f"model {spec.name!r} does not support source language {language!r}; "
            f"supported source languages: {supported}"
        )


def validate_translation_pair(spec: ModelSpec, language: str, target_language: str) -> None:
    source = canonical_language(language)
    target = canonical_language(target_language)

    if source is None:
        raise InvalidArgumentError(f"unsupported source language for translation: {language!r}")
    if target is None:
        raise InvalidArgumentError(
            f"unsupported target language for translation: {target_language!r}"
        )
    if source == target:
        raise InvalidArgumentError(
            f"source language {language!r} and target language {target_language!r} resolve to "
            "the same language; pass a distinct target_language for task=\"translate\""
        )
    if not spec.supports_translation or (source, target) not in spec.translation_pairs:
        supported = format_translation_pairs(spec)
        raise InvalidArgumentError(
            f"model {spec.name!r} does not support translation pair {source}->{target}; "
            f"supported pairs: {supported}"
        )


def format_translation_pairs(spec: ModelSpec) -> str:
    if not spec.translation_pairs:
        return "none"
    return ", ".join(f"{src}->{tgt}" for src, tgt in sorted(spec.translation_pairs))
