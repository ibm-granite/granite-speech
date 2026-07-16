# Copyright IBM Corp. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

class GraniteSpeechError(Exception):
    """Base class for exceptions raised by granite-speech."""


class ModelLoadError(GraniteSpeechError):
    """Raised when model download, loading, or device placement fails."""


class TransformersVersionError(ModelLoadError):
    """Raised when the installed transformers version is unsupported."""


class AudioDecodeError(GraniteSpeechError):
    """Raised when audio input cannot be decoded or normalized."""


class InvalidArgumentError(ValueError, GraniteSpeechError):
    """Raised when a public API argument is invalid."""


class TranscriptionError(GraniteSpeechError):
    """Raised when transcription cannot produce usable output."""
