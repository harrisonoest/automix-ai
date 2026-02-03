"""Custom exceptions for AutoMix AI."""


class AudioLoadError(Exception):
    """Raised when an audio file cannot be loaded or decoded."""

    pass


class AudioTooShortError(Exception):
    """Raised when an audio file is too short for reliable analysis."""

    pass
