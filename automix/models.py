"""Domain models for AutoMix AI."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AnalysisResult:
    """Result of audio analysis for a track."""

    bpm: Optional[float]
    key: str
    mix_in_point: float
    mix_out_point: float
    bpm_confidence: float
    key_confidence: float

    @property
    def bpm_str(self) -> str:
        """Human-readable BPM string."""
        return f"{self.bpm:.1f}" if self.bpm else "Unknown"


@dataclass
class Track:
    """Represents an audio track with its analysis."""

    path: str
    analysis: AnalysisResult
    title: Optional[str] = None
    artist: Optional[str] = None
    url: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Get display name for the track."""
        if self.artist and self.title:
            return f"{self.artist} - {self.title}"
        return self.path


@dataclass
class CompatibilityResult:
    """Result of compatibility check between two tracks."""

    compatible: bool
    tempo_diff: float
    key_reason: str
