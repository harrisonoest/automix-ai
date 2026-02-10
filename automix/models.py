"""Domain models for AutoMix AI."""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class EnergyProfile:
    """Energy analysis results for a track."""

    overall_energy: float  # 1-10 scale
    energy_at_boundaries: List[Tuple[float, float]]  # [(time_sec, energy_normalized), ...]
    intro_end: float  # seconds where intro ends
    outro_start: float  # seconds where outro starts


@dataclass
class AnalysisResult:
    """Result of audio analysis for a track."""

    bpm: Optional[float]
    key: str
    mix_in_point: float
    mix_out_point: float
    bpm_confidence: float
    key_confidence: float
    energy_profile: Optional[EnergyProfile] = None

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
class TransitionRecommendation:
    """Recommended transition between two tracks."""

    mix_duration_bars: int  # 8, 16, or 32
    transition_type: str  # "blend", "cut", or "echo"
    eq_strategy: str  # "bass_swap", "filter_sweep", or "simple_fade"


@dataclass
class CompatibilityResult:
    """Result of compatibility check between two tracks."""

    compatible: bool
    tempo_diff: float
    key_reason: str
    transition: Optional[TransitionRecommendation] = None
