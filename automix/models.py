"""Domain models for AutoMix AI."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SpectralProfile:
    """Frequency band energy ratios for blend quality prediction."""

    bass_ratio: float  # 0-1, energy ratio for 0-250Hz
    mid_ratio: float  # 0-1, energy ratio for 250-4kHz
    treble_ratio: float  # 0-1, energy ratio for 4kHz+


@dataclass
class Section:
    """A detected section of a track."""

    type: str  # intro, buildup, drop, breakdown, outro
    start: float  # seconds
    end: float  # seconds
    energy: float  # 0-1 normalized energy level


@dataclass
class EnergyProfile:
    """Energy analysis results for a track."""

    overall_energy: float  # 1-10 scale
    energy_at_boundaries: List[Tuple[float, float]]  # [(time_sec, energy_normalized), ...]
    intro_end: float  # seconds where intro ends
    outro_start: float  # seconds where outro starts
    sections: Optional[List[Section]] = None
    energy_shape: str = "flat"  # building, peaking, flat, declining


@dataclass
class MixCandidate:
    """A candidate mix point with scoring breakdown."""

    time: float  # seconds
    score: float  # total score
    section_type: str  # section type at this point
    energy_gradient: float  # energy change from previous boundary
    phrase_aligned: bool  # on a 16 or 32 bar boundary
    spectral_profile: Optional[SpectralProfile] = None  # local spectral character


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
    spectral_profile: Optional[SpectralProfile] = None
    mix_in_candidates: Optional[List[MixCandidate]] = None
    mix_out_candidates: Optional[List[MixCandidate]] = None
    version: int = 1  # cache invalidation version

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
    score: float = 0.0
    score_breakdown: Optional[Dict[str, float]] = None
    optimal_mix_out: Optional[float] = None
    optimal_mix_in: Optional[float] = None
