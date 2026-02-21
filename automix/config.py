"""Configuration for AutoMix AI analysis parameters."""

from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """Configuration for audio analysis."""

    # Minimum audio duration for analysis (seconds)
    min_duration: float = 10.0

    # Mix point offsets (in bars)
    mix_in_bars: int = 16
    mix_out_bars: int = 32

    # Compatibility thresholds
    tempo_tolerance: float = 6.0

    # Section detection
    section_low_energy_ratio: float = 0.3
    section_high_energy_ratio: float = 0.7
    section_gradient_threshold: float = 0.1
    outro_min_position: float = 0.75
    intro_min_position: float = 0.0

    # Energy shape classification
    energy_shape_threshold: float = 0.15

    # Mix candidate scoring weights
    mix_score_section_match: float = 3.0
    mix_score_energy_gradient: float = 2.0
    mix_score_phrase_32bar: float = 2.0
    mix_score_phrase_16bar: float = 1.0
    mix_score_proximity: float = 1.0
    mix_penalty_low_energy: float = -1.0
    mix_penalty_peak_energy: float = -1.0

    # Compatibility scoring weights (should sum to 100)
    compat_key_weight: float = 40.0
    compat_tempo_weight: float = 30.0
    compat_energy_weight: float = 15.0
    compat_section_weight: float = 15.0
    compat_min_score: float = 30.0

    # Spectral EQ thresholds
    bass_heavy_threshold: float = 0.5
    treble_heavy_threshold: float = 0.3

    # Phrase alignment tolerance (seconds)
    phrase_alignment_tolerance: float = 0.05

    # Pair optimization
    pair_candidate_limit: int = 5


# Default configuration instance
DEFAULT_CONFIG = AnalysisConfig()
