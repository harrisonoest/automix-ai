"""Configuration for AutoMix AI analysis parameters."""

from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """Configuration for audio analysis."""

    # Minimum audio duration for analysis (seconds)
    min_duration: float = 10.0

    # Mix point offsets
    mix_in_offset: float = 5.0  # Minimum seconds from start
    mix_out_offset: float = 10.0  # Minimum seconds from end

    # Compatibility thresholds
    tempo_tolerance: float = 6.0  # Maximum BPM difference for compatibility


# Default configuration instance
DEFAULT_CONFIG = AnalysisConfig()
