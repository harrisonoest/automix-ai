"""Configuration for AutoMix AI analysis parameters."""

from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """Configuration for audio analysis."""

    # Minimum audio duration for analysis (seconds)
    min_duration: float = 10.0

    # Mix point offsets (in bars)
    mix_in_bars: int = 16  # Bars from start for mix-in point
    mix_out_bars: int = 32  # Bars before end for mix-out point

    # Compatibility thresholds
    tempo_tolerance: float = 6.0  # Maximum BPM difference for compatibility


# Default configuration instance
DEFAULT_CONFIG = AnalysisConfig()
