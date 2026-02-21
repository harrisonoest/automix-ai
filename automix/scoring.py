"""Pure scoring functions for mix point selection and compatibility."""

from typing import Optional, Tuple

import numpy as np

from .models import SpectralProfile

# --- Mix candidate scoring (Phase 6) ---


def score_section_match(
    section_type: str, is_mix_in: bool, section_weight: float, buildup_weight: float
) -> float:
    """Score a candidate based on section type suitability."""
    if is_mix_in and section_type == "intro":
        return section_weight
    if not is_mix_in and section_type == "outro":
        return section_weight
    if section_type in ("buildup", "breakdown"):
        return buildup_weight
    return 0.0


def score_energy_gradient(gradient: float, is_mix_in: bool, weight: float) -> float:
    """Score based on energy direction (rising for mix-in, falling for mix-out)."""
    if is_mix_in and gradient > 0:
        return weight
    if not is_mix_in and gradient < 0:
        return weight
    return 0.0


def is_near_boundary(t: float, boundary_times: np.ndarray, tolerance: float) -> bool:
    """Check if t is within tolerance of any boundary time."""
    if len(boundary_times) == 0:
        return False
    return float(np.min(np.abs(boundary_times - t))) < tolerance


def score_phrase_alignment(
    t: float,
    bar32_times: np.ndarray,
    bar16_times: np.ndarray,
    tolerance: float,
    weight_32: float,
    weight_16: float,
) -> Tuple[float, bool]:
    """Score phrase alignment. Returns (score, is_aligned)."""
    if is_near_boundary(t, bar32_times, tolerance):
        return weight_32, True
    if is_near_boundary(t, bar16_times, tolerance):
        return weight_16, True
    return 0.0, False


def score_proximity(t: float, reference_time: float, threshold: float, weight: float) -> float:
    """Score proximity to a reference point (intro_end or outro_start)."""
    if abs(t - reference_time) < threshold:
        return weight
    return 0.0


def score_energy_penalty(
    energy: float, max_energy: float, is_mix_in: bool, low_penalty: float, peak_penalty: float
) -> float:
    """Penalize low-energy mix-ins or peak-energy mix-outs."""
    if is_mix_in and energy < 0.1:
        return low_penalty
    if not is_mix_in and max_energy > 0 and energy > max_energy * 0.8:
        return peak_penalty
    return 0.0


# --- Compatibility scoring (Phase 6) ---

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def parse_key(k: str) -> Tuple[int, str]:
    """Parse key string into (index, mode)."""
    if k.endswith("m"):
        return KEY_NAMES.index(k[:-1]), "minor"
    return KEY_NAMES.index(k), "major"


def score_key_compatibility(key1: str, key2: str, max_weight: float) -> Optional[Tuple[float, str]]:
    """Score key compatibility. Returns (score, reason) or None if incompatible."""
    idx1, mode1 = parse_key(key1)
    idx2, mode2 = parse_key(key2)
    semitone_diff = (idx2 - idx1) % 12

    scores = {
        "same key": max_weight,
        "relative major": max_weight * 0.875,
        "relative minor": max_weight * 0.875,
        "perfect fifth": max_weight * 0.75,
        "perfect fourth": max_weight * 0.625,
        "adjacent key": max_weight * 0.375,
    }

    if key1 == key2:
        reason = "same key"
    elif semitone_diff == 3 and mode1 == "minor" and mode2 == "major":
        reason = "relative major"
    elif semitone_diff == 9 and mode1 == "major" and mode2 == "minor":
        reason = "relative minor"
    elif semitone_diff == 7:
        reason = "perfect fifth"
    elif semitone_diff == 5:
        reason = "perfect fourth"
    elif semitone_diff in (1, 11):
        reason = "adjacent key"
    else:
        return None

    return round(scores[reason], 1), reason


def score_tempo_proximity(tempo_diff: float, tolerance: float, max_weight: float) -> float:
    """Score tempo proximity (linear falloff)."""
    if tolerance <= 0:
        return max_weight
    return round(max_weight * (1 - abs(tempo_diff) / tolerance), 1)


# Energy flow truth table: (out_level, in_level) -> base score out of 15
_ENERGY_FLOW = {
    ("low", "low"): 12,
    ("low", "mid"): 10,
    ("low", "high"): 6,
    ("mid", "low"): 8,
    ("mid", "mid"): 8,
    ("mid", "high"): 7,
    ("high", "low"): 6,
    ("high", "mid"): 7,
    ("high", "high"): 5,
}

_SHAPE_BONUS = {
    ("declining", "building"): 3,
    ("peaking", "building"): 2,
    ("declining", "flat"): 1,
}

_SHAPE_PENALTY = {
    ("building", "building"): -2,
    ("peaking", "peaking"): -1,
}


def _energy_level(e: float) -> str:
    if e <= 0.4:
        return "low"
    if e > 0.7:
        return "high"
    return "mid"


def score_energy_flow(
    out_energy: float,
    in_energy: float,
    out_gradient: Optional[float],
    shape1: str,
    shape2: str,
    max_weight: float,
) -> float:
    """Score energy flow between two tracks."""
    base = _ENERGY_FLOW.get((_energy_level(out_energy), _energy_level(in_energy)), 5)

    # Bonus for declining→rising pattern
    if out_gradient is not None and out_gradient < 0 and in_energy < 0.5:
        base = min(max_weight, base + 5)

    base = min(max_weight, base + _SHAPE_BONUS.get((shape1, shape2), 0))
    base = max(0, base + _SHAPE_PENALTY.get((shape1, shape2), 0))
    return float(base)


def score_section_compatibility(
    out_section: Optional[str], in_section: Optional[str], max_weight: float
) -> float:
    """Score section pairing between outgoing and incoming tracks."""
    good_pairs = {
        ("outro", "intro"): 1.0,
        ("breakdown", "buildup"): 0.8,
        ("outro", "buildup"): 0.67,
        ("breakdown", "intro"): 0.67,
        ("drop", "intro"): 0.33,
    }
    ratio = good_pairs.get((out_section, in_section), 0.2)
    return round(max_weight * ratio, 1)


def score_spectral_blend(sp1: Optional[SpectralProfile], sp2: Optional[SpectralProfile]) -> float:
    """Score how well two spectral profiles blend (0-5 bonus)."""
    if not sp1 or not sp2:
        return 0.0
    # Complementary spectra blend better than identical ones
    bass_diff = abs(sp1.bass_ratio - sp2.bass_ratio)
    treble_diff = abs(sp1.treble_ratio - sp2.treble_ratio)
    # Reward moderate difference (complementary), penalize identical heavy profiles
    return round(min(5.0, (bass_diff + treble_diff) * 5), 1)
