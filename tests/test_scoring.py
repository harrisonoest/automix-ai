"""Tests for scoring pure functions."""

import numpy as np

from automix.models import SpectralProfile
from automix.scoring import (
    is_near_boundary,
    score_energy_flow,
    score_energy_gradient,
    score_energy_penalty,
    score_key_compatibility,
    score_phrase_alignment,
    score_section_compatibility,
    score_section_match,
    score_spectral_blend,
    score_tempo_proximity,
)

# --- Phrase alignment (Phase 2 fix) ---


class TestPhraseAlignment:
    def test_exact_match_32bar(self):
        boundaries = np.array([0.0, 16.0, 32.0, 48.0])
        assert is_near_boundary(32.0, boundaries, 0.05)

    def test_near_match_within_tolerance(self):
        boundaries = np.array([0.0, 16.0, 32.0])
        assert is_near_boundary(16.03, boundaries, 0.05)

    def test_no_match_outside_tolerance(self):
        boundaries = np.array([0.0, 16.0, 32.0])
        assert not is_near_boundary(16.1, boundaries, 0.05)

    def test_floating_point_drift(self):
        """Beat times with slight float drift still match."""
        boundaries = np.array([15.999999, 31.999998])
        assert is_near_boundary(16.0, boundaries, 0.05)

    def test_empty_boundaries(self):
        assert not is_near_boundary(10.0, np.array([]), 0.05)

    def test_score_32bar_higher_than_16bar(self):
        bar32 = np.array([0.0, 32.0, 64.0])
        bar16 = np.array([0.0, 16.0, 32.0, 48.0, 64.0])
        score_32, aligned_32 = score_phrase_alignment(32.0, bar32, bar16, 0.05, 2.0, 1.0)
        score_16, aligned_16 = score_phrase_alignment(16.0, bar32, bar16, 0.05, 2.0, 1.0)
        # 16.0 is only in bar16, not bar32 → should get 1.0
        # 32.0 is in bar32 → should get 2.0
        assert score_32 == 2.0
        assert aligned_32
        # 16.0 is in bar32? No (bar32 = [0, 32, 64]). It's in bar16 → 1.0
        assert score_16 == 1.0
        assert aligned_16

    def test_off_grid_gets_zero(self):
        bar32 = np.array([0.0, 32.0])
        bar16 = np.array([0.0, 16.0, 32.0])
        score, aligned = score_phrase_alignment(10.0, bar32, bar16, 0.05, 2.0, 1.0)
        assert score == 0.0
        assert not aligned


# --- Section detection scoring ---


class TestSectionScoring:
    def test_intro_for_mix_in(self):
        assert score_section_match("intro", True, 3.0, 2.0) == 3.0

    def test_outro_for_mix_out(self):
        assert score_section_match("outro", False, 3.0, 2.0) == 3.0

    def test_buildup_gets_secondary_weight(self):
        assert score_section_match("buildup", True, 3.0, 2.0) == 2.0
        assert score_section_match("breakdown", False, 3.0, 2.0) == 2.0

    def test_drop_for_mix_in_gets_zero(self):
        assert score_section_match("drop", True, 3.0, 2.0) == 0.0

    def test_intro_for_mix_out_gets_zero(self):
        assert score_section_match("intro", False, 3.0, 2.0) == 0.0


# --- Energy gradient scoring ---


class TestEnergyGradient:
    def test_rising_for_mix_in(self):
        assert score_energy_gradient(0.3, True, 2.0) == 2.0

    def test_falling_for_mix_out(self):
        assert score_energy_gradient(-0.3, False, 2.0) == 2.0

    def test_falling_for_mix_in_zero(self):
        assert score_energy_gradient(-0.3, True, 2.0) == 0.0

    def test_rising_for_mix_out_zero(self):
        assert score_energy_gradient(0.3, False, 2.0) == 0.0


# --- Energy penalty ---


class TestEnergyPenalty:
    def test_low_energy_mix_in_penalized(self):
        assert score_energy_penalty(0.05, 1.0, True, -1.0, -1.0) == -1.0

    def test_peak_energy_mix_out_penalized(self):
        assert score_energy_penalty(0.9, 1.0, False, -1.0, -1.0) == -1.0

    def test_normal_energy_no_penalty(self):
        assert score_energy_penalty(0.5, 1.0, True, -1.0, -1.0) == 0.0
        assert score_energy_penalty(0.3, 1.0, False, -1.0, -1.0) == 0.0


# --- Key compatibility ---


class TestKeyCompatibility:
    def test_same_key_max_score(self):
        score, reason = score_key_compatibility("Am", "Am", 40.0)
        assert score == 40.0
        assert reason == "same key"

    def test_relative_major(self):
        score, reason = score_key_compatibility("Am", "C", 40.0)
        assert reason == "relative major"
        assert score == 35.0

    def test_relative_minor(self):
        score, reason = score_key_compatibility("C", "Am", 40.0)
        assert reason == "relative minor"
        assert score == 35.0

    def test_perfect_fifth(self):
        score, reason = score_key_compatibility("Am", "Em", 40.0)
        assert reason == "perfect fifth"
        assert score == 30.0

    def test_perfect_fourth(self):
        score, reason = score_key_compatibility("Am", "Dm", 40.0)
        assert reason == "perfect fourth"
        assert score == 25.0

    def test_adjacent_key(self):
        score, reason = score_key_compatibility("Am", "A#m", 40.0)
        assert reason == "adjacent key"
        assert score == 15.0

    def test_incompatible_returns_none(self):
        assert score_key_compatibility("Am", "F#m", 40.0) is None

    def test_same_key_same_bpm_near_100(self):
        """Same key + same BPM should produce a score near 100."""
        key_score, _ = score_key_compatibility("Am", "Am", 40.0)
        tempo_score = score_tempo_proximity(0.0, 6.0, 30.0)
        # With max energy (15) and max section (15) = 100
        assert key_score + tempo_score == 70.0

    def test_adjacent_key_max_tempo_diff_near_threshold(self):
        """Adjacent key + max tempo diff should be near the min threshold."""
        key_score, _ = score_key_compatibility("Am", "A#m", 40.0)
        tempo_score = score_tempo_proximity(5.9, 6.0, 30.0)
        # 15 + ~0.5 = ~15.5, well below 30 threshold without energy/section
        assert key_score + tempo_score < 20


# --- Tempo proximity ---


class TestTempoProximity:
    def test_same_tempo_max_score(self):
        assert score_tempo_proximity(0.0, 6.0, 30.0) == 30.0

    def test_max_diff_zero_score(self):
        assert score_tempo_proximity(6.0, 6.0, 30.0) == 0.0

    def test_half_diff_half_score(self):
        assert score_tempo_proximity(3.0, 6.0, 30.0) == 15.0


# --- Energy flow ---


class TestEnergyFlow:
    def test_both_low_smooth(self):
        score = score_energy_flow(0.3, 0.3, None, "flat", "flat", 15.0)
        assert score >= 10

    def test_both_high_lower_score(self):
        score = score_energy_flow(0.8, 0.8, None, "flat", "flat", 15.0)
        assert score <= 7

    def test_declining_to_building_bonus(self):
        base = score_energy_flow(0.3, 0.3, None, "flat", "flat", 15.0)
        with_shape = score_energy_flow(0.3, 0.3, None, "declining", "building", 15.0)
        assert with_shape > base

    def test_building_building_penalty(self):
        base = score_energy_flow(0.5, 0.5, None, "flat", "flat", 15.0)
        penalized = score_energy_flow(0.5, 0.5, None, "building", "building", 15.0)
        assert penalized < base

    def test_declining_gradient_bonus(self):
        without = score_energy_flow(0.3, 0.3, None, "flat", "flat", 15.0)
        with_grad = score_energy_flow(0.3, 0.3, -0.2, "flat", "flat", 15.0)
        assert with_grad >= without

    def test_both_high_no_gradient_bonus(self):
        """Both high + in_energy > 0.5 means gradient bonus should NOT apply."""
        score = score_energy_flow(0.8, 0.8, -0.3, "flat", "flat", 15.0)
        # in_energy=0.8 > 0.5, so gradient bonus doesn't apply
        assert score == 5


# --- Section compatibility ---


class TestSectionCompatibility:
    def test_outro_intro_max(self):
        score = score_section_compatibility("outro", "intro", 15.0)
        assert score == 15.0

    def test_breakdown_buildup_high(self):
        score = score_section_compatibility("breakdown", "buildup", 15.0)
        assert score == 12.0

    def test_drop_drop_low(self):
        score = score_section_compatibility("drop", "drop", 15.0)
        assert score == 3.0

    def test_declining_out_rising_in_higher(self):
        """outro→intro should score higher than drop→drop."""
        good = score_section_compatibility("outro", "intro", 15.0)
        bad = score_section_compatibility("drop", "drop", 15.0)
        assert good > bad


# --- Spectral blend ---


class TestSpectralBlend:
    def test_complementary_spectra_score_higher(self):
        bass_heavy = SpectralProfile(bass_ratio=0.7, mid_ratio=0.2, treble_ratio=0.1)
        treble_heavy = SpectralProfile(bass_ratio=0.1, mid_ratio=0.2, treble_ratio=0.7)
        similar = SpectralProfile(bass_ratio=0.65, mid_ratio=0.2, treble_ratio=0.15)
        assert score_spectral_blend(bass_heavy, treble_heavy) > score_spectral_blend(
            bass_heavy, similar
        )

    def test_none_profiles_zero(self):
        assert score_spectral_blend(None, None) == 0.0
        sp = SpectralProfile(bass_ratio=0.5, mid_ratio=0.3, treble_ratio=0.2)
        assert score_spectral_blend(sp, None) == 0.0
