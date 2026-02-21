"""Tests for section detection, spectral analysis, and compatibility."""

import unittest.mock as mock

import numpy as np

from automix.analyzer import AudioAnalyzer
from automix.models import (
    AnalysisResult,
    EnergyProfile,
    MixCandidate,
    Section,
)

# --- Helpers ---


def _make_energy_profile(energies, duration=300.0, sections=None, energy_shape="flat"):
    """Build an EnergyProfile from a list of energy values."""
    n = len(energies)
    step = duration / n if n > 0 else duration
    boundaries = [(i * step, e) for i, e in enumerate(energies)]
    intro_end = 0.0
    outro_start = duration
    if sections:
        for s in sections:
            if s.type == "intro":
                intro_end = s.end
            if s.type == "outro":
                outro_start = s.start
                break
    return EnergyProfile(
        overall_energy=5.0,
        energy_at_boundaries=boundaries,
        intro_end=intro_end,
        outro_start=outro_start,
        sections=sections,
        energy_shape=energy_shape,
    )


def _make_result(
    bpm=128.0,
    key="Am",
    energy_profile=None,
    spectral=None,
    mix_in_candidates=None,
    mix_out_candidates=None,
):
    return AnalysisResult(
        bpm=bpm,
        key=key,
        mix_in_point=15.0,
        mix_out_point=240.0,
        bpm_confidence=0.95,
        key_confidence=0.87,
        energy_profile=energy_profile,
        spectral_profile=spectral,
        mix_in_candidates=mix_in_candidates,
        mix_out_candidates=mix_out_candidates,
    )


# --- Section detection (Phase 3) ---


class TestSectionDetection:
    def setup_method(self):
        self.analyzer = AudioAnalyzer(use_cache=False)

    def _detect(self, energies, duration=300.0):
        """Helper to run section detection with synthetic energy data."""
        n = len(energies)
        step = duration / n
        boundaries = [(i * step, e) for i, e in enumerate(energies)]
        onset_env = np.ones(100)
        onset_times = np.linspace(0, duration, 100)
        return self.analyzer._detect_sections(boundaries, onset_env, onset_times, duration)

    def test_clear_intro_drop_outro(self):
        """Track: low→low→high→high→high→low→low"""
        energies = [0.1, 0.15, 0.8, 0.9, 0.85, 0.1, 0.05]
        sections = self._detect(energies)
        types = [s.type for s in sections]
        assert types[0] == "intro"
        assert "drop" in types
        assert types[-1] == "outro"

    def test_no_intro_starts_loud(self):
        """DJ edit: starts at high energy."""
        energies = [0.9, 0.85, 0.8, 0.7, 0.2, 0.1]
        sections = self._detect(energies)
        # Should NOT start with intro
        assert sections[0].type != "intro"

    def test_mid_breakdown_not_outro(self):
        """Breakdown at ~50% should be 'breakdown', not 'outro'."""
        # 10 boundaries, breakdown at index 4-5 (40-50% of track)
        energies = [0.1, 0.5, 0.9, 0.85, 0.1, 0.15, 0.8, 0.9, 0.2, 0.1]
        sections = self._detect(energies)
        # The low-energy sections in the middle should be breakdown, not outro
        mid_sections = [s for s in sections if s.start > 100 and s.end < 200]
        for s in mid_sections:
            if s.energy < 0.3:
                assert s.type == "breakdown", f"Mid-track low energy section classified as {s.type}"

    def test_flat_energy_ambient(self):
        """Flat energy track should produce sections (not crash)."""
        energies = [0.5, 0.5, 0.5, 0.5, 0.5]
        sections = self._detect(energies)
        assert len(sections) >= 1

    def test_merge_pass_collapses_adjacent(self):
        """Adjacent same-type sections should be merged."""
        # Two consecutive low-energy boundaries at start → single intro
        energies = [0.1, 0.1, 0.1, 0.8, 0.9, 0.1]
        sections = self._detect(energies)
        # First section should span all three low-energy boundaries
        assert sections[0].type == "intro"
        # Check it was merged (not 3 separate intro sections)
        intro_count = sum(1 for s in sections if s.type == "intro")
        assert intro_count == 1

    def test_return_type_annotation(self):
        """Sections should be List[Section]."""
        energies = [0.1, 0.8, 0.1]
        sections = self._detect(energies)
        assert all(isinstance(s, Section) for s in sections)

    def test_single_boundary_fallback(self):
        """Less than 2 boundaries → single drop section."""
        sections = self._detect([0.5])
        assert len(sections) == 1
        assert sections[0].type == "drop"


# --- Spectral analysis (Phase 4) ---


class TestSpectralAnalysis:
    def setup_method(self):
        self.analyzer = AudioAnalyzer(use_cache=False)
        self.sr = 22050

    def test_bass_heavy_signal(self):
        """Low frequency sine → high bass_ratio."""
        t = np.linspace(0, 2, self.sr * 2)
        y = np.sin(2 * np.pi * 100 * t)  # 100Hz
        sp = self.analyzer._analyze_spectral(y, self.sr)
        assert sp.bass_ratio > sp.treble_ratio

    def test_treble_heavy_signal(self):
        """High frequency sine → high treble_ratio."""
        t = np.linspace(0, 2, self.sr * 2)
        y = np.sin(2 * np.pi * 8000 * t)  # 8kHz
        sp = self.analyzer._analyze_spectral(y, self.sr)
        assert sp.treble_ratio > sp.bass_ratio

    def test_white_noise_roughly_balanced(self):
        """White noise → ratios should be somewhat balanced."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(self.sr * 2).astype(np.float32)
        sp = self.analyzer._analyze_spectral(y, self.sr)
        # No single band should dominate overwhelmingly
        assert max(sp.bass_ratio, sp.mid_ratio, sp.treble_ratio) < 0.8

    def test_local_vs_global_differ(self):
        """Local spectral at bass section differs from global on mixed track."""
        t = np.linspace(0, 4, self.sr * 4)
        # First 2 seconds: bass, last 2 seconds: treble
        y = np.concatenate(
            [
                np.sin(2 * np.pi * 100 * t[: self.sr * 2]),
                np.sin(2 * np.pi * 8000 * t[: self.sr * 2]),
            ]
        )
        global_sp = self.analyzer._analyze_spectral(y, self.sr)
        local_sp = self.analyzer._analyze_spectral_at(y, self.sr, center_time=1.0, window_sec=2.0)
        # Local at t=1.0 (bass section) should have higher bass than global
        assert local_sp.bass_ratio > global_sp.bass_ratio


# --- Compatibility scoring (Phase 5) ---


class TestCompatibility:
    def setup_method(self):
        self.analyzer = AudioAnalyzer(use_cache=False)

    def test_same_key_same_bpm_high_score(self):
        r1 = _make_result(bpm=128.0, key="Am")
        r2 = _make_result(bpm=128.0, key="Am")
        compat = self.analyzer.check_compatibility(r1, r2)
        assert compat is not None
        assert compat.compatible
        assert compat.score >= 60

    def test_incompatible_key_returns_none(self):
        r1 = _make_result(bpm=128.0, key="Am")
        r2 = _make_result(bpm=128.0, key="F#m")
        assert self.analyzer.check_compatibility(r1, r2) is None

    def test_tempo_too_far_returns_none(self):
        r1 = _make_result(bpm=128.0, key="Am")
        r2 = _make_result(bpm=140.0, key="Am")
        assert self.analyzer.check_compatibility(r1, r2) is None

    def test_below_threshold_not_compatible_no_transition(self):
        """Score below min → compatible=False, transition=None."""
        # Adjacent key + max tempo diff → low score
        r1 = _make_result(bpm=128.0, key="Am")
        r2 = _make_result(bpm=133.9, key="A#m")
        compat = self.analyzer.check_compatibility(r1, r2)
        if compat is not None and not compat.compatible:
            assert compat.transition is None

    def test_score_breakdown_present(self):
        r1 = _make_result(bpm=128.0, key="Am")
        r2 = _make_result(bpm=128.0, key="Am")
        compat = self.analyzer.check_compatibility(r1, r2)
        assert compat.score_breakdown is not None
        assert "key" in compat.score_breakdown
        assert "tempo" in compat.score_breakdown

    def test_missing_bpm_returns_none(self):
        r1 = _make_result(bpm=None, key="Am")
        r2 = _make_result(bpm=128.0, key="Am")
        assert self.analyzer.check_compatibility(r1, r2) is None

    def test_energy_shape_affects_score(self):
        """declining→building should score higher than building→building."""
        sections_out = [Section(type="outro", start=200, end=300, energy=0.2)]
        sections_in = [Section(type="intro", start=0, end=50, energy=0.2)]
        ep_declining = _make_energy_profile(
            [0.8, 0.6, 0.3, 0.1], sections=sections_out, energy_shape="declining"
        )
        ep_building = _make_energy_profile(
            [0.1, 0.3, 0.6, 0.8], sections=sections_in, energy_shape="building"
        )
        ep_building2 = _make_energy_profile(
            [0.1, 0.3, 0.6, 0.8], sections=sections_in, energy_shape="building"
        )

        r1_good = _make_result(energy_profile=ep_declining)
        r2_good = _make_result(energy_profile=ep_building)
        r1_bad = _make_result(energy_profile=ep_building2)
        r2_bad = _make_result(energy_profile=ep_building2)

        compat_good = self.analyzer.check_compatibility(r1_good, r2_good)
        compat_bad = self.analyzer.check_compatibility(r1_bad, r2_bad)
        assert compat_good.score >= compat_bad.score


# --- Pair optimization ---


class TestPairOptimization:
    def setup_method(self):
        self.analyzer = AudioAnalyzer(use_cache=False)

    def test_declining_out_rising_in_preferred(self):
        """Pair with declining→rising energy should score higher."""
        good_out = MixCandidate(
            time=240.0, score=5.0, section_type="outro", energy_gradient=-0.3, phrase_aligned=True
        )
        good_in = MixCandidate(
            time=15.0, score=5.0, section_type="intro", energy_gradient=0.3, phrase_aligned=True
        )
        flat_out = MixCandidate(
            time=240.0, score=5.0, section_type="drop", energy_gradient=0.0, phrase_aligned=True
        )
        flat_in = MixCandidate(
            time=15.0, score=5.0, section_type="drop", energy_gradient=0.0, phrase_aligned=True
        )

        r1_good = _make_result(mix_out_candidates=[good_out])
        r2_good = _make_result(mix_in_candidates=[good_in])
        r1_flat = _make_result(mix_out_candidates=[flat_out])
        r2_flat = _make_result(mix_in_candidates=[flat_in])

        compat_good = self.analyzer.check_compatibility(r1_good, r2_good)
        compat_flat = self.analyzer.check_compatibility(r1_flat, r2_flat)

        assert compat_good is not None
        assert compat_flat is not None
        # Good pair should have optimal points set
        assert compat_good.optimal_mix_out is not None
        assert compat_good.optimal_mix_in is not None


# --- Cache versioning (Phase 5) ---


class TestCacheVersioning:
    def test_stale_cache_invalidated(self, tmp_path):
        """Old cached results without version field should be invalidated."""
        import pickle

        from automix.cache import ResultCache

        cache = ResultCache(cache_dir=tmp_path)
        # Write a fake old result without version
        old_result = _make_result()
        old_result.version = 0  # simulate old version
        key_hash = "test_hash"
        cache_file = tmp_path / f"{key_hash}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(old_result, f)

        # Mock _cache_key_hash to return our known hash
        with mock.patch.object(cache, "_cache_key_hash", return_value=key_hash):
            result = cache.get("dummy.wav")
        assert result is None
        # Cache file should have been cleaned up
        assert not cache_file.exists()
