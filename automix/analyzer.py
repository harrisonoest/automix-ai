"""Audio analysis module for DJ mix transition detection."""

from typing import List, Optional

import librosa
import numpy as np

from .cache import CACHE_VERSION, ResultCache
from .config import DEFAULT_CONFIG, AnalysisConfig
from .exceptions import AudioLoadError, AudioTooShortError
from .logging_config import get_logger
from .models import (
    AnalysisResult,
    CompatibilityResult,
    EnergyProfile,
    MixCandidate,
    Section,
    SpectralProfile,
    TransitionRecommendation,
)
from .scoring import (
    score_energy_flow,
    score_energy_gradient,
    score_energy_penalty,
    score_key_compatibility,
    score_phrase_alignment,
    score_proximity,
    score_section_compatibility,
    score_section_match,
    score_spectral_blend,
    score_tempo_proximity,
)

logger = get_logger(__name__)


class AudioAnalyzer:
    """Analyzes audio files to detect BPM, key, and optimal mix points."""

    def __init__(self, config: AnalysisConfig = None, use_cache: bool = True):
        """Initialize analyzer with configuration.

        Args:
            config: Analysis configuration. Uses DEFAULT_CONFIG if not provided.
            use_cache: Enable result caching. Default True.
        """
        self.config = config or DEFAULT_CONFIG
        self.cache = ResultCache() if use_cache else None

    def analyze(self, file_path: str, cache_key: Optional[str] = None) -> AnalysisResult:
        """Analyzes an audio file for DJ mixing parameters.

        Args:
            file_path: Path to the audio file to analyze.
            cache_key: Optional cache key (e.g., URL for downloaded tracks).

        Returns:
            AnalysisResult: Analysis results model.

        Raises:
            AudioLoadError: If the audio file cannot be loaded.
            AudioTooShortError: If the audio file is too short for analysis.
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(file_path, cache_key)
            if cached:
                return cached

        logger.debug("Loading audio file: %s", file_path)
        try:
            y, sr = librosa.load(file_path)
        except Exception as e:
            logger.error("Failed to load audio file: %s", e)
            raise AudioLoadError("Unable to load audio file")

        duration = librosa.get_duration(y=y, sr=sr)
        logger.debug("Audio duration: %.1fs", duration)

        if duration < self.config.min_duration:
            logger.warning("Audio too short: %.1fs < %.1fs", duration, self.config.min_duration)
            raise AudioTooShortError("File too short for reliable analysis")

        # Beat detection
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = np.asarray(tempo).item() if np.asarray(tempo).ndim > 0 else float(tempo)
        bpm = float(tempo) if len(beats) > 0 else None

        # BPM confidence from beat strength consistency
        if len(beats) > 1:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            beat_frames = librosa.time_to_frames(librosa.frames_to_time(beats, sr=sr), sr=sr)
            beat_frames = beat_frames[beat_frames < len(onset_env)]
            if len(beat_frames) > 0:
                beat_strengths = onset_env[beat_frames]
                bpm_confidence = float(
                    1.0 - np.std(beat_strengths) / (np.mean(beat_strengths) + 1e-6)
                )
                bpm_confidence = np.clip(bpm_confidence, 0.0, 1.0)
            else:
                bpm_confidence = 0.0
        else:
            bpm_confidence = 0.0

        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Key detection with Krumhansl-Schmuckler algorithm
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )

        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)
        chroma_norm = chroma_mean / (np.sum(chroma_mean) + 1e-6)

        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        max_corr = -1
        best_key = "C"
        best_mode = "major"

        for i in range(12):
            major_rot = np.roll(major_profile, i)
            minor_rot = np.roll(minor_profile, i)

            major_corr = np.corrcoef(chroma_norm, major_rot)[0, 1]
            minor_corr = np.corrcoef(chroma_norm, minor_rot)[0, 1]

            if major_corr > max_corr:
                max_corr = major_corr
                best_key = keys[i]
                best_mode = "major"

            if minor_corr > max_corr:
                max_corr = minor_corr
                best_key = keys[i]
                best_mode = "minor"

        key = best_key if best_mode == "major" else best_key + "m"
        key_confidence = float(np.clip(max_corr, 0.0, 1.0)) if max_corr > 0 else 0.0

        # Calculate bar-based offsets (bars / (bpm / 60 / 4))
        bars_per_second = bpm / 60 / 4 if bpm else 0
        mix_in_offset = self.config.mix_in_bars / bars_per_second if bars_per_second > 0 else 0
        mix_out_offset = self.config.mix_out_bars / bars_per_second if bars_per_second > 0 else 0

        # Energy analysis (computed before mix points for energy-aware selection)
        energy_profile = self._analyze_energy(y, sr, beat_times, duration)

        # Spectral analysis (global)
        spectral_profile = self._analyze_spectral(y, sr)

        # Bar-based phrase boundaries as fallback
        bar_based_mix_in = self._find_phrase_boundary(
            beat_times, mix_in_offset, duration, search_forward=True
        )
        bar_based_mix_out = self._find_phrase_boundary(
            beat_times, duration - mix_out_offset, duration, search_forward=False
        )

        # Energy-aware mix point selection with bar-based fallback
        mix_in_point = bar_based_mix_in
        mix_out_point = bar_based_mix_out
        mix_in_candidates = None
        mix_out_candidates = None

        if energy_profile:
            in_candidates = self._select_energy_aware_mix_point(
                energy_profile, bar_based_mix_in, duration, True, mix_in_offset, beat_times, y, sr
            )
            if in_candidates:
                mix_in_candidates = in_candidates
                mix_in_point = in_candidates[0].time

            out_candidates = self._select_energy_aware_mix_point(
                energy_profile,
                bar_based_mix_out,
                duration,
                False,
                mix_out_offset,
                beat_times,
                y,
                sr,
            )
            if out_candidates:
                mix_out_candidates = out_candidates
                mix_out_point = out_candidates[0].time

        logger.info(
            "Analysis complete: BPM=%.1f (%.2f), Key=%s (%.2f)",
            bpm if bpm else 0,
            bpm_confidence,
            key,
            key_confidence,
        )

        result = AnalysisResult(
            bpm=bpm,
            key=key,
            mix_in_point=mix_in_point,
            mix_out_point=mix_out_point,
            bpm_confidence=bpm_confidence,
            key_confidence=key_confidence,
            energy_profile=energy_profile,
            spectral_profile=spectral_profile,
            mix_in_candidates=mix_in_candidates,
            mix_out_candidates=mix_out_candidates,
            version=CACHE_VERSION,
        )

        # Cache result
        if self.cache:
            self.cache.set(file_path, result, cache_key)

        return result

    def _find_phrase_boundary(
        self, beat_times: np.ndarray, target_time: float, duration: float, search_forward: bool
    ) -> float:
        """Find nearest phrase boundary (16 or 32 bar) to target time.

        Args:
            beat_times: Array of beat times in seconds.
            target_time: Target time to search from.
            duration: Total track duration.
            search_forward: Search forward if True, backward if False.

        Returns:
            Time of phrase boundary in seconds.
        """
        if len(beat_times) < 16:
            return target_time

        target_idx = np.argmin(np.abs(beat_times - target_time))

        for phrase_len in [32, 16]:
            if search_forward:
                phrase_idx = ((target_idx + phrase_len - 1) // phrase_len) * phrase_len
            else:
                phrase_idx = (target_idx // phrase_len) * phrase_len

            if 0 <= phrase_idx < len(beat_times):
                return float(beat_times[phrase_idx])

        return target_time

    def _select_energy_aware_mix_point(
        self,
        energy_profile: EnergyProfile,
        bar_based_time: float,
        duration: float,
        is_mix_in: bool,
        min_offset: float,
        beat_times: np.ndarray = None,
        y: np.ndarray = None,
        sr: int = None,
    ) -> List[MixCandidate]:
        """Select mix points using energy gradient scoring.

        Returns sorted list of MixCandidate objects. Empty list on flat energy
        or insufficient data.
        """
        boundaries = energy_profile.energy_at_boundaries
        if len(boundaries) < 2:
            return []

        energies = [e for _, e in boundaries]
        if np.std(energies) < 0.01:
            return []

        half = duration / 2
        if is_mix_in:
            candidates = [
                (i, t, e) for i, (t, e) in enumerate(boundaries) if t >= min_offset and t <= half
            ]
        else:
            candidates = [
                (i, t, e)
                for i, (t, e) in enumerate(boundaries)
                if t <= duration - min_offset and t >= half
            ]

        if not candidates:
            return []

        # Precompute phrase boundary arrays for tolerance-based matching (Phase 2)
        bar32_times = np.array([], dtype=float)
        bar16_times = np.array([], dtype=float)
        if beat_times is not None and len(beat_times) > 0:
            bar32_times = np.array([float(beat_times[i]) for i in range(0, len(beat_times), 128)])
            bar16_times = np.array([float(beat_times[i]) for i in range(0, len(beat_times), 64)])

        cfg = self.config
        sections = energy_profile.sections or []
        max_e = max(energies)
        proximity_threshold = duration * 0.1
        scored = []

        for idx, t, e in candidates:
            prev_e = boundaries[idx - 1][1] if idx > 0 else 0.0
            gradient = e - prev_e

            # Find section type at this point
            section_type = "unknown"
            for s in sections:
                if s.start <= t < s.end:
                    section_type = s.type
                    break

            # Score using pure functions from scoring.py
            score = 0.0
            score += score_section_match(
                section_type, is_mix_in, cfg.mix_score_section_match, cfg.mix_score_energy_gradient
            )
            if idx > 0:
                score += score_energy_gradient(gradient, is_mix_in, cfg.mix_score_energy_gradient)

            phrase_score, phrase_aligned = score_phrase_alignment(
                t,
                bar32_times,
                bar16_times,
                cfg.phrase_alignment_tolerance,
                cfg.mix_score_phrase_32bar,
                cfg.mix_score_phrase_16bar,
            )
            score += phrase_score

            ref_time = energy_profile.intro_end if is_mix_in else energy_profile.outro_start
            score += score_proximity(t, ref_time, proximity_threshold, cfg.mix_score_proximity)
            score += score_energy_penalty(
                e, max_e, is_mix_in, cfg.mix_penalty_low_energy, cfg.mix_penalty_peak_energy
            )

            # Local spectral profile at this candidate (Phase 4)
            local_spectral = None
            if y is not None and sr is not None:
                local_spectral = self._analyze_spectral_at(y, sr, t)

            scored.append(
                (
                    score,
                    abs(t - bar_based_time),
                    MixCandidate(
                        time=t,
                        score=score,
                        section_type=section_type,
                        energy_gradient=round(gradient, 4),
                        phrase_aligned=phrase_aligned,
                        spectral_profile=local_spectral,
                    ),
                )
            )

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [mc for _, _, mc in scored]

    def _analyze_energy(
        self, y: np.ndarray, sr: int, beat_times: np.ndarray, duration: float
    ) -> EnergyProfile:
        """Compute RMS energy profile over the track.

        Args:
            y: Audio time series.
            sr: Sample rate.
            beat_times: Array of beat times in seconds.
            duration: Track duration in seconds.

        Returns:
            EnergyProfile with energy data at phrase boundaries and sections.
        """
        rms = librosa.feature.rms(y=y)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        peak = rms.max()
        rms_norm = rms / (peak + 1e-10)

        phrase_indices = list(range(0, len(beat_times), 64))
        if not phrase_indices:
            phrase_indices = list(range(0, len(beat_times), 16))

        energy_at_boundaries = []
        for idx in phrase_indices:
            t = float(beat_times[idx])
            frame = np.argmin(np.abs(rms_times - t))
            energy_at_boundaries.append((t, float(rms_norm[frame])))

        overall_energy = float(np.clip(np.mean(rms_norm) * 10, 1.0, 10.0))

        # Detect sections (Phase 3: hardened)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
        sections = self._detect_sections(energy_at_boundaries, onset_env, onset_times, duration)

        # Derive intro_end / outro_start from sections
        intro_end = 0.0
        outro_start = duration
        has_intro = False
        for s in sections:
            if s.type == "intro":
                intro_end = s.end
                has_intro = True
        # Phase 3 fix: if no intro detected, use first boundary as intro_end
        if not has_intro and energy_at_boundaries:
            intro_end = energy_at_boundaries[0][0]
        for s in sections:
            if s.type == "outro":
                outro_start = s.start
                break

        # Classify energy shape
        cfg = self.config
        energy_shape = "flat"
        energies = [e for _, e in energy_at_boundaries]
        if len(energies) >= 4:
            q = len(energies) // 4
            start_avg = np.mean(energies[:q])
            end_avg = np.mean(energies[-q:])
            mid_avg = np.mean(energies[q:-q]) if len(energies) > 2 * q else np.mean(energies)
            if (
                mid_avg > start_avg
                and mid_avg > end_avg
                and (mid_avg - min(start_avg, end_avg) > cfg.energy_shape_threshold * 0.67)
            ):
                energy_shape = "peaking"
            elif end_avg - start_avg > cfg.energy_shape_threshold:
                energy_shape = "building"
            elif start_avg - end_avg > cfg.energy_shape_threshold:
                energy_shape = "declining"

        return EnergyProfile(
            overall_energy=round(overall_energy, 1),
            energy_at_boundaries=energy_at_boundaries,
            intro_end=intro_end,
            outro_start=outro_start,
            sections=sections,
            energy_shape=energy_shape,
        )

    def _analyze_spectral(self, y: np.ndarray, sr: int) -> SpectralProfile:
        """Compute global frequency band energy ratios (bass/mid/treble)."""
        spec = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        bass = spec[freqs <= 250].sum()
        mid = spec[(freqs > 250) & (freqs <= 4000)].sum()
        treble = spec[freqs > 4000].sum()
        total = bass + mid + treble + 1e-10
        return SpectralProfile(
            bass_ratio=round(float(bass / total), 3),
            mid_ratio=round(float(mid / total), 3),
            treble_ratio=round(float(treble / total), 3),
        )

    def _analyze_spectral_at(
        self, y: np.ndarray, sr: int, center_time: float, window_sec: float = 8.0
    ) -> SpectralProfile:
        """Compute spectral profile around a specific timestamp (Phase 4).

        Args:
            y: Audio time series.
            sr: Sample rate.
            center_time: Center of the analysis window in seconds.
            window_sec: Window duration in seconds.

        Returns:
            SpectralProfile for the windowed region.
        """
        half_win = int(window_sec / 2 * sr)
        center_sample = int(center_time * sr)
        start = max(0, center_sample - half_win)
        end = min(len(y), center_sample + half_win)
        return self._analyze_spectral(y[start:end], sr)

    def _detect_sections(
        self,
        energy_at_boundaries: list,
        onset_env: np.ndarray,
        onset_times: np.ndarray,
        duration: float,
    ) -> List[Section]:
        """Classify track segments into sections using energy contour.

        Uses a two-pass approach:
        1. Classify each boundary by energy level, gradient, and position.
        2. Merge adjacent same-type sections and enforce minimum duration.
        """
        if len(energy_at_boundaries) < 2:
            return [Section(type="drop", start=0.0, end=duration, energy=0.5)]

        cfg = self.config
        energies = [e for _, e in energy_at_boundaries]
        peak_e = max(energies) if energies else 1.0
        low_thresh = cfg.section_low_energy_ratio * peak_e
        high_thresh = cfg.section_high_energy_ratio * peak_e

        # Pass 1: classify each boundary
        raw_sections = []
        for i in range(len(energy_at_boundaries)):
            t, e = energy_at_boundaries[i]
            end_t = (
                energy_at_boundaries[i + 1][0] if i + 1 < len(energy_at_boundaries) else duration
            )
            prev_e = energy_at_boundaries[i - 1][1] if i > 0 else 0.0
            gradient = e - prev_e
            position_ratio = t / duration if duration > 0 else 0

            if i == 0 and e <= low_thresh:
                stype = "intro"
            elif e <= low_thresh and position_ratio >= cfg.outro_min_position:
                # Phase 3 fix: only classify as outro in final portion of track
                stype = "outro"
            elif e <= low_thresh and prev_e > low_thresh:
                stype = "breakdown"
            elif e <= low_thresh and position_ratio < 0.25:
                stype = "intro"
            elif e <= low_thresh:
                stype = "breakdown"
            elif gradient > cfg.section_gradient_threshold and e < high_thresh:
                stype = "buildup"
            elif e >= high_thresh:
                stype = "drop"
            elif gradient < -cfg.section_gradient_threshold and e < high_thresh:
                stype = "breakdown"
            else:
                stype = "drop" if e > low_thresh else "breakdown"

            raw_sections.append(Section(type=stype, start=t, end=end_t, energy=round(e, 3)))

        # Pass 2: merge adjacent same-type sections
        if not raw_sections:
            return raw_sections

        merged = [raw_sections[0]]
        for s in raw_sections[1:]:
            if s.type == merged[-1].type:
                # Extend previous section
                merged[-1] = Section(
                    type=merged[-1].type,
                    start=merged[-1].start,
                    end=s.end,
                    energy=round(max(merged[-1].energy, s.energy), 3),
                )
            else:
                merged.append(s)

        return merged

    def recommend_transition(
        self,
        result1: AnalysisResult,
        result2: AnalysisResult,
        key_compatible: bool,
    ) -> TransitionRecommendation:
        """Recommend transition parameters between two compatible tracks."""
        e1 = result1.energy_profile.overall_energy if result1.energy_profile else 5.0
        e2 = result2.energy_profile.overall_energy if result2.energy_profile else 5.0
        energy_diff = abs(e1 - e2)

        # Mix duration
        if e1 > 5 and e2 > 5:
            mix_duration_bars = 32
        elif energy_diff > 3:
            mix_duration_bars = 8
        else:
            mix_duration_bars = 16

        # Transition type
        is_breakdown_to_buildup = (
            result1.energy_profile is not None
            and result2.energy_profile is not None
            and result1.mix_out_point >= result1.energy_profile.outro_start
            and result2.mix_in_point <= result2.energy_profile.intro_end
        )

        if energy_diff <= 2 and key_compatible:
            transition_type = "blend"
        elif energy_diff > 4:
            transition_type = "cut"
        elif is_breakdown_to_buildup:
            transition_type = "echo"
        else:
            transition_type = "blend"

        # EQ strategy — prefer local spectral profiles from candidates (Phase 4)
        sp1 = self._get_transition_spectral(result1, is_out=True)
        sp2 = self._get_transition_spectral(result2, is_out=False)

        cfg = self.config
        if transition_type == "cut":
            eq_strategy = "simple_fade"
        elif transition_type == "echo":
            eq_strategy = "filter_sweep"
        elif sp1 and sp2:
            if (
                sp1.bass_ratio > cfg.bass_heavy_threshold
                and sp2.bass_ratio > cfg.bass_heavy_threshold
            ):
                eq_strategy = "bass_swap"
            elif (
                sp1.treble_ratio > cfg.treble_heavy_threshold
                and sp2.treble_ratio > cfg.treble_heavy_threshold
            ):
                eq_strategy = "filter_sweep"
            elif e1 > 5 and e2 > 5:
                eq_strategy = "bass_swap"
            else:
                eq_strategy = "filter_sweep"
        elif e1 > 5 and e2 > 5:
            eq_strategy = "bass_swap"
        else:
            eq_strategy = "filter_sweep"

        return TransitionRecommendation(
            mix_duration_bars=mix_duration_bars,
            transition_type=transition_type,
            eq_strategy=eq_strategy,
        )

    def _get_transition_spectral(
        self, result: AnalysisResult, is_out: bool
    ) -> Optional[SpectralProfile]:
        """Get the best spectral profile for a transition point.

        Prefers local spectral from the top mix candidate, falls back to global.
        """
        candidates = result.mix_out_candidates if is_out else result.mix_in_candidates
        if candidates and candidates[0].spectral_profile:
            return candidates[0].spectral_profile
        return result.spectral_profile

    def check_compatibility(
        self, result1: AnalysisResult, result2: AnalysisResult
    ) -> Optional[CompatibilityResult]:
        """Checks if two analyzed tracks are compatible for mixing.

        Returns scored CompatibilityResult (0-100) or None if fundamentally
        incompatible (missing BPM, tempo too far, no key relationship).
        """
        if result1.bpm is None or result2.bpm is None:
            return None

        cfg = self.config
        tempo_diff = result2.bpm - result1.bpm
        if abs(tempo_diff) > cfg.tempo_tolerance:
            return None

        # Key compatibility
        key_result = score_key_compatibility(result1.key, result2.key, cfg.compat_key_weight)
        if key_result is None:
            return None
        key_score, reason = key_result

        # Tempo proximity
        tempo_score = score_tempo_proximity(
            tempo_diff, cfg.tempo_tolerance, cfg.compat_tempo_weight
        )

        # Energy flow
        energy_score = 0.0
        ep1 = result1.energy_profile
        ep2 = result2.energy_profile
        if ep1 and ep2:
            out_energy = ep1.energy_at_boundaries[-1][1] if ep1.energy_at_boundaries else 0.5
            in_energy = ep2.energy_at_boundaries[0][1] if ep2.energy_at_boundaries else 0.5
            out_gradient = None
            if len(ep1.energy_at_boundaries) >= 2:
                out_gradient = out_energy - ep1.energy_at_boundaries[-2][1]

            energy_score = score_energy_flow(
                out_energy,
                in_energy,
                out_gradient,
                ep1.energy_shape,
                ep2.energy_shape,
                cfg.compat_energy_weight,
            )

        # Section compatibility
        section_score = 0.0
        if ep1 and ep1.sections and ep2 and ep2.sections:
            out_section = ep1.sections[-1].type
            in_section = ep2.sections[0].type
            section_score = score_section_compatibility(
                out_section, in_section, cfg.compat_section_weight
            )

        total_score = round(key_score + tempo_score + energy_score + section_score, 1)
        breakdown = {
            "key": key_score,
            "tempo": tempo_score,
            "energy": energy_score,
            "sections": section_score,
        }

        # Phase 5 fix: early return when below threshold — no transition computed
        if total_score < cfg.compat_min_score:
            return CompatibilityResult(
                compatible=False,
                tempo_diff=tempo_diff,
                key_reason=reason,
                transition=None,
                score=total_score,
                score_breakdown=breakdown,
            )

        # Pair-optimize mix points from candidates
        optimal_mix_out = None
        optimal_mix_in = None
        out_candidates = result1.mix_out_candidates
        in_candidates = result2.mix_in_candidates
        limit = cfg.pair_candidate_limit
        if out_candidates and in_candidates:
            best_pair_score = -1.0
            for oc in out_candidates[:limit]:
                for ic in in_candidates[:limit]:
                    pair_score = oc.score + ic.score
                    # Bonus for smooth energy transition at these specific points
                    if oc.energy_gradient < 0 and ic.energy_gradient > 0:
                        pair_score += 2
                    # Bonus for outro→intro section pairing
                    if oc.section_type == "outro" and ic.section_type == "intro":
                        pair_score += 2
                    # Phase 5: spectral blend bonus from local profiles
                    pair_score += score_spectral_blend(oc.spectral_profile, ic.spectral_profile)
                    if pair_score > best_pair_score:
                        best_pair_score = pair_score
                        optimal_mix_out = oc.time
                        optimal_mix_in = ic.time

        transition = self.recommend_transition(result1, result2, key_compatible=True)

        return CompatibilityResult(
            compatible=True,
            tempo_diff=tempo_diff,
            key_reason=reason,
            transition=transition,
            score=total_score,
            score_breakdown=breakdown,
            optimal_mix_out=optimal_mix_out,
            optimal_mix_in=optimal_mix_in,
        )
