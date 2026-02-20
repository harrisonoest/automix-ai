"""Audio analysis module for DJ mix transition detection."""

from typing import List, Optional

import librosa
import numpy as np

from .cache import ResultCache
from .config import DEFAULT_CONFIG, AnalysisConfig
from .exceptions import AudioLoadError, AudioTooShortError
from .logging_config import get_logger
from .models import (
    AnalysisResult,
    CompatibilityResult,
    EnergyProfile,
    MixCandidate,
    Section,
    TransitionRecommendation,
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
                energy_profile, bar_based_mix_in, duration, True, mix_in_offset, beat_times
            )
            if in_candidates:
                mix_in_candidates = in_candidates
                mix_in_point = in_candidates[0].time

            out_candidates = self._select_energy_aware_mix_point(
                energy_profile, bar_based_mix_out, duration, False, mix_out_offset, beat_times
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
            mix_in_candidates=mix_in_candidates,
            mix_out_candidates=mix_out_candidates,
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
            # Not enough beats, fall back to simple method
            return target_time

        # Find beat closest to target
        target_idx = np.argmin(np.abs(beat_times - target_time))

        # Check for 32-bar phrases first (preferred), then 16-bar
        for phrase_len in [32, 16]:
            if search_forward:
                # Round up to next phrase boundary
                phrase_idx = ((target_idx + phrase_len - 1) // phrase_len) * phrase_len
            else:
                # Round down to previous phrase boundary
                phrase_idx = (target_idx // phrase_len) * phrase_len

            if 0 <= phrase_idx < len(beat_times):
                return float(beat_times[phrase_idx])

        # Fallback to target time
        return target_time

    def _select_energy_aware_mix_point(
        self,
        energy_profile: EnergyProfile,
        bar_based_time: float,
        duration: float,
        is_mix_in: bool,
        min_offset: float,
        beat_times: np.ndarray = None,
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

        # Precompute phrase boundary sets for alignment scoring
        bar32_times = set()
        bar16_times = set()
        if beat_times is not None and len(beat_times) > 0:
            for i in range(0, len(beat_times), 128):  # 32 bars * 4 beats
                bar32_times.add(round(float(beat_times[i]), 6))
            for i in range(0, len(beat_times), 64):  # 16 bars * 4 beats
                bar16_times.add(round(float(beat_times[i]), 6))

        sections = energy_profile.sections or []
        max_e = max(energies)
        proximity_threshold = duration * 0.1
        scored = []

        for idx, t, e in candidates:
            score = 0.0
            prev_e = boundaries[idx - 1][1] if idx > 0 else 0.0
            gradient = e - prev_e

            # Section type suitability: +3 intro/outro, +2 buildup/breakdown
            section_type = "unknown"
            for s in sections:
                if s.start <= t < s.end:
                    section_type = s.type
                    break
            if is_mix_in and section_type == "intro":
                score += 3
            elif not is_mix_in and section_type == "outro":
                score += 3
            elif section_type in ("buildup", "breakdown"):
                score += 2

            # Energy gradient: +2 for rising (mix-in) or falling (mix-out)
            if idx > 0:
                if is_mix_in and e > prev_e:
                    score += 2
                elif not is_mix_in and e < prev_e:
                    score += 2

            # Phrase alignment: +2 for 32-bar, +1 for 16-bar
            t_rounded = round(t, 6)
            phrase_aligned = False
            if t_rounded in bar32_times:
                score += 2
                phrase_aligned = True
            elif t_rounded in bar16_times:
                score += 1
                phrase_aligned = True

            # Proximity to intro_end / outro_start: +1
            if is_mix_in and abs(t - energy_profile.intro_end) < proximity_threshold:
                score += 1
            elif not is_mix_in and abs(t - energy_profile.outro_start) < proximity_threshold:
                score += 1

            # Penalties: -1 for low energy mix-in or peak energy mix-out
            if is_mix_in and e < 0.1:
                score -= 1
            elif not is_mix_in and max_e > 0 and e > max_e * 0.8:
                score -= 1

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
        # 1. Compute RMS energy and normalize to 0-1
        rms = librosa.feature.rms(y=y)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        peak = rms.max()
        rms_norm = rms / (peak + 1e-10)

        # 2. Get phrase boundary times (every 16 beats = 4 bars, using 16-bar = 64 beats)
        phrase_indices = list(range(0, len(beat_times), 64))  # 16 bars * 4 beats
        if not phrase_indices:
            phrase_indices = list(range(0, len(beat_times), 16))  # fallback to 4 bars

        # 3. Sample energy at each phrase boundary
        energy_at_boundaries = []
        for idx in phrase_indices:
            t = float(beat_times[idx])
            frame = np.argmin(np.abs(rms_times - t))
            energy_at_boundaries.append((t, float(rms_norm[frame])))

        # 4. Overall energy: mean RMS mapped to 1-10
        overall_energy = float(np.clip(np.mean(rms_norm) * 10, 1.0, 10.0))

        # 5. Detect sections and derive intro_end/outro_start
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
        sections = self._detect_sections(energy_at_boundaries, onset_env, onset_times, duration)

        intro_end = 0.0
        outro_start = duration
        for s in sections:
            if s.type == "intro":
                intro_end = s.end
            if s.type == "outro":
                outro_start = s.start
                break  # first outro boundary

        return EnergyProfile(
            overall_energy=round(overall_energy, 1),
            energy_at_boundaries=energy_at_boundaries,
            intro_end=intro_end,
            outro_start=outro_start,
            sections=sections,
        )

    def _detect_sections(
        self,
        energy_at_boundaries: list,
        onset_env: np.ndarray,
        onset_times: np.ndarray,
        duration: float,
    ) -> list:
        """Classify track segments into sections using energy contour and onset strength.

        Args:
            energy_at_boundaries: List of (time, energy) tuples at phrase boundaries.
            onset_env: Onset strength envelope.
            onset_times: Times corresponding to onset envelope frames.
            duration: Track duration in seconds.

        Returns:
            List of Section objects covering the full track.
        """
        if len(energy_at_boundaries) < 2:
            return [Section(type="drop", start=0.0, end=duration, energy=0.5)]

        sections = []
        energies = [e for _, e in energy_at_boundaries]
        peak_e = max(energies) if energies else 1.0
        low_thresh = 0.3 * peak_e
        high_thresh = 0.7 * peak_e

        for i in range(len(energy_at_boundaries)):
            t, e = energy_at_boundaries[i]
            has_next = i + 1 < len(energy_at_boundaries)
            end_t = energy_at_boundaries[i + 1][0] if has_next else duration
            prev_e = energy_at_boundaries[i - 1][1] if i > 0 else 0.0
            gradient = e - prev_e

            # Classify by position, energy level, and gradient
            if i == 0 and e <= low_thresh:
                stype = "intro"
            elif i == len(energy_at_boundaries) - 1 and e <= low_thresh:
                stype = "outro"
            elif e <= low_thresh and prev_e > low_thresh:
                stype = "outro" if t > duration / 2 else "breakdown"
            elif e <= low_thresh:
                stype = "intro" if t < duration / 2 else "outro"
            elif gradient > 0.1 and e < high_thresh:
                stype = "buildup"
            elif e >= high_thresh:
                stype = "drop"
            elif gradient < -0.1 and e < high_thresh:
                stype = "breakdown"
            else:
                stype = "drop" if e > low_thresh else "breakdown"

            sections.append(Section(type=stype, start=t, end=end_t, energy=round(e, 3)))

        return sections

    def recommend_transition(
        self,
        result1: AnalysisResult,
        result2: AnalysisResult,
        key_compatible: bool,
    ) -> TransitionRecommendation:
        """Recommend transition parameters between two compatible tracks.

        Args:
            result1: Analysis result from outgoing track.
            result2: Analysis result from incoming track.
            key_compatible: Whether the tracks are key-compatible.

        Returns:
            TransitionRecommendation with mix duration, type, and EQ strategy.
        """
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

        # EQ strategy
        if transition_type == "cut":
            eq_strategy = "simple_fade"
        elif transition_type == "echo":
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

    def check_compatibility(
        self, result1: AnalysisResult, result2: AnalysisResult
    ) -> CompatibilityResult:
        """Checks if two analyzed tracks are compatible for mixing.

        Args:
            result1: Analysis result from first track.
            result2: Analysis result from second track.

        Returns:
            CompatibilityResult or None: Compatibility info if compatible, None otherwise.
        """
        if result1.bpm is None or result2.bpm is None:
            return None

        tempo_diff = result2.bpm - result1.bpm
        if abs(tempo_diff) > self.config.tempo_tolerance:
            return None

        key1 = result1.key
        key2 = result2.key

        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        def parse_key(k):
            if k.endswith("m"):
                return keys.index(k[:-1]), "minor"
            return keys.index(k), "major"

        idx1, mode1 = parse_key(key1)
        idx2, mode2 = parse_key(key2)

        semitone_diff = (idx2 - idx1) % 12

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
        elif semitone_diff == 1 or semitone_diff == 11:
            reason = "adjacent key"
        else:
            return None

        transition = self.recommend_transition(result1, result2, key_compatible=True)

        return CompatibilityResult(
            compatible=True, tempo_diff=tempo_diff, key_reason=reason, transition=transition
        )
