"""Audio analysis module for DJ mix transition detection."""

from typing import Optional

import librosa
import numpy as np

from .cache import ResultCache
from .config import DEFAULT_CONFIG, AnalysisConfig
from .exceptions import AudioLoadError, AudioTooShortError
from .logging_config import get_logger
from .models import AnalysisResult, CompatibilityResult, EnergyProfile

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

        # Phrase-aware mix points
        mix_in_point = self._find_phrase_boundary(
            beat_times, mix_in_offset, duration, search_forward=True
        )
        mix_out_point = self._find_phrase_boundary(
            beat_times, duration - mix_out_offset, duration, search_forward=False
        )

        logger.info(
            "Analysis complete: BPM=%.1f (%.2f), Key=%s (%.2f)",
            bpm if bpm else 0,
            bpm_confidence,
            key,
            key_confidence,
        )

        # Energy analysis
        energy_profile = self._analyze_energy(y, sr, beat_times, duration)

        result = AnalysisResult(
            bpm=bpm,
            key=key,
            mix_in_point=mix_in_point,
            mix_out_point=mix_out_point,
            bpm_confidence=bpm_confidence,
            key_confidence=key_confidence,
            energy_profile=energy_profile,
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
            EnergyProfile with energy data at phrase boundaries.
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

        # 4. Detect intro_end: first boundary where energy > 50% peak, sustained ≥1 phrase
        threshold = 0.5
        intro_end = 0.0
        for i, (t, e) in enumerate(energy_at_boundaries):
            if t > duration / 2:
                break
            if e > threshold and (
                i + 1 >= len(energy_at_boundaries)
                or energy_at_boundaries[i + 1][1] > threshold
            ):
                intro_end = t
                break

        # 5. Detect outro_start: last boundary where energy drops below 50%, sustained to end
        outro_start = duration
        for i in range(len(energy_at_boundaries) - 1, -1, -1):
            t, e = energy_at_boundaries[i]
            if t < duration / 2:
                break
            if e < threshold:
                outro_start = t
            else:
                break

        # 6. Overall energy: mean RMS mapped to 1-10
        overall_energy = float(np.clip(np.mean(rms_norm) * 10, 1.0, 10.0))

        return EnergyProfile(
            overall_energy=round(overall_energy, 1),
            energy_at_boundaries=energy_at_boundaries,
            intro_end=intro_end,
            outro_start=outro_start,
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

        return CompatibilityResult(compatible=True, tempo_diff=tempo_diff, key_reason=reason)
