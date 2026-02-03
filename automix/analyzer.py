"""Audio analysis module for DJ mix transition detection."""

import librosa
import numpy as np

from .exceptions import AudioLoadError, AudioTooShortError


class AudioAnalyzer:
    """Analyzes audio files to detect BPM, key, and optimal mix points."""

    def analyze(self, file_path):
        """Analyzes an audio file for DJ mixing parameters.

        Args:
            file_path: Path to the audio file to analyze.

        Returns:
            dict: Analysis results containing:
                - bpm (float|None): Detected tempo in beats per minute
                - bpm_str (str): Human-readable BPM string
                - key (str): Detected musical key (e.g., 'Am', 'C')
                - mix_in_point (float|None): Suggested mix-in time in seconds
                - mix_out_point (float|None): Suggested mix-out time in seconds
                - confidence (dict): Confidence scores for bpm and key
                - warning (str, optional): Warning message for short files

        Raises:
            AudioLoadError: If the audio file cannot be loaded.
            AudioTooShortError: If the audio file is too short for analysis.

        Example:
            >>> analyzer = AudioAnalyzer()
            >>> result = analyzer.analyze('track.mp3')
            >>> print(f"BPM: {result['bpm']}, Key: {result['key']}")
        """
        try:
            y, sr = librosa.load(file_path)
        except Exception:
            raise AudioLoadError("Unable to load audio file")

        duration = librosa.get_duration(y=y, sr=sr)

        if duration < 10:
            raise AudioTooShortError("File too short for reliable analysis")

        # Beat detection with real confidence
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if len(beats) > 0 else None

        # Calculate BPM confidence from beat strength consistency
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

        bpm_str = f"{bpm:.1f}" if bpm else "Unknown"
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Key detection with major/minor classification
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Major and minor profiles (Krumhansl-Schmuckler key-finding)
        major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )

        # Normalize profiles
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)

        # Normalize chroma
        chroma_norm = chroma_mean / (np.sum(chroma_mean) + 1e-6)

        # Calculate correlation for all keys
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        max_corr = -1
        best_key = "C"
        best_mode = "major"

        for i in range(12):
            # Rotate profiles to match key
            major_rot = np.roll(major_profile, i)
            minor_rot = np.roll(minor_profile, i)

            # Calculate correlation
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

        # Calculate key confidence from correlation
        key_confidence = float(np.clip(max_corr, 0.0, 1.0)) if max_corr > 0 else 0.0

        # Mix-in: first beat after minimum 5 seconds from start
        mix_in_point = 0.0
        for bt in beat_times:
            if bt >= 5.0:
                mix_in_point = bt
                break
        if mix_in_point == 0.0 and len(beat_times) > 0:
            mix_in_point = max(beat_times[0], 5.0)

        # Mix-out: last beat before minimum 10 seconds from end
        mix_out_point = duration * 0.9
        min_mix_out = duration - 10.0
        for bt in reversed(beat_times):
            if bt <= min_mix_out:
                mix_out_point = bt
                break

        return {
            "bpm": bpm,
            "bpm_str": bpm_str,
            "key": key,
            "mix_in_point": mix_in_point,
            "mix_out_point": mix_out_point,
            "confidence": {"bpm": bpm_confidence, "key": key_confidence},
        }

    def check_compatibility(self, result1, result2):
        """Checks if two analyzed tracks are compatible for mixing.

        Args:
            result1: Analysis result dict from first track.
            result2: Analysis result dict from second track.

        Returns:
            dict|None: Compatibility info if compatible, None otherwise. Contains:
                - compatible (bool): Always True when returned
                - tempo_diff (float): BPM difference between tracks
                - key_reason (str): Reason for key compatibility

        Example:
            >>> compat = analyzer.check_compatibility(result1, result2)
            >>> if compat:
            ...     print(f"Compatible: {compat['key_reason']}")
        """
        if result1["bpm"] is None or result2["bpm"] is None:
            return None

        tempo_diff = result2["bpm"] - result1["bpm"]
        if abs(tempo_diff) > 6:
            return None

        key1 = result1["key"]
        key2 = result2["key"]

        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        def parse_key(k):
            if k.endswith("m"):
                return keys.index(k[:-1]), "minor"
            return keys.index(k), "major"

        idx1, mode1 = parse_key(key1)
        idx2, mode2 = parse_key(key2)

        semitone_diff = (idx2 - idx1) % 12

        # Same key
        if key1 == key2:
            reason = "same key"
        # Relative major/minor (3 semitones apart, opposite modes)
        elif semitone_diff == 3 and mode1 == "minor" and mode2 == "major":
            reason = "relative major"
        elif semitone_diff == 9 and mode1 == "major" and mode2 == "minor":
            reason = "relative minor"
        # Perfect fifth (+7 semitones)
        elif semitone_diff == 7:
            reason = "perfect fifth"
        # Perfect fourth (-5 semitones = +7 semitones)
        elif semitone_diff == 5:
            reason = "perfect fourth"
        # Adjacent keys (±1 semitone)
        elif semitone_diff == 1 or semitone_diff == 11:
            reason = "adjacent key"
        else:
            return None

        return {"compatible": True, "tempo_diff": tempo_diff, "key_reason": reason}
