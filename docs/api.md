# AudioAnalyzer API Documentation

## Overview

The `AudioAnalyzer` class analyzes audio files to detect tempo (BPM), musical key, and optimal DJ mix transition points.

## Class: AudioAnalyzer

```python
from automix.analyzer import AudioAnalyzer

analyzer = AudioAnalyzer()
```

### Methods

#### analyze(file_path)

Analyzes an audio file for DJ mixing parameters.

**Parameters:**
- `file_path` (str): Path to the audio file to analyze

**Returns:**

Dictionary containing:
- `bpm` (float|None): Detected tempo in beats per minute, or None if no beats detected
- `bpm_str` (str): Human-readable BPM string (e.g., "128.5" or "Unknown")
- `key` (str): Detected musical key (e.g., "Am", "C")
- `mix_in_point` (float|None): Suggested mix-in time in seconds, or None for short files
- `mix_out_point` (float|None): Suggested mix-out time in seconds, or None for short files
- `confidence` (dict): Confidence scores with keys:
  - `bpm` (float): BPM detection confidence (0.0-1.0)
  - `key` (float): Key detection confidence (0.0-1.0)
- `warning` (str, optional): Warning message for files too short for analysis

**Raises:**
- `ValueError`: If the audio file cannot be loaded

**Example:**

```python
analyzer = AudioAnalyzer()
result = analyzer.analyze('track.mp3')

print(f"BPM: {result['bpm']}")
print(f"Key: {result['key']}")
print(f"Mix in at: {result['mix_in_point']:.1f}s")
print(f"Mix out at: {result['mix_out_point']:.1f}s")
```

**Mix Point Calculation:**
- Tracks >20 seconds: Mix-in at 5% of duration, mix-out at 95%
- Tracks ≤20 seconds: Mix-in at 10% of duration, mix-out at 90%

**Edge Cases:**
- Files <10 seconds return warning and None values
- Files with no clear beats return `bpm=None` and `bpm_str="Unknown"`

---

#### check_compatibility(result1, result2)

Checks if two analyzed tracks are compatible for mixing based on tempo and harmonic key.

**Parameters:**
- `result1` (dict): Analysis result from first track (from `analyze()`)
- `result2` (dict): Analysis result from second track (from `analyze()`)

**Returns:**

Dictionary if compatible, None otherwise. When compatible, contains:
- `compatible` (bool): Always True when returned
- `tempo_diff` (float): BPM difference (result2 - result1)
- `key_reason` (str): Reason for key compatibility, one of:
  - `"same key"`: Exact key match
  - `"relative major"`: Relative major/minor relationship
  - `"relative minor"`: Relative major/minor relationship
  - `"perfect fifth"`: +7 semitones
  - `"perfect fourth"`: -5 semitones
  - `"adjacent key"`: ±1 semitone

**Compatibility Rules:**

Tempo: BPM difference must be ≤6 BPM

Key: Must match one of the harmonic relationships above

**Example:**

```python
analyzer = AudioAnalyzer()
result1 = analyzer.analyze('track1.mp3')
result2 = analyzer.analyze('track2.mp3')

compat = analyzer.check_compatibility(result1, result2)
if compat:
    print(f"Compatible! Key: {compat['key_reason']}, Tempo: {compat['tempo_diff']:+.1f} BPM")
else:
    print("Not compatible for mixing")
```

**Returns None when:**
- Either track has `bpm=None`
- Tempo difference exceeds 6 BPM
- Keys are not harmonically compatible

## Supported Audio Formats

- MP3
- WAV
- FLAC
- OGG

## Dependencies

- librosa >= 0.10.0
- numpy >= 1.24.0
