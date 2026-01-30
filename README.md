# AutoMix AI

A Python CLI tool that analyzes audio files to identify optimal DJ mix transition points using beat detection, harmonic analysis, and structural segmentation.

## Features

- **Beat Detection**: Identifies tempo (BPM) and beat positions
- **Harmonic Analysis**: Detects musical key for harmonic mixing
- **Mix Point Detection**: Finds optimal mix-in and mix-out points
- **Compatibility Checking**: Suggests compatible track pairs based on key and tempo
- **Multiple Formats**: Supports MP3, WAV, FLAC, and OGG files
- **JSON Output**: Machine-readable output for integration

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the project
uv pip install -e .
```

Or with pip:

```bash
pip install -e .
```

## Usage

### Analyze a Single Track

```bash
automix analyze track.mp3
```

Output:
```
Analyzing: track.mp3
BPM: 128.5 (confidence: 0.95)
Key: Am (confidence: 0.87)
Mix-in point: 0:15.2
Mix-out point: 4:05.8
```

### Analyze Multiple Tracks

```bash
automix analyze track1.mp3 track2.mp3
```

Output shows individual analysis plus compatible pairs:
```
Analyzing: track1.mp3
BPM: 128.5 (confidence: 0.95)
Key: Am (confidence: 0.87)
Mix-in point: 0:15.2
Mix-out point: 4:05.8

Analyzing: track2.mp3
BPM: 130.0 (confidence: 0.95)
Key: Cm (confidence: 0.87)
Mix-in point: 0:08.5
Mix-out point: 3:45.2

Compatible pairs:
✓ track1.mp3 → track2.mp3 (key: relative major, tempo: +1.5 BPM)
```

### No Compatible Pairs

When tracks are incompatible:

```bash
automix analyze track1.mp3 track2.mp3
```

Output:
```
Analyzing: track1.mp3
BPM: 128.5 (confidence: 0.95)
Key: Am (confidence: 0.87)
Mix-in point: 0:15.2
Mix-out point: 4:05.8

Analyzing: track2.mp3
BPM: 95.0 (confidence: 0.95)
Key: F#m (confidence: 0.87)
Mix-in point: 0:08.5
Mix-out point: 3:45.2

No compatible mix pairs found
```

### JSON Output

Single file:
```bash
automix analyze track.mp3 --format json
```

Output:
```json
{
  "file": "track.mp3",
  "bpm": 128.5,
  "key": "Am",
  "mix_in_point": 15.2,
  "mix_out_point": 245.8,
  "confidence": {
    "bpm": 0.95,
    "key": 0.87
  }
}
```

Multiple files:
```bash
automix analyze track1.mp3 track2.mp3 --format json
```

Output:
```json
[
  {
    "file": "track1.mp3",
    "bpm": 128.5,
    "key": "Am",
    "mix_in_point": 15.2,
    "mix_out_point": 245.8,
    "confidence": {
      "bpm": 0.95,
      "key": 0.87
    }
  },
  {
    "file": "track2.mp3",
    "bpm": 130.0,
    "key": "Cm",
    "mix_in_point": 8.5,
    "mix_out_point": 225.2,
    "confidence": {
      "bpm": 0.95,
      "key": 0.87
    }
  }
]
```

## Compatibility Rules

### Key Compatibility

Tracks are harmonically compatible if they meet any of these criteria:

- **Same key**: Exact match (e.g., Am → Am)
- **Relative major/minor**: Share same notes (e.g., Am → C, C → Am)
- **Perfect fifth**: +7 semitones (e.g., Am → Em, C → G)
- **Perfect fourth**: -5 semitones (e.g., Am → Dm, C → F)
- **Adjacent keys**: ±1 semitone (e.g., Am → A#m, C → C#)

### Tempo Compatibility

Tracks are tempo-compatible if BPM difference is ≤6 BPM.

## Requirements

- Python 3.8+
- librosa >= 0.10.0
- numpy >= 1.24.0
- click >= 8.1.0

## Development

Install development dependencies:

```bash
uv pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Format and lint code with ruff:

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Lint and auto-fix issues
ruff check --fix .
```

## Mix Point Calculation

Mix points are calculated based on track duration:

- **Tracks >20 seconds**: Mix-in at 5% of duration, mix-out at 95%
- **Tracks ≤20 seconds**: Mix-in at 10% of duration, mix-out at 90%

This ensures longer tracks have more intro/outro space while shorter tracks maintain usable mix windows.

## Limitations

- Analysis requires at least 10 seconds of audio
- Tracks without clear beats report "Unknown" BPM
- Performance: ~10 seconds for a 5-minute audio file

## Edge Cases

- **Very short audio files (<10 seconds)**: 
  ```
  Analyzing: short.mp3
  File too short for reliable analysis
  BPM: Unknown (confidence: 0.00)
  Key: None (confidence: 0.00)
  Mix-in point: None
  Mix-out point: None
  ```
- **Audio with no clear beat**: Report BPM as "Unknown" with confidence 0.00
- **Corrupted audio files**: Display "Error: Unable to decode audio file"
- **Empty file path**: Display usage help

## License

MIT