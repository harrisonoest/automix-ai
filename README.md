# AutoMix AI

A Python CLI tool that analyzes audio files to identify optimal DJ mix transition points using beat detection, harmonic analysis, energy-aware section detection, and spectral analysis.

## Features

- **Beat Detection**: Identifies tempo (BPM) and beat positions with real confidence scores
- **Harmonic Analysis**: Detects musical key (major/minor) using Krumhansl-Schmuckler algorithm
- **Phrase Detection**: Finds mix points at 16/32 bar boundaries for DJ-quality transitions
- **Section Detection**: Classifies track structure (intro, buildup, drop, breakdown, outro) using energy contour analysis
- **Spectral Analysis**: Computes bass/mid/treble frequency ratios globally and at mix points for EQ strategy recommendations
- **Scored Mix Candidates**: Ranks multiple mix-in/mix-out candidates by section type, energy gradient, phrase alignment, and proximity
- **Compatibility Scoring**: Rates track pairs 0-100 with breakdown across key, tempo, energy flow, and section compatibility
- **Result Caching**: 100x faster repeated analysis with automatic caching and version-based invalidation
- **Logging**: Production-ready logging with `--verbose` flag
- **Multiple Formats**: Supports MP3, WAV, FLAC, and OGG files
- **JSON Output**: Machine-readable output for integration
- **SoundCloud Integration**: Search and analyze tracks directly from SoundCloud
- **Waveform Visualizer**: Terminal waveform display with section boundaries and mix candidate positions

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
BPM: 128.5 (confidence: 0.92)
Key: Am (confidence: 0.85)
Energy: 6.2/10
Mix-in point: 0:16.0
Mix-out point: 4:32.0
  Mix-in candidates:
    1. 0:16.0  score=7.0  intro ↑ ✓ phrase
    2. 0:32.0  score=5.0  buildup ↑ ✓ phrase
    3. 0:48.0  score=3.0  drop →   off-grid
  Mix-out candidates:
    1. 4:32.0  score=8.0  outro ↓ ✓ phrase
    2. 4:16.0  score=5.0  breakdown ↓ ✓ phrase
    3. 4:00.0  score=3.0  drop →   off-grid
```

### Visualize Waveform

```bash
automix analyze track.mp3 --visualize
```

Displays a terminal waveform with section boundaries (INTRO, DROP, OUTRO), mix-in/mix-out markers, and numbered candidate positions.

### Verbose Logging

Enable detailed logging with the `--verbose` or `-v` flag:

```bash
automix -v analyze track.mp3
```

### Analyze Multiple Tracks

```bash
automix analyze track1.mp3 track2.mp3
```

Output shows individual analysis plus compatible pairs with score breakdown:
```
Analyzing: track1.mp3
BPM: 128.5 (confidence: 0.95)
Key: Am (confidence: 0.87)
Energy: 6.5/10
Mix-in point: 0:15.2
Mix-out point: 4:05.8

Analyzing: track2.mp3
BPM: 130.0 (confidence: 0.95)
Key: Cm (confidence: 0.87)
Energy: 7.1/10
Mix-in point: 0:08.5
Mix-out point: 3:45.2

Compatible pairs:
✓ track1.mp3 → track2.mp3 (key: relative major, tempo: +1.5 BPM)
  Score: Key: 35 + Tempo: 22 + Energy: 12 + Sections: 15 = 84/100
  Transition: blend over 32 bars, bass_swap, energy: 6.5 → 7.1
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

### Search SoundCloud

Search for tracks on SoundCloud:

```bash
automix search "deep house"
```

Output:
```
Found 10 track(s):

1. Artist Name - Track Title
   Duration: 5:23
   Genre: Deep House
   URL: https://soundcloud.com/artist/track

2. Another Artist - Another Track
   Duration: 6:15
   Genre: House
   URL: https://soundcloud.com/artist/track2
```

Limit results:

```bash
automix search "techno" --limit 5
```

Search for multiple different tracks:

```bash
automix search "matroda gimme some keys" "fisher losing it"
```

Search and analyze tracks:

```bash
automix search "deep house" --limit 3 --analyze
```

Search for multiple specific tracks and compare them:

```bash
automix search "matroda gimme some keys" "fisher losing it" --analyze
```

Output:
```
Downloading: MATRODA - Matroda - Gimme Some Keys
Analyzing: MATRODA - Matroda - Gimme Some Keys
Downloading: FISHER - FISHER - Losing It
Analyzing: FISHER - FISHER - Losing It

MATRODA - Matroda - Gimme Some Keys
URL: https://soundcloud.com/matrodamusic/gimmesomekeys
BPM: 123.0 (confidence: 0.95)
Key: Em (confidence: 0.87)
Mix-in point: 0:05.4
Mix-out point: 3:09.9

FISHER - FISHER - Losing It
URL: https://soundcloud.com/fish-tales/fisher-losing-it
BPM: 123.0 (confidence: 0.95)
Key: Gm (confidence: 0.87)
Mix-in point: 0:05.4
Mix-out point: 3:57.9

Compatible pairs:
✓ MATRODA - Matroda - Gimme Some Keys → FISHER - FISHER - Losing It (key: perfect fifth, tempo: +0.0 BPM)
  Score: Key: 30 + Tempo: 30 + Energy: 10 + Sections: 12 = 82/100
```

**Caching**: Analysis results are automatically cached. Searching for the same track again will use the cached result:

```bash
# First search - downloads and analyzes
automix search "artist track" --analyze

# Second search - uses cached analysis (much faster!)
automix search "artist track" --analyze
# Output: Using cached analysis: Artist - Track
```

Note: Downloaded audio files are stored in a temporary directory and automatically deleted after analysis. Analysis results are cached in `./.automix/cache/` for future use. Cache entries are automatically invalidated when the analysis schema changes.

Authentication (optional):

```bash
# No authentication required - yt-dlp handles SoundCloud automatically
automix search "techno" --limit 5 --analyze
```

Note: The `--client-id` and `--auth-token` options are available for the search functionality but are not required for downloading tracks.

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
  "energy": 6.2,
  "mix_in_point": 15.2,
  "mix_out_point": 245.8,
  "confidence": {
    "bpm": 0.95,
    "key": 0.87
  }
}
```

Multiple files with compatibility:
```bash
automix analyze track1.mp3 track2.mp3 --format json
```

Output:
```json
{
  "tracks": [...],
  "compatible_pairs": [
    {
      "track1": "track1.mp3",
      "track2": "track2.mp3",
      "compatible": true,
      "score": 84.0,
      "score_breakdown": {
        "key": 35.0,
        "tempo": 22.5,
        "energy": 12.0,
        "sections": 15.0
      },
      "tempo_diff": 1.5,
      "key_reason": "relative major",
      "transition": {
        "mix_duration_bars": 32,
        "transition_type": "blend",
        "eq_strategy": "bass_swap"
      }
    }
  ]
}
```

## Key Features

### Real Confidence Scores

Confidence scores are calculated from actual analysis metrics:
- **BPM confidence**: Based on beat strength consistency
- **Key confidence**: Based on correlation with key profiles

No fake hardcoded values - confidence reflects actual analysis quality.

### Result Caching

Analysis results are automatically cached in `./.automix/cache/`:
- **100x faster** for repeated analysis
- **Content-based**: Works across file renames/moves
- **URL-based**: Works with `search --analyze` across sessions
- **Version-aware**: Stale cache entries auto-invalidate when the analysis schema changes
- **Automatic**: No configuration needed

### Section Detection

Tracks are automatically segmented into structural sections:
- **Intro**: Low energy at the start of the track
- **Buildup**: Rising energy leading into a drop
- **Drop**: High energy peak sections
- **Breakdown**: Energy dip mid-track (not confused with outro)
- **Outro**: Low energy in the final portion of the track (≥75% position)

Section detection uses a two-pass approach: classify by energy + gradient, then merge adjacent same-type sections to prevent oscillating labels.

### Scored Mix Candidates

Instead of a single mix point, the tool ranks multiple candidates by:
- **Section type**: Intro/outro preferred (+3), buildup/breakdown secondary (+2)
- **Energy gradient**: Rising energy for mix-in, falling for mix-out (+2)
- **Phrase alignment**: 32-bar boundary (+2) or 16-bar boundary (+1), using tolerance-based matching
- **Proximity**: Nearness to intro end / outro start (+1)
- **Penalties**: Low energy mix-in (-1), peak energy mix-out (-1)

The top 3 candidates are shown so the DJ can make an informed choice.

### Spectral Analysis

Each track gets a global spectral profile (bass/mid/treble ratios), and each mix candidate gets a local spectral profile computed around that specific timestamp. This drives EQ strategy recommendations:
- **bass_swap**: Both tracks are bass-heavy at the transition point
- **filter_sweep**: Both tracks are treble-heavy, or mixed energy
- **simple_fade**: For cuts or when spectral data isn't available

### Compatibility Scoring (0-100)

Track pairs are scored across four dimensions:
- **Key** (40 pts max): Same key (40), relative major/minor (35), perfect fifth (30), perfect fourth (25), adjacent (15)
- **Tempo** (30 pts max): Linear falloff from 30 (same BPM) to 0 (at tolerance limit)
- **Energy** (15 pts max): Rewards smooth energy transitions (declining→rising), penalizes jarring ones (both peaking)
- **Sections** (15 pts max): outro→intro (15), breakdown→buildup (12), drop→intro (5)

Pairs scoring below 30 are marked incompatible and no transition is recommended.

### Phrase Detection

Mix points align with 16/32 bar phrase boundaries:
- **DJ-quality**: Matches how professional DJs mix
- **Musical**: Aligns with track structure (intros/outros)
- **Smart**: Prefers 32-bar phrases, falls back to 16-bar
- **Tolerance-based**: Uses ±50ms matching instead of exact float comparison to handle numerical drift

### Pair Optimization

When two tracks are compatible, the tool cross-references their mix candidates to find the best pair:
- Bonus for declining-out + rising-in energy gradient
- Bonus for outro→intro section pairing
- Spectral blend scoring rewards complementary frequency profiles

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

## Configuration

All scoring weights and thresholds are configurable via `AnalysisConfig`:

```python
from automix.config import AnalysisConfig
from automix.analyzer import AudioAnalyzer

config = AnalysisConfig(
    mix_in_bars=16,
    mix_out_bars=32,
    tempo_tolerance=6.0,
    # Section detection
    section_low_energy_ratio=0.3,
    section_high_energy_ratio=0.7,
    outro_min_position=0.75,
    # Scoring weights
    mix_score_section_match=3.0,
    mix_score_phrase_32bar=2.0,
    compat_key_weight=40.0,
    compat_tempo_weight=30.0,
    compat_min_score=30.0,
    # ... see config.py for all options
)
analyzer = AudioAnalyzer(config=config)
```

## Requirements

- Python 3.8+
- librosa >= 0.10.0
- numpy >= 1.24.0
- click >= 8.1.0
- rich >= 13.0.0 (for waveform visualization)
- yt-dlp >= 2024.0.0 (for SoundCloud downloads)
- ffmpeg (required by yt-dlp for audio conversion)

## Development

Install development dependencies:

```bash
uv pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

The test suite covers:
- Scoring functions (phrase alignment, section matching, energy flow, key compatibility)
- Section detection edge cases (no intro, mid-track breakdown, flat energy, merge pass)
- Spectral analysis (bass-heavy, treble-heavy, white noise, local vs. global)
- Compatibility scoring (score breakdown, threshold behavior, energy shape effects)
- Cache versioning and invalidation
- CLI output and integration tests

Format and lint code with ruff:

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Lint and auto-fix issues
ruff check --fix .
```

## Architecture

```
automix/
├── analyzer.py      # Core analysis engine (beat, key, energy, sections, spectral)
├── scoring.py       # Pure scoring functions (no state, easy to test and tune)
├── models.py        # Data models (AnalysisResult, MixCandidate, Section, etc.)
├── config.py        # All thresholds and weights in one place
├── cache.py         # Version-aware result caching
├── cli.py           # Click CLI with text/JSON output
├── visualizer.py    # Terminal waveform renderer with sections and candidates
└── exceptions.py    # Custom exceptions
```

## Mix Point Calculation

Mix points are calculated using energy-aware, section-aware phrase detection:

- **Bar-Based Offsets**: Mix-in at 16 bars from start, mix-out at 32 bars before end
- **Tempo-Adaptive**: Offsets scale with track tempo (faster tracks = shorter time, slower tracks = longer time)
- **Phrase Boundaries**: Mix points align with 16/32 bar phrase boundaries using tolerance-based matching
- **Section-Aware**: Prefers intro/outro sections, avoids drops for mix-in
- **Energy-Aware**: Prefers rising energy for mix-in, falling for mix-out
- **Multiple Candidates**: Top 3 candidates shown with scores so the DJ can choose
- **Intelligent Fallback**: Uses 16-bar phrases for shorter tracks

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
- **Tracks with no intro (DJ edits)**: First boundary used as intro_end so mix-in logic still works
- **Mid-track breakdowns**: Classified as "breakdown", not "outro" (outro requires ≥75% track position)
- **Flat energy tracks (ambient)**: Section detection produces valid output without crashing
- **Stale cache entries**: Automatically invalidated when analysis schema version changes

## License

MIT
