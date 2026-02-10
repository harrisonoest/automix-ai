# Pro DJ Mixing Specification

## Summary

Upgrade AutoMix to use energy analysis for smarter mix point selection and transition recommendations. Builds on the existing bar-based phrase alignment (already solid) by adding RMS energy profiling, energy-aware phrase boundary scoring, transition type suggestions, and track energy scoring for set building.

## Background

The current implementation detects BPM, key, and phrase-aligned mix points (16/32 bar boundaries). It works well for standard electronic music but treats all phrase boundaries equally. Professional DJs choose mix points based on energy characteristics — mixing out during breakdowns and mixing in during buildups. This spec adds that intelligence.

### Current State (What Exists)

- Bar-based mix point offsets (`mix_in_bars=16`, `mix_out_bars=32`) in `AnalysisConfig`
- Phrase boundary alignment at 16/32 bar boundaries in `_find_phrase_boundary()`
- BPM/key detection with real confidence scores
- Harmonic compatibility checking (Camelot Wheel rules)
- Tempo tolerance of ±6 BPM
- Result caching and SoundCloud integration

### What This Spec Adds

1. **Energy profiling** — RMS energy curve over time, per-phrase-boundary energy levels
2. **Energy-aware mix point selection** — score phrase boundaries, prefer breakdowns (mix-out) and buildups (mix-in)
3. **Transition recommendations** — mix duration, transition type, EQ strategy
4. **Track energy scoring** — overall energy level (1-10) for set building

## Feature 1: Energy Analysis

### Description

Compute an RMS energy profile over the track duration. Calculate energy values at each detected phrase boundary. Detect intro/outro sections by finding sustained low-energy regions at track start/end.

### Implementation

Add an `_analyze_energy()` method to `AudioAnalyzer` that returns an `EnergyProfile` dataclass:

```python
@dataclass
class EnergyProfile:
    """Energy analysis results for a track."""
    overall_energy: float          # 1-10 scale, average RMS normalized
    energy_at_boundaries: list     # [(time_sec, energy_normalized), ...] at each phrase boundary
    intro_end: float               # Time (seconds) where intro ends (first sustained energy rise)
    outro_start: float             # Time (seconds) where outro starts (last sustained energy drop)
```

**Algorithm:**
1. Compute RMS energy in frames using `librosa.feature.rms(y=y)`
2. Convert to time-series, normalize to 0.0–1.0 range
3. For each phrase boundary (from existing `_find_phrase_boundary` beat grid), sample the energy value
4. Detect intro end: first phrase boundary where energy exceeds 50% of track peak energy (sustained for ≥1 phrase)
5. Detect outro start: last phrase boundary where energy drops below 50% of track peak energy (sustained to end)
6. Overall energy: mean RMS mapped to 1–10 scale (1=ambient, 10=peak club)

### Acceptance Criteria

**Given** a valid audio file with clear intro/main/outro structure
**When** the file is analyzed
**Then** `EnergyProfile` is returned with:
- `overall_energy` between 1.0 and 10.0
- `energy_at_boundaries` contains one entry per detected phrase boundary
- `intro_end` ≤ track midpoint
- `outro_start` ≥ track midpoint
- `intro_end` < `outro_start`

**Given** a track with no clear intro (immediate high energy)
**When** the file is analyzed
**Then** `intro_end` equals 0.0 (no intro detected)

**Given** a track with no clear outro (abrupt ending)
**When** the file is analyzed
**Then** `outro_start` equals the track duration (no outro detected)

## Feature 2: Energy-Aware Mix Point Selection

### Description

Replace the current fixed-offset phrase boundary selection with energy-scored selection. Score each candidate phrase boundary and pick the best one for mix-in and mix-out.

### Scoring Rules

**Mix-out candidates** (phrase boundaries in the second half of the track):
- Prefer boundaries where energy is *decreasing* (breakdown) — score +2
- Prefer boundaries near `outro_start` from energy profile — score +1
- Penalize boundaries during peak energy sections — score -1
- Must be at least `mix_out_bars` before track end (existing constraint)

**Mix-in candidates** (phrase boundaries in the first half of the track):
- Prefer boundaries where energy is *increasing* (buildup) — score +2
- Prefer boundaries near `intro_end` from energy profile — score +1
- Penalize boundaries during silence/very low energy — score -1
- Must be at least `mix_in_bars` from track start (existing constraint)

Select the highest-scoring candidate. On tie, prefer the boundary closest to the current bar-based offset (preserving existing behavior as tiebreaker).

### Acceptance Criteria

**Given** a house track with 32-bar intro → buildup → drop → breakdown → outro
**When** analyzed with energy-aware selection
**Then** mix-in is at or near the buildup phrase boundary (not the silent intro)
**And** mix-out is at or near the breakdown phrase boundary (not during the drop)

**Given** a track where all phrase boundaries have similar energy
**When** analyzed with energy-aware selection
**Then** mix points fall back to the existing bar-based offset positions (backward compatible)

**Given** a very short track (<60 seconds) with few phrase boundaries
**When** analyzed with energy-aware selection
**Then** the best available boundary is selected without error

### Output Change

The `AnalysisResult` model gains an optional `energy` field:

```python
@dataclass
class AnalysisResult:
    bpm: Optional[float]
    key: str
    mix_in_point: float
    mix_out_point: float
    bpm_confidence: float
    key_confidence: float
    energy: Optional[EnergyProfile] = None  # NEW
```

## Feature 3: Mix Transition Recommendations

### Description

When two tracks are checked for compatibility, recommend how to transition between them: mix duration (in bars), transition type, and EQ strategy.

### Transition Types

| Type | When to Use | Description |
|------|-------------|-------------|
| `blend` | Similar energy, compatible key | Gradual crossfade over 16-32 bars |
| `cut` | Large energy difference, or fast tempo | Quick swap over 1-4 bars |
| `echo` | Breakdown → buildup transition | Echo/delay on outgoing, fade in incoming |

### EQ Strategies

| Strategy | When to Use | Description |
|----------|-------------|-------------|
| `bass_swap` | Both tracks have strong bass | Cut bass on incoming, swap at drop |
| `filter_sweep` | Key-compatible blend | High-pass filter outgoing while low-pass incoming |
| `simple_fade` | Fallback / short mixes | Volume crossfade only |

### Decision Logic

```
mix_duration_bars:
  - If both tracks have energy > 5: 32 bars (long blend)
  - If energy difference > 3: 8 bars (quick transition)
  - Default: 16 bars

transition_type:
  - If energy difference ≤ 2 AND key compatible: "blend"
  - If energy difference > 4: "cut"
  - If mix-out is breakdown AND mix-in is buildup: "echo"
  - Default: "blend"

eq_strategy:
  - If transition_type == "cut": "simple_fade"
  - If transition_type == "echo": "filter_sweep"
  - If both tracks energy > 5: "bass_swap"
  - Default: "filter_sweep"
```

### Acceptance Criteria

**Given** two compatible tracks with similar energy (both 6-8)
**When** compatibility is checked
**Then** recommendation is: blend, 32 bars, bass_swap

**Given** two compatible tracks with large energy difference (3 vs 8)
**When** compatibility is checked
**Then** recommendation is: cut, 8 bars, simple_fade

**Given** track A mix-out at breakdown, track B mix-in at buildup, compatible key
**When** compatibility is checked
**Then** recommendation is: echo, 16 bars, filter_sweep

**Given** two tracks that are NOT compatible (key/tempo mismatch)
**When** compatibility is checked
**Then** no recommendation is returned (same as current behavior — returns None)

### Model Changes

```python
@dataclass
class TransitionRecommendation:
    """Recommended transition between two tracks."""
    mix_duration_bars: int          # 8, 16, or 32
    transition_type: str            # "blend", "cut", or "echo"
    eq_strategy: str                # "bass_swap", "filter_sweep", or "simple_fade"

@dataclass
class CompatibilityResult:
    compatible: bool
    tempo_diff: float
    key_reason: str
    transition: Optional[TransitionRecommendation] = None  # NEW
```

## Feature 4: Track Energy Scoring

### Description

Expose the overall energy score (1-10) in CLI output and JSON for set building. DJs use energy levels to plan set flow (e.g., build energy over time, create peaks and valleys).

### CLI Output Change

```
Analyzing: track.mp3
BPM: 128.5 (confidence: 0.92)
Key: Am (confidence: 0.85)
Energy: 7.2/10
Mix-in point: 0:32.0
Mix-out point: 4:08.0
```

### JSON Output Change

```json
{
  "file": "track.mp3",
  "bpm": 128.5,
  "key": "Am",
  "energy": 7.2,
  "mix_in_point": 32.0,
  "mix_out_point": 248.0,
  "confidence": {
    "bpm": 0.92,
    "key": 0.85
  }
}
```

### Compatibility Output Change

```
Compatible pairs:
✓ track1.mp3 → track2.mp3 (key: relative major, tempo: +1.5 BPM)
  Transition: blend over 32 bars, bass swap, energy: 7.2 → 6.8
```

### JSON Compatibility Output

```json
{
  "compatible": true,
  "tempo_diff": 1.5,
  "key_reason": "relative major",
  "transition": {
    "mix_duration_bars": 32,
    "transition_type": "blend",
    "eq_strategy": "bass_swap"
  }
}
```

### Acceptance Criteria

**Given** a track is analyzed
**When** output format is text
**Then** energy score is displayed as `Energy: X.X/10`

**Given** a track is analyzed
**When** output format is JSON
**Then** `energy` field is present as a float

**Given** two compatible tracks are analyzed
**When** compatibility is displayed in text
**Then** transition recommendation and energy flow are shown

**Given** two compatible tracks are analyzed
**When** compatibility is displayed in JSON
**Then** `transition` object is present in compatibility result

## Edge Cases

1. **Silent/near-silent track**: Energy profile is all zeros → `overall_energy` = 1.0, mix points fall back to bar-based offsets
2. **Constant energy (no dynamics)**: All phrase boundaries score equally → fall back to bar-based offsets (backward compatible)
3. **Very short track (<30s)**: Few phrase boundaries available → select best available, skip transition recommendation if only 1-2 boundaries exist
4. **Track with no detected beats**: Energy profile still computed from RMS, but mix points remain `None` (existing behavior)
5. **Cached results**: Existing cache entries without energy data should still work (energy field is Optional). Re-analysis populates energy on next uncached run.
6. **One track has energy, other doesn't**: Transition recommendation uses defaults (blend, 16 bars, filter_sweep)

## Implementation Order

1. **Energy profiling** (`EnergyProfile` model + `_analyze_energy()`) — no output changes, internal only
2. **Energy-aware mix point selection** — update `_find_phrase_boundary()` to accept energy scores
3. **Track energy in output** — add energy to CLI and JSON output
4. **Transition recommendations** — update `check_compatibility()` and output formatting

Each step is independently testable and deployable. Steps 1-2 change mix point quality. Steps 3-4 add new user-facing information.

## Out of Scope

- Real-time energy visualization or waveform display
- Automatic mixing/crossfading of audio files
- Genre detection or genre-specific tuning
- Spectral centroid or timbral analysis (future enhancement)
- User-configurable energy thresholds via CLI flags (use sensible defaults first)
