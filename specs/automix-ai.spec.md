# AutoMix AI Specification

## Summary
A Python CLI tool that analyzes audio files to identify optimal DJ mix transition points using beat detection, harmonic analysis, and structural segmentation.

## Acceptance Criteria

### Given: A valid audio file path
**When:** User runs `automix analyze <audio_file>`
**Then:**
- Audio file is loaded and analyzed
- Beat positions and tempo (BPM) are detected
- Harmonic key is identified
- Structural segments (intro, verse, chorus, outro) are detected
- Optimal mix-in and mix-out points are identified
- Results are displayed showing:
  - BPM
  - Detected key
  - Mix-in point (timestamp)
  - Mix-out point (timestamp)
  - Confidence scores for each metric

### Given: An unsupported or missing audio file
**When:** User runs `automix analyze <invalid_file>`
**Then:**
- Error message is displayed: "Error: Unable to load audio file"
- Exit code is non-zero

### Given: Multiple audio files with compatible keys/tempo
**When:** User runs `automix analyze file1.mp3 file2.mp3`
**Then:**
- Each file is analyzed sequentially
- Results for each file are displayed separately
- Compatible mix pairs are suggested based on key and tempo compatibility rules

### Given: Multiple audio files with no compatible pairs
**When:** User runs `automix analyze file1.mp3 file2.mp3` where files are incompatible
**Then:**
- Each file is analyzed and displayed
- Message displayed: "No compatible mix pairs found"

### Given: User wants JSON output
**When:** User runs `automix analyze <audio_file> --format json`
**Then:**
- Results are output as valid JSON with structure:
```json
{
  "file": "path/to/file.mp3",
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

## Input/Output Examples

### Example 1: Single file analysis
```bash
$ automix analyze track.mp3

Analyzing: track.mp3
BPM: 128.5 (confidence: 0.95)
Key: Am (confidence: 0.87)
Mix-in point: 0:15.2
Mix-out point: 4:05.8
```

### Example 2: Multiple files with compatibility
```bash
$ automix analyze track1.mp3 track2.mp3

Analyzing: track1.mp3
BPM: 128.5, Key: Am
Mix-in: 0:15.2, Mix-out: 4:05.8

Analyzing: track2.mp3
BPM: 130.0, Key: C
Mix-in: 0:08.5, Mix-out: 3:45.2

Compatible pairs:
✓ track1.mp3 → track2.mp3 (key: relative major, tempo: +1.5 BPM)
```

### Example 3: No compatible pairs
```bash
$ automix analyze track1.mp3 track2.mp3

Analyzing: track1.mp3
BPM: 128.5, Key: Am
Mix-in: 0:15.2, Mix-out: 4:05.8

Analyzing: track2.mp3
BPM: 95.0, Key: F#
Mix-in: 0:08.5, Mix-out: 3:45.2

No compatible mix pairs found
```

## Edge Cases

1. **Very short audio files (<10 seconds)**: Display warning "File too short for reliable analysis"
2. **Audio with no clear beat**: Report BPM as "Unknown" with confidence 0.0
3. **Corrupted audio files**: Display "Error: Unable to decode audio file"
4. **Empty file path**: Display usage help

## Key Compatibility Rules

Keys are compatible for mixing if they meet ANY of these criteria:
- **Same key**: Exact match (e.g., Am → Am)
- **Relative major/minor**: Share same notes (e.g., Am → C, C → Am)
- **Perfect fifth**: +7 semitones (e.g., Am → Em, C → G)
- **Perfect fourth**: -5 semitones (e.g., Am → Dm, C → F)
- **Adjacent keys**: ±1 semitone (e.g., Am → A#m, C → C#)

**Key notation**: Use traditional notation (C, C#, D, D#, E, F, F#, G, G#, A, A#, B) with major (no suffix) or minor (m suffix). Examples: C, Am, F#, Bbm.

## Tempo Compatibility Rules

Tracks are tempo-compatible if:
- BPM difference is ≤6 BPM (±4.7% at 128 BPM)

## Mix Point Detection Criteria

**Mix-in point** (where to start mixing into this track):
- First strong beat after intro/silence ends
- Minimum 5 seconds from track start
- Occurs at a structural boundary (start of verse/chorus)

**Mix-out point** (where to start mixing out of this track):
- Last strong beat before outro/fadeout begins
- Minimum 10 seconds before track end
- Occurs at a structural boundary (end of chorus/bridge)

## Non-Functional Requirements

- **Performance**: Analysis should complete within 10 seconds for a 5-minute audio file
- **Supported formats**: MP3, WAV, FLAC, OGG
- **Dependencies**: librosa, numpy, click (for CLI)
- **Python version**: 3.8+

## Out of Scope

- Real-time audio processing
- Audio playback functionality
- Automatic mixing/crossfading
- GUI interface
- Batch processing with parallel execution
- Audio file conversion
- Waveform visualization
