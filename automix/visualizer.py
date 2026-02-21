"""Rich terminal waveform renderer for audio analysis results."""

import os
from typing import List

import librosa
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from automix.models import AnalysisResult

BLOCKS = " ▁▂▃▄▅▆▇█"
COLORS = ["blue", "cyan", "green", "yellow", "red"]


def _amplitude_color(level: float) -> str:
    """Map normalized amplitude (0-1) to a color name."""
    idx = min(int(level * len(COLORS)), len(COLORS) - 1)
    return COLORS[idx]


def _format_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def _compute_envelope(file_path_or_y, width: int, sr: int = None) -> tuple:
    """Compute RMS envelope downsampled to width columns.

    Args:
        file_path_or_y: Path to audio file (str) or pre-loaded numpy array.
        width: Number of columns for the envelope.
        sr: Sample rate (required when passing numpy array).

    Returns (envelope as list of floats 0-1, duration in seconds).
    """
    if isinstance(file_path_or_y, str):
        y, sr = librosa.load(file_path_or_y, sr=None, mono=True)
    else:
        y = file_path_or_y
    duration = float(len(y)) / sr
    hop = max(1, len(y) // width)
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    # Downsample/pad to exactly width
    if len(rms) > width:
        indices = np.linspace(0, len(rms) - 1, width, dtype=int)
        rms = rms[indices]
    elif len(rms) < width:
        rms = np.pad(rms, (0, width - len(rms)))
    peak = rms.max()
    if peak > 0:
        rms = rms / peak
    return rms.tolist(), duration


def _build_waveform_line(envelope: List[float]) -> Text:
    """Build a Rich Text line of colored Unicode block characters."""
    text = Text()
    for amp in envelope:
        idx = min(int(amp * (len(BLOCKS) - 1)), len(BLOCKS) - 1)
        text.append(BLOCKS[idx], style=_amplitude_color(amp))
    return text


def _build_marker_line(duration: float, width: int, mix_in: float, mix_out: float) -> Text:
    """Build marker line with mix-in/mix-out indicators."""
    in_pos = min(int(mix_in / duration * width), width - 1) if duration > 0 else 0
    out_pos = min(int(mix_out / duration * width), width - 1) if duration > 0 else width - 1

    in_label = f"▼ Mix-in {_format_time(mix_in)}"
    out_label = f"Mix-out {_format_time(mix_out)} ▼"

    # Place mix-in label
    line = Text(" " * width)
    end = min(in_pos + len(in_label), width)
    for i, ch in enumerate(in_label):
        pos = in_pos + i
        if pos < width:
            line.plain = line.plain[:pos] + ch + line.plain[pos + 1 :]
    line.stylize("green", in_pos, end)

    # Place mix-out label
    out_start = max(out_pos - len(out_label) + 1, 0)
    for i, ch in enumerate(out_label):
        pos = out_start + i
        if pos < width:
            line.plain = line.plain[:pos] + ch + line.plain[pos + 1 :]
    line.stylize("red", out_start, out_start + len(out_label))

    return line


def _build_timeline(duration: float, width: int) -> Text:
    """Build a timeline ruler with minute markers."""
    line = Text(" " * width)
    if duration <= 0:
        return line
    t = 0.0
    while t <= duration:
        pos = int(t / duration * (width - 1))
        label = _format_time(t)
        for i, ch in enumerate(label):
            p = pos + i
            if p < width:
                line.plain = line.plain[:p] + ch + line.plain[p + 1 :]
        t += 60.0
    line.stylize("dim", 0, width)
    return line


def _build_section_line(duration: float, width: int, sections) -> Text:
    """Build a section label line showing detected sections."""
    if not sections:
        return Text()
    line = Text(" " * width)
    colors = {
        "intro": "cyan",
        "buildup": "yellow",
        "drop": "red",
        "breakdown": "magenta",
        "outro": "blue",
    }
    for s in sections:
        start_pos = min(int(s.start / duration * width), width - 1) if duration > 0 else 0
        end_pos = min(int(s.end / duration * width), width) if duration > 0 else width
        label = s.type[:5].upper()
        mid = start_pos + (end_pos - start_pos) // 2 - len(label) // 2
        mid = max(start_pos, min(mid, width - len(label)))
        for i, ch in enumerate(label):
            pos = mid + i
            if 0 <= pos < width:
                line.plain = line.plain[:pos] + ch + line.plain[pos + 1 :]
        color = colors.get(s.type, "white")
        line.stylize(color, mid, min(mid + len(label), width))
    return line


def _build_candidate_line(duration: float, width: int, candidates, label: str, color: str) -> Text:
    """Build a line showing top 3 candidate positions."""
    if not candidates:
        return Text()
    line = Text(" " * width)
    for i, c in enumerate(candidates[:3], 1):
        pos = min(int(c.time / duration * width), width - 1) if duration > 0 else 0
        marker = f"{i}"
        if 0 <= pos < width:
            line.plain = line.plain[:pos] + marker + line.plain[pos + 1 :]
            line.stylize(f"bold {color}", pos, pos + 1)
    return line


def render_waveform(file_path: str, result: AnalysisResult, width: int = 70,
                    audio_data: tuple = None) -> None:
    """Render a waveform visualization of an audio file to the terminal.

    Args:
        file_path: Path to the audio file (used for title; also loaded if audio_data not provided).
        result: AnalysisResult from analysis.
        width: Character width of the waveform display.
        audio_data: Optional (y, sr) tuple to avoid reloading the file.
    """
    console = Console()
    if audio_data is not None:
        y, sr = audio_data
        envelope, duration = _compute_envelope(y, width, sr=sr)
    else:
        envelope, duration = _compute_envelope(file_path, width)

    # Header info
    bpm_conf = f"{result.bpm_confidence:.2f}"
    key_conf = f"{result.key_confidence:.2f}"
    energy_str = ""
    if result.energy_profile is not None:
        energy_str = f"  Energy: {result.energy_profile.overall_energy:.1f}/10"
    header = Text(f"BPM: {result.bpm_str} ({bpm_conf})  Key: {result.key} ({key_conf}){energy_str}")

    # Build lines
    waveform = _build_waveform_line(envelope)
    markers = _build_marker_line(duration, width, result.mix_in_point, result.mix_out_point)
    timeline = _build_timeline(duration, width)

    # Section and candidate overlays (Phase 8)
    sections = result.energy_profile.sections if result.energy_profile else None
    section_line = _build_section_line(duration, width, sections)
    in_cand_line = _build_candidate_line(duration, width, result.mix_in_candidates, "in", "green")
    out_cand_line = _build_candidate_line(duration, width, result.mix_out_candidates, "out", "red")

    # Combine into panel content
    content = Text()
    content.append_text(header)
    content.append("\n\n")
    if section_line.plain.strip():
        content.append_text(section_line)
        content.append("\n")
    content.append_text(waveform)
    content.append("\n")
    content.append_text(markers)
    content.append("\n")
    if in_cand_line.plain.strip() or out_cand_line.plain.strip():
        # Merge candidate lines into one
        merged = Text(" " * width)
        for line_text, color in [(in_cand_line, "green"), (out_cand_line, "red")]:
            for i, ch in enumerate(line_text.plain):
                if ch != " " and i < width:
                    merged.plain = merged.plain[:i] + ch + merged.plain[i + 1 :]
                    merged.stylize(f"bold {color}", i, i + 1)
        content.append_text(merged)
        content.append("  ← candidates (in=green, out=red)\n")
    content.append_text(timeline)

    title = os.path.basename(file_path)
    panel = Panel(content, title=title, expand=False)
    console.print(panel)
