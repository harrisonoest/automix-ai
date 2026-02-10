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


def _compute_envelope(file_path: str, width: int) -> tuple:
    """Load audio and compute RMS envelope downsampled to width columns.

    Returns (envelope as list of floats 0-1, duration in seconds).
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)
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


def render_waveform(file_path: str, result: AnalysisResult, width: int = 70) -> None:
    """Render a waveform visualization of an audio file to the terminal.

    Args:
        file_path: Path to the audio file.
        result: AnalysisResult from analysis.
        width: Character width of the waveform display.
    """
    console = Console()
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

    # Combine into panel content
    content = Text()
    content.append_text(header)
    content.append("\n\n")
    content.append_text(waveform)
    content.append("\n")
    content.append_text(markers)
    content.append("\n")
    content.append_text(timeline)

    title = os.path.basename(file_path)
    panel = Panel(content, title=title, expand=False)
    console.print(panel)
