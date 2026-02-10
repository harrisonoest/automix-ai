"""Command-line interface for AutoMix AI audio analysis."""

import json
import sys
import tempfile
from itertools import islice
from pathlib import Path

import click
from soundcloud import SoundCloud

from .analyzer import AudioAnalyzer
from .exceptions import AudioLoadError, AudioTooShortError
from .logging_config import setup_logging
from .soundcloud_downloader import download_track


def format_time(seconds):
    """Formats seconds into MM:SS.S format.

    Args:
        seconds: Time in seconds, or None.

    Returns:
        str: Formatted time string or "N/A" if None.

    Example:
        >>> format_time(125.3)
        '2:05.3'
    """
    if seconds is None:
        return "N/A"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:04.1f}"


def _get_energy(result):
    """Extract energy value from a result dict, or None."""
    model = result.get("_model")
    if model and getattr(model, "energy_profile", None):
        return model.energy_profile.overall_energy
    return None


def _format_transition_line(compat, model1, model2):
    """Format transition recommendation as indented text line."""
    if not compat.transition:
        return None
    t = compat.transition
    e1 = model1.energy_profile.overall_energy if getattr(model1, "energy_profile", None) else None
    e2 = model2.energy_profile.overall_energy if getattr(model2, "energy_profile", None) else None
    line = f"  Transition: {t.transition_type} over {t.mix_duration_bars} bars, {t.eq_strategy}"
    if e1 is not None and e2 is not None:
        line += f", energy: {e1:.1f} → {e2:.1f}"
    return line


def _transition_dict(compat):
    """Convert transition recommendation to JSON-serializable dict."""
    if not compat.transition:
        return None
    t = compat.transition
    return {
        "mix_duration_bars": t.mix_duration_bars,
        "transition_type": t.transition_type,
        "eq_strategy": t.eq_strategy,
    }


def format_analysis_result(result, name_key="file"):
    """Format a single analysis result for text output.

    Args:
        result: Analysis result dict
        name_key: Key to use for track name (default: "file")

    Returns:
        str: Formatted analysis output
    """
    lines = []
    track_name = result.get(name_key, result.get("file", "Unknown"))

    # For search results with artist/title
    if "artist" in result and "title" in result:
        lines.append(f"{result['artist']} - {result['title']}")
        if "url" in result:
            lines.append(f"URL: {result['url']}")
    else:
        lines.append(f"Analyzing: {track_name}")

    if result.get("warning"):
        lines.append(result["warning"])

    bpm_display = result.get("bpm_str", f"{result['bpm']:.1f}" if result["bpm"] else "Unknown")
    lines.append(f"BPM: {bpm_display} (confidence: {result['confidence']['bpm']:.2f})")
    lines.append(f"Key: {result['key']} (confidence: {result['confidence']['key']:.2f})")
    model = result.get("_model")
    if model and getattr(model, "energy_profile", None):
        lines.append(f"Energy: {model.energy_profile.overall_energy:.1f}/10")
    lines.append(f"Mix-in point: {format_time(result['mix_in_point'])}")
    lines.append(f"Mix-out point: {format_time(result['mix_out_point'])}")

    return "\n".join(lines)


def format_compatibility_output(results, analyzer, name_formatter=None):
    """Format compatibility pairs for text output.

    Args:
        results: List of analysis results
        analyzer: AudioAnalyzer instance
        name_formatter: Optional function to format track names (default: uses "file" key)

    Returns:
        str: Formatted compatibility output or empty string if no pairs
    """
    if len(results) < 2:
        return ""

    lines = []
    compatible_found = False

    for i in range(len(results) - 1):
        for j in range(i + 1, len(results)):
            compat = analyzer.check_compatibility(results[i], results[j])
            if compat:
                if not compatible_found:
                    lines.append("Compatible pairs:")
                    compatible_found = True

                tempo_sign = "+" if compat["tempo_diff"] >= 0 else ""
                tempo_info = f"{tempo_sign}{compat['tempo_diff']:.1f} BPM"

                if name_formatter:
                    name1 = name_formatter(results[i])
                    name2 = name_formatter(results[j])
                else:
                    name1 = results[i].get("file", "Unknown")
                    name2 = results[j].get("file", "Unknown")

                lines.append(
                    f"✓ {name1} → {name2} (key: {compat['key_reason']}, tempo: {tempo_info})"
                )

    if not compatible_found:
        lines.append("No compatible mix pairs found")

    return "\n".join(lines)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, verbose):
    """AutoMix AI - Audio analysis for DJ mixing."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("audio_files", nargs=-1, required=True)
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json"]))
@click.pass_context
def analyze(ctx, audio_files, output_format):
    """Analyze audio files for DJ mixing parameters.

    Detects BPM, musical key, and optimal mix points. When multiple files
    are provided, also identifies compatible track pairs.

    Args:
        audio_files: One or more audio file paths to analyze.
        output_format: Output format ('text' or 'json').

    Example:
        automix analyze track1.mp3 track2.mp3 --format json
    """
    if len(audio_files) == 0:
        click.echo(ctx.get_help())
        sys.exit(1)

    analyzer = AudioAnalyzer()
    results = []

    for file_path in audio_files:
        if not Path(file_path).exists():
            click.echo("Error: Unable to load audio file", err=True)
            sys.exit(1)

        try:
            analysis = analyzer.analyze(file_path)
            # Convert model to dict for compatibility with existing code
            results.append(
                {
                    "file": file_path,
                    "bpm": analysis.bpm,
                    "bpm_str": analysis.bpm_str,
                    "key": analysis.key,
                    "mix_in_point": analysis.mix_in_point,
                    "mix_out_point": analysis.mix_out_point,
                    "confidence": {
                        "bpm": analysis.bpm_confidence,
                        "key": analysis.key_confidence,
                    },
                    "_model": analysis,  # Keep model for compatibility checks
                }
            )
        except AudioLoadError:
            click.echo("Error: Unable to load audio file", err=True)
            sys.exit(1)
        except AudioTooShortError:
            results.append(
                {
                    "file": file_path,
                    "warning": "File too short for reliable analysis",
                    "bpm": None,
                    "bpm_str": "Unknown",
                    "key": "None",
                    "mix_in_point": None,
                    "mix_out_point": None,
                    "confidence": {"bpm": 0.0, "key": 0.0},
                }
            )
        except Exception:
            click.echo("Error: Unable to decode audio file", err=True)
            sys.exit(1)

    if output_format == "json":
        if len(results) == 1:
            output = {
                "file": results[0]["file"],
                "bpm": results[0]["bpm"],
                "key": results[0]["key"],
                "energy": _get_energy(results[0]),
                "mix_in_point": results[0]["mix_in_point"],
                "mix_out_point": results[0]["mix_out_point"],
                "confidence": results[0]["confidence"],
            }
        else:
            tracks = [
                {
                    "file": r["file"],
                    "bpm": r["bpm"],
                    "key": r["key"],
                    "energy": _get_energy(r),
                    "mix_in_point": r["mix_in_point"],
                    "mix_out_point": r["mix_out_point"],
                    "confidence": r["confidence"],
                }
                for r in results
            ]
            compatible_pairs = []
            for i in range(len(results) - 1):
                for j in range(i + 1, len(results)):
                    if "_model" in results[i] and "_model" in results[j]:
                        compat = analyzer.check_compatibility(
                            results[i]["_model"], results[j]["_model"]
                        )
                        if compat:
                            pair = {
                                "track1": results[i]["file"],
                                "track2": results[j]["file"],
                                "compatible": True,
                                "tempo_diff": compat.tempo_diff,
                                "key_reason": compat.key_reason,
                                "transition": _transition_dict(compat),
                            }
                            compatible_pairs.append(pair)
            output = {"tracks": tracks, "compatible_pairs": compatible_pairs}
        click.echo(json.dumps(output, indent=2))
    else:
        for i, result in enumerate(results):
            if i > 0:
                click.echo()
            click.echo(format_analysis_result(result))

        if len(results) > 1:
            click.echo()
            compatible_found = False
            for i in range(len(results) - 1):
                for j in range(i + 1, len(results)):
                    # Use models if available, otherwise skip
                    if "_model" in results[i] and "_model" in results[j]:
                        model1 = results[i]["_model"]
                        model2 = results[j]["_model"]
                        compat = analyzer.check_compatibility(model1, model2)
                        if compat:
                            if not compatible_found:
                                click.echo("Compatible pairs:")
                                compatible_found = True
                            tempo_sign = "+" if compat.tempo_diff >= 0 else ""
                            tempo_info = f"{tempo_sign}{compat.tempo_diff:.1f} BPM"
                            click.echo(
                                f"✓ {results[i]['file']} → {results[j]['file']} "
                                f"(key: {compat.key_reason}, tempo: {tempo_info})"
                            )
                            transition_line = _format_transition_line(compat, model1, model2)
                            if transition_line:
                                click.echo(transition_line)

            if not compatible_found:
                click.echo("No compatible mix pairs found")


@cli.command()
@click.argument("queries", nargs=-1, required=True)
@click.option("--limit", default=1, type=int, help="Maximum results per query (default: 1)")
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json"]))
@click.option(
    "--client-id",
    envvar="SOUNDCLOUD_CLIENT_ID",
    help="SoundCloud client ID (auto-generated if not provided)",
)
@click.option(
    "--auth-token",
    envvar="SOUNDCLOUD_AUTH_TOKEN",
    help="SoundCloud OAuth token (optional, for authenticated requests)",
)
@click.option("--analyze", is_flag=True, help="Download and analyze tracks")
def search(queries, limit, output_format, client_id, auth_token, analyze):
    """Search for tracks on SoundCloud.

    Args:
        queries: One or more search query strings.
        limit: Maximum number of results per query.
        output_format: Output format ('text' or 'json').
        client_id: SoundCloud client ID (optional).
        auth_token: SoundCloud OAuth token (optional).
        analyze: Download and analyze tracks (optional).

    Example:
        automix search "deep house" --limit 5
        automix search "track 1" "track 2" --analyze
    """
    try:
        sc = SoundCloud(client_id=client_id, auth_token=auth_token)

        # Collect tracks from all queries
        all_tracks = []
        for query in queries:
            tracks = list(islice(sc.search_tracks(query), limit))
            all_tracks.extend(tracks)

        if not all_tracks:
            click.echo("No tracks found")
            return

        if analyze:
            analyzer = AudioAnalyzer()
            analysis_results = []

            with tempfile.TemporaryDirectory() as tmpdir:
                for track in all_tracks:
                    try:
                        # Check cache first before downloading
                        if analyzer.cache:
                            cached = analyzer.cache.get("", cache_key=track.permalink_url)
                            if cached:
                                click.echo(
                                    f"Using cached analysis: {track.user.username} - {track.title}",
                                    err=True,
                                )
                                analysis_results.append(
                                    {
                                        "title": track.title,
                                        "artist": track.user.username,
                                        "url": track.permalink_url,
                                        "bpm": cached.bpm,
                                        "bpm_str": cached.bpm_str,
                                        "key": cached.key,
                                        "mix_in_point": cached.mix_in_point,
                                        "mix_out_point": cached.mix_out_point,
                                        "confidence": {
                                            "bpm": cached.bpm_confidence,
                                            "key": cached.key_confidence,
                                        },
                                        "_model": cached,
                                    }
                                )
                                continue

                        click.echo(f"Downloading: {track.user.username} - {track.title}", err=True)
                        file_path = download_track(track.permalink_url, tmpdir, client_id=client_id)

                        click.echo(f"Analyzing: {track.user.username} - {track.title}", err=True)
                        # Use track URL as cache key so caching works across temp directories
                        analysis = analyzer.analyze(file_path, cache_key=track.permalink_url)
                        analysis_results.append(
                            {
                                "title": track.title,
                                "artist": track.user.username,
                                "url": track.permalink_url,
                                "bpm": analysis.bpm,
                                "bpm_str": analysis.bpm_str,
                                "key": analysis.key,
                                "mix_in_point": analysis.mix_in_point,
                                "mix_out_point": analysis.mix_out_point,
                                "confidence": {
                                    "bpm": analysis.bpm_confidence,
                                    "key": analysis.key_confidence,
                                },
                                "_model": analysis,
                            }
                        )
                    except (AudioLoadError, AudioTooShortError) as e:
                        click.echo(f"Skipping {track.title}: {str(e)}", err=True)
                    except Exception as e:
                        click.echo(f"Error processing {track.title}: {e}", err=True)

            if output_format == "json":
                tracks = [
                    {
                        "title": r["title"],
                        "artist": r["artist"],
                        "url": r["url"],
                        "bpm": r["bpm"],
                        "key": r["key"],
                        "energy": _get_energy(r),
                        "mix_in_point": r["mix_in_point"],
                        "mix_out_point": r["mix_out_point"],
                        "confidence": r["confidence"],
                    }
                    for r in analysis_results
                ]
                if len(analysis_results) > 1:
                    compatible_pairs = []
                    for i in range(len(analysis_results) - 1):
                        for j in range(i + 1, len(analysis_results)):
                            if "_model" in analysis_results[i] and "_model" in analysis_results[j]:
                                compat = analyzer.check_compatibility(
                                    analysis_results[i]["_model"],
                                    analysis_results[j]["_model"],
                                )
                                if compat:
                                    r1, r2 = analysis_results[i], analysis_results[j]
                                    compatible_pairs.append(
                                        {
                                            "track1": f"{r1['artist']} - {r1['title']}",
                                            "track2": f"{r2['artist']} - {r2['title']}",
                                            "compatible": True,
                                            "tempo_diff": compat.tempo_diff,
                                            "key_reason": compat.key_reason,
                                            "transition": _transition_dict(compat),
                                        }
                                    )
                    output = {"tracks": tracks, "compatible_pairs": compatible_pairs}
                else:
                    output = tracks
                click.echo(json.dumps(output, indent=2))
            else:
                for i, result in enumerate(analysis_results):
                    if i > 0:
                        click.echo()
                    click.echo(format_analysis_result(result))

                if len(analysis_results) > 1:
                    click.echo()
                    compatible_found = False
                    for i in range(len(analysis_results) - 1):
                        for j in range(i + 1, len(analysis_results)):
                            if "_model" in analysis_results[i] and "_model" in analysis_results[j]:
                                compat = analyzer.check_compatibility(
                                    analysis_results[i]["_model"], analysis_results[j]["_model"]
                                )
                                if compat:
                                    if not compatible_found:
                                        click.echo("Compatible pairs:")
                                        compatible_found = True
                                    tempo_sign = "+" if compat.tempo_diff >= 0 else ""
                                    tempo_info = f"{tempo_sign}{compat.tempo_diff:.1f} BPM"
                                    r1 = analysis_results[i]
                                    r2 = analysis_results[j]
                                    track1 = f"{r1['artist']} - {r1['title']}"
                                    track2 = f"{r2['artist']} - {r2['title']}"
                                    click.echo(
                                        f"✓ {track1} → {track2} "
                                        f"(key: {compat.key_reason}, tempo: {tempo_info})"
                                    )
                                    transition_line = _format_transition_line(
                                        compat,
                                        analysis_results[i]["_model"],
                                        analysis_results[j]["_model"],
                                    )
                                    if transition_line:
                                        click.echo(transition_line)

                    if not compatible_found:
                        click.echo("No compatible mix pairs found")
        else:
            if output_format == "json":
                results = [
                    {
                        "title": track.title,
                        "artist": track.user.username,
                        "duration": track.duration,
                        "url": track.permalink_url,
                        "genre": track.genre,
                    }
                    for track in all_tracks
                ]
                click.echo(json.dumps(results, indent=2))
            else:
                click.echo(f"Found {len(all_tracks)} track(s):\n")
                for i, track in enumerate(all_tracks, 1):
                    duration_min = track.duration // 60000
                    duration_sec = (track.duration % 60000) // 1000
                    click.echo(f"{i}. {track.user.username} - {track.title}")
                    click.echo(f"   Duration: {duration_min}:{duration_sec:02d}")
                    if track.genre:
                        click.echo(f"   Genre: {track.genre}")
                    click.echo(f"   URL: {track.permalink_url}")
                    if i < len(all_tracks):
                        click.echo()
    except Exception as e:
        click.echo(f"Error: Unable to search SoundCloud - {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
