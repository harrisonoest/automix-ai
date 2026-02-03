"""Command-line interface for AutoMix AI audio analysis."""

import json
import sys
import tempfile
from itertools import islice
from pathlib import Path

import click
from soundcloud import SoundCloud

from .analyzer import AudioAnalyzer
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


@click.group()
def cli():
    """AutoMix AI - Audio analysis for DJ mixing."""
    pass


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
            result = analyzer.analyze(file_path)
            result["file"] = file_path
            results.append(result)
        except ValueError as e:
            if "Unable to load audio file" in str(e):
                click.echo("Error: Unable to load audio file", err=True)
            else:
                click.echo("Error: Unable to decode audio file", err=True)
            sys.exit(1)
        except Exception:
            click.echo("Error: Unable to decode audio file", err=True)
            sys.exit(1)

    if output_format == "json":
        if len(results) == 1:
            output = {
                "file": results[0]["file"],
                "bpm": results[0]["bpm"],
                "key": results[0]["key"],
                "mix_in_point": results[0]["mix_in_point"],
                "mix_out_point": results[0]["mix_out_point"],
                "confidence": results[0]["confidence"],
            }
        else:
            output = [
                {
                    "file": r["file"],
                    "bpm": r["bpm"],
                    "key": r["key"],
                    "mix_in_point": r["mix_in_point"],
                    "mix_out_point": r["mix_out_point"],
                    "confidence": r["confidence"],
                }
                for r in results
            ]
        click.echo(json.dumps(output, indent=2))
    else:
        for i, result in enumerate(results):
            if i > 0:
                click.echo()

            click.echo(f"Analyzing: {result['file']}")

            if result.get("warning"):
                click.echo(result["warning"])

            bpm_display = (
                result["bpm_str"]
                if "bpm_str" in result
                else (f"{result['bpm']:.1f}" if result["bpm"] else "Unknown")
            )
            click.echo(f"BPM: {bpm_display} (confidence: {result['confidence']['bpm']:.2f})")
            click.echo(f"Key: {result['key']} (confidence: {result['confidence']['key']:.2f})")
            click.echo(f"Mix-in point: {format_time(result['mix_in_point'])}")
            click.echo(f"Mix-out point: {format_time(result['mix_out_point'])}")

        if len(results) > 1:
            click.echo()
            compatible_found = False
            for i in range(len(results) - 1):
                for j in range(i + 1, len(results)):
                    compat = analyzer.check_compatibility(results[i], results[j])
                    if compat:
                        if not compatible_found:
                            click.echo("Compatible pairs:")
                            compatible_found = True
                        tempo_sign = "+" if compat["tempo_diff"] >= 0 else ""
                        key_reason = compat["key_reason"]
                        tempo_info = f"{tempo_sign}{compat['tempo_diff']:.1f} BPM"
                        click.echo(
                            f"✓ {results[i]['file']} → {results[j]['file']} "
                            f"(key: {key_reason}, tempo: {tempo_info})"
                        )

            if not compatible_found:
                click.echo("No compatible mix pairs found")


@cli.command()
@click.argument("query")
@click.option("--limit", default=10, type=int, help="Maximum number of results (default: 10)")
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
def search(query, limit, output_format, client_id, auth_token, analyze):
    """Search for tracks on SoundCloud.

    Args:
        query: Search query string.
        limit: Maximum number of results to return.
        output_format: Output format ('text' or 'json').
        client_id: SoundCloud client ID (optional).
        auth_token: SoundCloud OAuth token (optional).
        analyze: Download and analyze tracks (optional).

    Example:
        automix search "deep house" --limit 5
        automix search "deep house" --limit 5 --analyze
    """
    try:
        sc = SoundCloud(client_id=client_id, auth_token=auth_token)
        tracks = list(islice(sc.search_tracks(query), limit))

        if not tracks:
            click.echo("No tracks found")
            return

        if analyze:
            analyzer = AudioAnalyzer()
            analysis_results = []

            with tempfile.TemporaryDirectory() as tmpdir:
                for track in tracks:
                    try:
                        click.echo(f"Downloading: {track.user.username} - {track.title}", err=True)
                        file_path = download_track(track.permalink_url, tmpdir, client_id=client_id)

                        click.echo(f"Analyzing: {track.user.username} - {track.title}", err=True)
                        result = analyzer.analyze(file_path)
                        result["title"] = track.title
                        result["artist"] = track.user.username
                        result["url"] = track.permalink_url
                        analysis_results.append(result)
                    except Exception as e:
                        click.echo(f"Error processing {track.title}: {e}", err=True)

            if output_format == "json":
                output = [
                    {
                        "title": r["title"],
                        "artist": r["artist"],
                        "url": r["url"],
                        "bpm": r["bpm"],
                        "key": r["key"],
                        "mix_in_point": r["mix_in_point"],
                        "mix_out_point": r["mix_out_point"],
                        "confidence": r["confidence"],
                    }
                    for r in analysis_results
                ]
                click.echo(json.dumps(output, indent=2))
            else:
                for i, result in enumerate(analysis_results):
                    if i > 0:
                        click.echo()

                    click.echo(f"{result['artist']} - {result['title']}")
                    click.echo(f"URL: {result['url']}")

                    bpm_display = (
                        result["bpm_str"]
                        if "bpm_str" in result
                        else (f"{result['bpm']:.1f}" if result["bpm"] else "Unknown")
                    )
                    bpm_conf = result["confidence"]["bpm"]
                    key_conf = result["confidence"]["key"]
                    click.echo(f"BPM: {bpm_display} (confidence: {bpm_conf:.2f})")
                    click.echo(f"Key: {result['key']} (confidence: {key_conf:.2f})")
                    click.echo(f"Mix-in point: {format_time(result['mix_in_point'])}")
                    click.echo(f"Mix-out point: {format_time(result['mix_out_point'])}")

                if len(analysis_results) > 1:
                    click.echo()
                    compatible_found = False
                    for i in range(len(analysis_results) - 1):
                        for j in range(i + 1, len(analysis_results)):
                            r1 = analysis_results[i]
                            r2 = analysis_results[j]
                            compat = analyzer.check_compatibility(r1, r2)
                            if compat:
                                if not compatible_found:
                                    click.echo("Compatible pairs:")
                                    compatible_found = True
                                tempo_sign = "+" if compat["tempo_diff"] >= 0 else ""
                                key_reason = compat["key_reason"]
                                tempo_info = f"{tempo_sign}{compat['tempo_diff']:.1f} BPM"
                                track1 = f"{r1['artist']} - {r1['title']}"
                                track2 = f"{r2['artist']} - {r2['title']}"
                                click.echo(
                                    f"✓ {track1} → {track2} "
                                    f"(key: {key_reason}, tempo: {tempo_info})"
                                )

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
                    for track in tracks
                ]
                click.echo(json.dumps(results, indent=2))
            else:
                click.echo(f"Found {len(tracks)} track(s):\n")
                for i, track in enumerate(tracks, 1):
                    duration_min = track.duration // 60000
                    duration_sec = (track.duration % 60000) // 1000
                    click.echo(f"{i}. {track.user.username} - {track.title}")
                    click.echo(f"   Duration: {duration_min}:{duration_sec:02d}")
                    if track.genre:
                        click.echo(f"   Genre: {track.genre}")
                    click.echo(f"   URL: {track.permalink_url}")
                    if i < len(tracks):
                        click.echo()
    except Exception as e:
        click.echo(f"Error: Unable to search SoundCloud - {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
