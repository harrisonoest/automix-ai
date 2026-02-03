"""SoundCloud track downloader using soundcloud-lib."""

from pathlib import Path

from sclib import SoundcloudAPI, Track


def download_track(url: str, output_dir: str = ".") -> str:
    """Download a SoundCloud track and return the path to the MP3 file.

    Args:
        url: SoundCloud track URL
        output_dir: Directory to save the downloaded file (default: current directory)

    Returns:
        Path to the downloaded MP3 file

    Raises:
        ValueError: If the URL is invalid or track cannot be found
        RuntimeError: If download fails
    """
    api = SoundcloudAPI()

    try:
        track = api.resolve(url)
    except Exception as e:
        raise ValueError(f"Failed to resolve SoundCloud URL: {e}")

    if not isinstance(track, Track):
        raise ValueError("URL does not point to a valid track")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        filename = track.write_mp3_to(output_path)
        return str(output_path / filename)
    except Exception as e:
        raise RuntimeError(f"Failed to download track: {e}")
