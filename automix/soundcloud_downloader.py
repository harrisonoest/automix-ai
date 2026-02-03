"""SoundCloud track downloader using yt-dlp."""

from pathlib import Path
import yt_dlp


def download_track(url: str, output_dir: str = ".", client_id: str = None) -> str:
    """Download a SoundCloud track and return the path to the MP3 file.

    Args:
        url: SoundCloud track URL
        output_dir: Directory to save the downloaded file (default: current directory)
        client_id: SoundCloud client ID (optional, unused with yt-dlp)

    Returns:
        Path to the downloaded MP3 file

    Raises:
        ValueError: If the URL is invalid or track cannot be found
        RuntimeError: If download fails
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_template = str(output_path / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp3"
            return filename
    except Exception as e:
        raise RuntimeError(f"Failed to download track: {e}")
