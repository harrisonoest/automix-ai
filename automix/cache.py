"""Result caching for AutoMix AI."""

import hashlib
import pickle
from pathlib import Path
from typing import Optional

from .logging_config import get_logger
from .models import AnalysisResult

logger = get_logger(__name__)


class ResultCache:
    """Cache for analysis results."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ./.automix/cache
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / ".automix" / "cache"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Cache directory: %s", self.cache_dir)

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file content.

        Args:
            file_path: Path to file.

        Returns:
            MD5 hash as hex string.
        """
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get(self, file_path: str, cache_key: Optional[str] = None) -> Optional[AnalysisResult]:
        """Get cached result for file.

        Args:
            file_path: Path to audio file (used for hashing if cache_key not provided).
            cache_key: Optional custom cache key (e.g., URL for downloaded tracks).

        Returns:
            Cached AnalysisResult or None if not cached.
        """
        try:
            if cache_key:
                # Use custom key (e.g., URL)
                key_hash = hashlib.md5(cache_key.encode()).hexdigest()
            else:
                # Use file content hash
                key_hash = self._get_file_hash(file_path)

            cache_file = self.cache_dir / f"{key_hash}.pkl"

            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                logger.debug("Cache hit: %s", cache_key or file_path)
                return result

            logger.debug("Cache miss: %s", cache_key or file_path)
            return None
        except Exception as e:
            logger.warning("Cache read error: %s", e)
            return None

    def set(self, file_path: str, result: AnalysisResult, cache_key: Optional[str] = None):
        """Cache result for file.

        Args:
            file_path: Path to audio file (used for hashing if cache_key not provided).
            result: Analysis result to cache.
            cache_key: Optional custom cache key (e.g., URL for downloaded tracks).
        """
        try:
            if cache_key:
                # Use custom key (e.g., URL)
                key_hash = hashlib.md5(cache_key.encode()).hexdigest()
            else:
                # Use file content hash
                key_hash = self._get_file_hash(file_path)

            cache_file = self.cache_dir / f"{key_hash}.pkl"

            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            logger.debug("Cached result: %s", cache_key or file_path)
        except Exception as e:
            logger.warning("Cache write error: %s", e)

    def clear(self):
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")
