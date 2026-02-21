"""Result caching for AutoMix AI."""

import hashlib
import pickle
from pathlib import Path
from typing import Optional

from .logging_config import get_logger
from .models import AnalysisResult

logger = get_logger(__name__)

# Bump this when AnalysisResult schema changes to invalidate stale cache
CACHE_VERSION = 2


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
        """Calculate MD5 hash of file content."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _cache_key_hash(self, file_path: str, cache_key: Optional[str]) -> str:
        if cache_key:
            return hashlib.md5(cache_key.encode()).hexdigest()
        return self._get_file_hash(file_path)

    def get(self, file_path: str, cache_key: Optional[str] = None) -> Optional[AnalysisResult]:
        """Get cached result, returning None if missing or stale."""
        try:
            key_hash = self._cache_key_hash(file_path, cache_key)
            cache_file = self.cache_dir / f"{key_hash}.pkl"

            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)

                # Invalidate stale cache entries
                if not isinstance(result, AnalysisResult):
                    logger.debug("Cache stale (not AnalysisResult): %s", cache_key or file_path)
                    cache_file.unlink(missing_ok=True)
                    return None
                if getattr(result, "version", 0) < CACHE_VERSION:
                    logger.debug(
                        "Cache stale (version %s < %s): %s",
                        getattr(result, "version", 0),
                        CACHE_VERSION,
                        cache_key or file_path,
                    )
                    cache_file.unlink(missing_ok=True)
                    return None

                logger.debug("Cache hit: %s", cache_key or file_path)
                return result

            logger.debug("Cache miss: %s", cache_key or file_path)
            return None
        except Exception as e:
            logger.warning("Cache read error: %s", e)
            return None

    def set(self, file_path: str, result: AnalysisResult, cache_key: Optional[str] = None):
        """Cache result for file."""
        try:
            key_hash = self._cache_key_hash(file_path, cache_key)
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
