"""Intelligent caching system for AlphaGenome predictions."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, Any, Dict
import threading
from datetime import datetime, timedelta

from alphagenome.data import genome


class PredictionCache:
    """Thread-safe cache for AlphaGenome predictions.
    
    Features:
    - Disk-based persistent caching
    - LRU eviction policy
    - TTL (time-to-live) support
    - Thread-safe operations
    - Automatic cache size management
    
    Example:
        >>> cache = PredictionCache(".cache", max_size_gb=5.0, ttl_days=30)
        >>> key = cache.generate_key(variant, interval, ontology_terms)
        >>> cache.set(key, predictions)
        >>> cached = cache.get(key)
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache/alphagenome",
        max_size_gb: float = 10.0,
        ttl_days: int = 30,
    ):
        """Initialize cache.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
            ttl_days: Time-to-live for cache entries in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.ttl = timedelta(days=ttl_days)
        self._lock = threading.Lock()
        self._metadata_file = self.cache_dir / "metadata.json"
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self._metadata_file.exists():
            with open(self._metadata_file, 'r') as f:
                return json.load(f)
        return {"entries": {}, "total_size": 0}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        with open(self._metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2)
    
    def generate_key(
        self,
        variant: genome.Variant,
        interval: genome.Interval,
        ontology_terms: list,
    ) -> str:
        """Generate cache key from prediction parameters.
        
        Args:
            variant: Genomic variant
            interval: Genomic interval
            ontology_terms: Ontology terms list
            
        Returns:
            SHA256 hash as cache key
        """
        key_data = {
            "variant": {
                "chromosome": variant.chromosome,
                "position": variant.position,
                "ref": variant.reference_bases,
                "alt": variant.alternate_bases,
            },
            "interval": {
                "chromosome": interval.chromosome,
                "start": interval.start,
                "end": interval.end,
            },
            "ontology_terms": sorted(ontology_terms),
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached prediction.
        
        Args:
            key: Cache key
            
        Returns:
            Cached prediction or None if not found/expired
        """
        with self._lock:
            if key not in self._metadata["entries"]:
                return None
            
            entry = self._metadata["entries"][key]
            
            # Check TTL
            cached_time = datetime.fromisoformat(entry["timestamp"])
            if datetime.now() - cached_time > self.ttl:
                self._remove_entry(key)
                return None
            
            # Update access time for LRU
            entry["last_access"] = datetime.now().isoformat()
            self._save_metadata()
            
            # Load and return cached data
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Store prediction in cache.
        
        Args:
            key: Cache key
            value: Prediction to cache
        """
        with self._lock:
            # Serialize to file
            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            file_size = cache_file.stat().st_size
            now = datetime.now().isoformat()
            
            if key in self._metadata["entries"]:
                # Update existing entry
                old_size = self._metadata["entries"][key]["size"]
                self._metadata["total_size"] -= old_size
            
            self._metadata["entries"][key] = {
                "timestamp": now,
                "last_access": now,
                "size": file_size,
            }
            self._metadata["total_size"] += file_size
            
            # Enforce max size with LRU eviction
            self._enforce_size_limit()
            
            self._save_metadata()
    
    def _enforce_size_limit(self) -> None:
        """Evict old entries if cache exceeds size limit."""
        while self._metadata["total_size"] > self.max_size_bytes:
            # Find least recently accessed entry
            oldest_key = min(
                self._metadata["entries"].keys(),
                key=lambda k: self._metadata["entries"][k]["last_access"]
            )
            self._remove_entry(oldest_key)
    
    def _remove_entry(self, key: str) -> None:
        """Remove cache entry.
        
        Args:
            key: Cache key to remove
        """
        if key in self._metadata["entries"]:
            entry = self._metadata["entries"][key]
            self._metadata["total_size"] -= entry["size"]
            del self._metadata["entries"][key]
            
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            for key in list(self._metadata["entries"].keys()):
                self._remove_entry(key)
            self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            return {
                "entries": len(self._metadata["entries"]),
                "total_size_mb": self._metadata["total_size"] / (1024**2),
                "max_size_mb": self.max_size_bytes / (1024**2),
                "utilization": self._metadata["total_size"] / self.max_size_bytes,
            }