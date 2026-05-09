"""
PlusOne Unified Cache — LRU + TTL
===================================
Thread-safe, bounded in-memory cache shared by all hot API routes.

Time complexity:
  get()          O(1)  dict lookup + OrderedDict.move_to_end
  set()          O(1)  dict insert + O(1) LRU evict when full
  invalidate()   O(k)  k = matching keys (typically 1-5)
  invalidate_all O(1)  replace OrderedDict reference

Space complexity:
  O(max_size)   hard upper bound — LRU eviction prevents unbounded growth.

Design notes:
  - Combines TTL (time expiry) with LRU (size bound). Both policies apply.
  - collections.OrderedDict provides O(1) move_to_end() — the LRU primitive.
    Plain dict preserves insertion order in 3.7+ but lacks O(1) move_to_end.
  - time.monotonic() avoids wall-clock drift issues with time.time().
  - Single RLock guards all mutations; safe for multi-threaded FastAPI workers.
  - Background sweep every 60s proactively evicts expired keys so stale
    entries don't hold LRU slots after they will never be read again.
"""

import time
import threading
import logging
from collections import OrderedDict
from typing import Any, Optional, Tuple

log = logging.getLogger(__name__)

# Sentinel object distinguishes cache miss from a stored None value — O(1) identity check.
_MISS = object()

# Default TTLs (seconds)
TTL_UPCOMING = 15 * 60     # 15 min — upcoming fixtures list
TTL_PUBLIC   = 15 * 60     # 15 min — public predictions page
TTL_PREVIEW  = 30 * 60     # 30 min — per-match preview card
TTL_CONSENSUS = 30 * 60    # 30 min — per-match consensus prediction


class PlusOneCache:
    """
    LRU + TTL cache with O(1) get / O(1) set / O(1) evict.

    Eviction policy (two independent mechanisms):
      1. TTL   — checked lazily on every read; expired entries are deleted
                 immediately on access and proactively via background sweep.
      2. LRU   — when size reaches max_size, the least-recently-used entry
                 is evicted in O(1) via OrderedDict.popitem(last=False).

    Both policies apply; whichever fires first evicts the entry.
    """

    def __init__(self, max_size: int = 512, sweep_interval: int = 60):
        self._max_size        = max_size
        self._sweep_interval  = sweep_interval
        # OrderedDict: insertion order = LRU order (oldest at front)
        self._store: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock            = threading.RLock()
        self._hits            = 0
        self._misses          = 0

        # Daemon thread proactively removes expired keys every sweep_interval seconds.
        # This prevents stale entries from consuming LRU slots after the data they
        # represent is no longer valid (e.g. a match that has already been played).
        self._sweep = threading.Thread(
            target=self._background_sweep,
            daemon=True,
            name="plusone-cache-sweep",
        )
        self._sweep.start()

    # ── Core O(1) operations ─────────────────────────────────────────────────

    def get(self, key: str) -> Tuple[Any, bool]:
        """
        O(1): dict lookup + conditional move_to_end.
        Returns (value, True) on hit, (None, False) on miss or expiry.
        """
        with self._lock:
            entry = self._store.get(key, _MISS)
            if entry is _MISS:
                self._misses += 1
                return None, False
            value, expires_at = entry
            if expires_at <= time.monotonic():
                # Lazy TTL eviction — O(1)
                del self._store[key]
                self._misses += 1
                return None, False
            # LRU: promote to most-recently-used position — O(1)
            self._store.move_to_end(key)
            self._hits += 1
            return value, True

    def set(self, key: str, value: Any, ttl: float) -> None:
        """
        O(1): insert or update + O(1) LRU eviction if at capacity.
        ttl: seconds until expiry (uses monotonic clock).
        """
        expires_at = time.monotonic() + ttl
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = (value, expires_at)
            else:
                self._store[key] = (value, expires_at)
                if len(self._store) > self._max_size:
                    # Evict least-recently-used (front of OrderedDict) — O(1)
                    evicted, _ = self._store.popitem(last=False)
                    log.debug("Cache LRU evict: %s", evicted)

    def invalidate(self, prefix: str) -> int:
        """
        O(k): delete all keys starting with prefix.
        k is typically very small (e.g. all keys for one league = 1-3 keys).
        """
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
        if keys:
            log.debug("Cache invalidated %d keys (prefix=%r)", len(keys), prefix)
        return len(keys)

    def invalidate_all(self) -> int:
        """O(1): replace the OrderedDict reference entirely."""
        with self._lock:
            n = len(self._store)
            self._store = OrderedDict()
        log.info("Cache fully cleared (%d keys).", n)
        return n

    # ── Typed key helpers ────────────────────────────────────────────────────
    # Centralise key construction so routes never hard-code key strings.

    @staticmethod
    def _upcoming_key(league_id: Optional[int], limit: int) -> str:
        return f"upcoming:{league_id}:{limit}"

    @staticmethod
    def _public_key(page: int, per_page: int) -> str:
        return f"public:{page}:{per_page}"

    @staticmethod
    def _preview_key(home_id: int, away_id: int) -> str:
        # Canonical order: same key regardless of which side is "home"
        lo, hi = (home_id, away_id) if home_id <= away_id else (away_id, home_id)
        return f"preview:{lo}:{hi}"

    @staticmethod
    def _consensus_key(home_id: int, away_id: int, league_id: Optional[int]) -> str:
        return f"consensus:{home_id}:{away_id}:{league_id}"

    # ── Typed get/set wrappers ───────────────────────────────────────────────

    def get_upcoming(self, league_id, limit):
        return self.get(self._upcoming_key(league_id, limit))

    def set_upcoming(self, league_id, limit, value, ttl=TTL_UPCOMING):
        self.set(self._upcoming_key(league_id, limit), value, ttl)

    def get_public(self, page, per_page):
        return self.get(self._public_key(page, per_page))

    def set_public(self, page, per_page, value, ttl=TTL_PUBLIC):
        self.set(self._public_key(page, per_page), value, ttl)

    def get_preview(self, home_id, away_id):
        return self.get(self._preview_key(home_id, away_id))

    def set_preview(self, home_id, away_id, value, ttl=TTL_PREVIEW):
        self.set(self._preview_key(home_id, away_id), value, ttl)

    def get_consensus(self, home_id, away_id, league_id):
        return self.get(self._consensus_key(home_id, away_id, league_id))

    def set_consensus(self, home_id, away_id, league_id, value, ttl=TTL_CONSENSUS):
        self.set(self._consensus_key(home_id, away_id, league_id), value, ttl)

    # ── Observability ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            now   = time.monotonic()
            alive = sum(1 for _, (_, exp) in self._store.items() if exp > now)
        return {
            "total_keys":  len(self._store),
            "alive_keys":  alive,
            "max_size":    self._max_size,
            "hits":        self._hits,
            "misses":      self._misses,
            "hit_rate":    round(self._hits / total, 4) if total else 0.0,
        }

    # ── Background sweep ─────────────────────────────────────────────────────

    def _background_sweep(self):
        """
        Proactively evict expired entries every sweep_interval seconds.
        O(n) per sweep but runs in background — amortised O(1) cost per set().
        Prevents stale entries from occupying LRU slots when never re-read.
        """
        while True:
            time.sleep(self._sweep_interval)
            now = time.monotonic()
            with self._lock:
                expired = [k for k, (_, exp) in self._store.items() if exp <= now]
                for k in expired:
                    del self._store[k]
            if expired:
                log.debug("Cache sweep evicted %d expired keys.", len(expired))


# ── Module-level singleton ───────────────────────────────────────────────────
# All routes import this single instance. max_size=512 covers:
#  - up to 50 "upcoming" variants (league×limit combos)
#  - up to ~400 per-match consensus/preview entries (typical fixture list)
#  - headroom for public endpoint pages
_cache = PlusOneCache(max_size=512)
