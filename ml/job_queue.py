"""
PlusOne Background Job Queue
==============================
Replaces all bare threading.Thread spawns in routes/sync.py and routes/markets.py
with a managed, priority-ordered, deduplicating queue.

Time complexity:
  enqueue()    O(log k)   k = queue depth (always < 10) ≈ O(1)
  dedup check  O(1)       set lookup on finite set of job types
  worker loop  O(log k)   PriorityQueue.get()

Space complexity:
  O(J)   where J = number of distinct job types (= 8, constant)
  Queue depth is bounded: deduplication ensures at most one pending
  entry per job type, so max queue depth = 8 items.

Architecture:
  - One persistent daemon worker thread handles heavy jobs serially:
      EVALUATE → RECALIBRATE_* → RETRAIN → WARM_CACHE
    Serial execution prevents resource contention (two parallel DC
    retrains would exhaust memory and CPU simultaneously).

  - LOG_* jobs (prediction logging to DB) bypass the main queue and
    go to a ThreadPoolExecutor(max_workers=4). This ensures DB inserts
    never block or delay heavy background jobs.

  - Deduplication: a `set` of currently-pending/running job types.
    Since the job type set is finite (8 elements), the set is O(1)
    in both time and space — effectively a constant-size registry.
"""

import queue
import threading
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)


# ── Job priorities (lower int = higher priority) ─────────────────────────────

class Priority(IntEnum):
    EVALUATE    = 0   # Grade completed matches — prerequisite for calibration
    RECALIBRATE = 1   # ML/DC/market calibrators — prerequisite for warming
    RETRAIN     = 2   # Full ML ensemble retrain — heavy, runs last
    WARM_CACHE  = 3   # Precompute predictions after retrain/sync
    LOG         = 4   # DB inserts — lowest priority, goes to separate pool


# ── Job type constants ────────────────────────────────────────────────────────

class JobType:
    EVALUATE_PREDICTIONS = "evaluate_predictions"
    RECALIBRATE_ML       = "recalibrate_ml"
    RECALIBRATE_DC       = "recalibrate_dc"
    RECALIBRATE_MARKETS  = "recalibrate_markets"
    RETRAIN_ML           = "retrain_ml"
    WARM_CACHE           = "warm_cache"
    LOG_PREDICTION       = "log_prediction"
    LOG_MARKETS          = "log_markets"


_JOB_PRIORITY: Dict[str, int] = {
    JobType.EVALUATE_PREDICTIONS: Priority.EVALUATE,
    JobType.RECALIBRATE_ML:       Priority.RECALIBRATE,
    JobType.RECALIBRATE_DC:       Priority.RECALIBRATE,
    JobType.RECALIBRATE_MARKETS:  Priority.RECALIBRATE,
    JobType.RETRAIN_ML:           Priority.RETRAIN,
    JobType.WARM_CACHE:           Priority.WARM_CACHE,
    JobType.LOG_PREDICTION:       Priority.LOG,
    JobType.LOG_MARKETS:          Priority.LOG,
}

_LOG_JOB_TYPES = frozenset({JobType.LOG_PREDICTION, JobType.LOG_MARKETS})


@dataclass(order=True)
class _Job:
    """Priority queue item. Ordered by (priority, seq) — FIFO within same priority."""
    priority: int
    seq:      int                              # monotonic counter for FIFO tie-break
    job_type: str  = field(compare=False)
    payload:  Any  = field(compare=False, default=None)


# ── JobQueue ──────────────────────────────────────────────────────────────────

class JobQueue:
    """
    Single-worker priority job queue with O(1) deduplication.

    The dedup set ensures at most one pending entry per job type.
    Because the job type set is finite and small (8 types), the
    set is effectively a constant-size lookup table — O(1) always.
    """

    # Retrain gates (mirror sync.py originals, now enforced inside the queue)
    _MIN_RETRAIN_INTERVAL_DAYS   = 7
    _MIN_NEW_MATCHES_FOR_RETRAIN = 50

    def __init__(self):
        self._pq: queue.PriorityQueue = queue.PriorityQueue()
        self._seq      = 0
        self._pending: set = set()          # job types currently in the queue
        self._running: Optional[str] = None  # job type currently executing
        self._lock     = threading.Lock()
        self._status:  Dict[str, dict] = {}

        # Small thread pool for fire-and-forget DB log inserts.
        # max_workers=4 bounds the number of simultaneous DB connections
        # opened for logging, preventing connection pool exhaustion.
        self._log_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="plusone-log"
        )

        # Single persistent worker — heavy jobs are serialised.
        self._worker = threading.Thread(
            target=self._run_worker,
            daemon=True,
            name="plusone-job-worker",
        )

    def start(self) -> None:
        """Start the worker thread (idempotent)."""
        if not self._worker.is_alive():
            self._worker.start()
            log.info("JobQueue worker started.")

    # ── Public API ────────────────────────────────────────────────────────────

    def enqueue(self, job_type: str, payload: Any = None) -> bool:
        """
        O(log k) enqueue. Returns True if queued, False if deduplicated.

        LOG jobs bypass the main queue and go straight to the thread pool —
        no dedup, no blocking, no priority ordering needed.
        """
        if job_type in _LOG_JOB_TYPES:
            fn = self._resolve(job_type)
            if fn is not None:
                if payload is not None:
                    self._log_pool.submit(fn, payload)
                else:
                    self._log_pool.submit(fn)
            return True

        with self._lock:
            # Dedup: O(1) set lookup
            if job_type in self._pending or self._running == job_type:
                log.debug("JobQueue dedup: %s already pending/running.", job_type)
                return False
            self._seq += 1
            pri = _JOB_PRIORITY.get(job_type, Priority.WARM_CACHE)
            self._pq.put(_Job(priority=pri, seq=self._seq,
                               job_type=job_type, payload=payload))
            self._pending.add(job_type)
            self._status[job_type] = {
                "status":      "pending",
                "enqueued_at": time.time(),
            }
            log.debug("JobQueue enqueued %s (priority=%d).", job_type, pri)
            return True

    def queue_status(self) -> dict:
        """Return current queue state for observability endpoint."""
        with self._lock:
            return {
                "running": self._running,
                "pending": sorted(self._pending),
                "history": dict(self._status),
            }

    # ── Worker loop ───────────────────────────────────────────────────────────

    def _run_worker(self) -> None:
        log.info("JobQueue worker loop running.")
        while True:
            try:
                job = self._pq.get(timeout=5)
            except queue.Empty:
                continue

            with self._lock:
                self._pending.discard(job.job_type)
                self._running = job.job_type
                self._status[job.job_type].update({
                    "status":     "running",
                    "started_at": time.time(),
                })

            try:
                log.info("JobQueue starting: %s", job.job_type)
                fn = self._resolve(job.job_type)
                if fn is not None:
                    result = fn(job.payload) if job.payload is not None else fn()
                    with self._lock:
                        self._status[job.job_type].update({
                            "status":       "done",
                            "completed_at": time.time(),
                            "result":       str(result)[:200] if result else None,
                        })
                    log.info("JobQueue done: %s → %s", job.job_type, result)
            except Exception as exc:
                log.exception("JobQueue error: %s failed.", job.job_type)
                with self._lock:
                    self._status[job.job_type].update({
                        "status":       "error",
                        "error":        str(exc)[:300],
                        "completed_at": time.time(),
                    })
            finally:
                with self._lock:
                    self._running = None
                self._pq.task_done()

    # ── Job resolvers (lazy imports prevent circular dependencies) ────────────

    def _resolve(self, job_type: str) -> Optional[Callable]:
        try:
            if job_type == JobType.EVALUATE_PREDICTIONS:
                from routes.prediction_log import do_evaluate_predictions
                from database import get_connection
                def _eval():
                    conn = get_connection()
                    try:
                        n = do_evaluate_predictions(conn)
                        conn.commit()
                        return {"evaluated": n}
                    finally:
                        conn.close()
                return _eval

            if job_type == JobType.RECALIBRATE_ML:
                from ml.feedback_calibrator import recalibrate_with_feedback
                return recalibrate_with_feedback

            if job_type == JobType.RECALIBRATE_DC:
                def _dc_cal():
                    from ml.dc_engine import get_dc_predictor
                    dc = get_dc_predictor()
                    if dc and dc.fitted:
                        n = dc.fit_calibrator_from_log()
                        return {"dc_samples": n}
                    return {"dc_samples": 0}
                return _dc_cal

            if job_type == JobType.RECALIBRATE_MARKETS:
                from ml.market_recalibrator import recalibrate_markets_from_log
                return recalibrate_markets_from_log

            if job_type == JobType.RETRAIN_ML:
                def _retrain():
                    # Gate: minimum interval between retrains
                    import datetime
                    from ml.prediction_engine import train_model, _meta
                    trained_at_str = _meta.get("trained_at")
                    if trained_at_str:
                        try:
                            trained_at = datetime.datetime.fromisoformat(
                                trained_at_str.replace("Z", "+00:00")
                            )
                            days = (datetime.datetime.now(datetime.timezone.utc)
                                    - trained_at).days
                            if days < self._MIN_RETRAIN_INTERVAL_DAYS:
                                log.debug("Retrain skipped: only %d days since last.", days)
                                return {"skipped": True, "reason": f"{days}d < 7d"}
                        except Exception:
                            pass
                    result = train_model()
                    if result.get("success"):
                        # Retrain done — invalidate all cache and warm
                        from ml.cache import _cache
                        _cache.invalidate_all()
                        get_queue().enqueue(JobType.WARM_CACHE)
                    return result
                return _retrain

            if job_type == JobType.WARM_CACHE:
                from ml.cache_warmer import warm_upcoming_cache
                return warm_upcoming_cache

        except Exception as exc:
            log.warning("JobQueue _resolve(%s) failed: %s", job_type, exc)
        return None


# ── Module-level singleton ────────────────────────────────────────────────────

_queue: Optional[JobQueue] = None
_queue_lock = threading.Lock()


def get_queue() -> JobQueue:
    """Return the singleton JobQueue, creating and starting it if needed."""
    global _queue
    if _queue is None:
        with _queue_lock:
            if _queue is None:
                _queue = JobQueue()
                _queue.start()
    return _queue
