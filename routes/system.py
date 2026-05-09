"""
System / Observability Routes
==============================
Admin-only endpoints for monitoring cache health and job queue status.

GET  /api/system/cache-stats   — cache hit rate, key counts, TTLs
GET  /api/system/queue-status  — job queue running/pending/history
POST /api/system/cache-clear   — manual full cache bust (admin only)
POST /api/system/warm-cache    — manually trigger background cache warm
"""

import logging
from fastapi import APIRouter, Depends
from routes.deps import require_admin

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/cache-stats")
def cache_stats(_admin: dict = Depends(require_admin)):
    """Cache hit/miss rates, live key count, and storage bounds."""
    from ml.cache import _cache
    return {"success": True, "cache": _cache.stats()}


@router.get("/queue-status")
def queue_status(_admin: dict = Depends(require_admin)):
    """Job queue state: currently running job, pending jobs, history."""
    from ml.job_queue import get_queue
    return {"success": True, "queue": get_queue().queue_status()}


@router.post("/cache-clear")
def cache_clear(_admin: dict = Depends(require_admin)):
    """Manually bust the entire cache. Use when DB data has been manually edited."""
    from ml.cache import _cache
    n = _cache.invalidate_all()
    log.info("Manual cache clear: %d keys evicted.", n)
    return {"success": True, "keys_cleared": n}


@router.post("/warm-cache")
def warm_cache(_admin: dict = Depends(require_admin)):
    """Manually trigger background cache warming for upcoming predictions."""
    from ml.job_queue import get_queue, JobType
    enqueued = get_queue().enqueue(JobType.WARM_CACHE)
    return {
        "success":  True,
        "enqueued": enqueued,
        "message":  "Cache warming queued." if enqueued else "Already queued/running.",
    }
