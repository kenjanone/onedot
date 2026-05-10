from fastapi import APIRouter
from database import get_connection

router = APIRouter()

@router.get("/health")
def health_check():
    try:
        conn = get_connection()
        conn.close()

        # Check httpx is importable (used by prediction_ask.py for LLM calls)
        httpx_ok = False
        httpx_version = None
        try:
            import httpx
            httpx_ok = True
            httpx_version = getattr(httpx, "__version__", "unknown")
        except ImportError:
            pass

        return {
            "status": "healthy",
            "version": "1.0.0",
            "database": "connected",
            "httpx": f"v{httpx_version}" if httpx_ok else "not installed",
            "message": "All systems operational"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "version": "1.0.0",
            "database": "error",
            "error": str(e),
            "message": "Database connection failed"
        }
