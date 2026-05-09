import os
import shutil as _shutil
import subprocess as _sp
import logging as _logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv


from routes import leagues, teams, matches, standings, squad_stats, player_stats, sync, sync_enrichment, health, auth, cleanup, predictions, venue_stats, prediction_log, markets, performance, feedback, settings, prediction_ask

try:
    from routes import soccerdata_sync
    _soccerdata_available = True
except ImportError:
    _soccerdata_available = False


load_dotenv()

# ─── Numpy → psycopg2 type adapters ──────────────────────────────────────────
import numpy as np
import psycopg2.extensions

psycopg2.extensions.register_adapter(np.float64, lambda x: psycopg2.extensions.AsIs(float(x)))
psycopg2.extensions.register_adapter(np.float32, lambda x: psycopg2.extensions.AsIs(float(x)))
psycopg2.extensions.register_adapter(np.int64,   lambda x: psycopg2.extensions.AsIs(int(x)))
psycopg2.extensions.register_adapter(np.int32,   lambda x: psycopg2.extensions.AsIs(int(x)))
psycopg2.extensions.register_adapter(np.bool_,   lambda x: psycopg2.extensions.AsIs(bool(x)))


# ─── HTTPS redirect middleware ─────────────────────────────────────────────────
class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        proto = request.headers.get("x-forwarded-proto", "")
        host  = request.headers.get("host", "")
        is_local = host.startswith("localhost") or host.startswith("127.0.0.1")
        if proto == "http" and not is_local:
            https_url = str(request.url).replace("http://", "https://", 1)
            return RedirectResponse(url=https_url, status_code=301)
        return await call_next(request)


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Football Analytics API",
    description="Production API for football data scraped from FBref",
    version="1.0.0",
    redirect_slashes=False,
)

app.add_middleware(HTTPSRedirectMiddleware)

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:4000",
    "https://localhost:5173",
    "https://football-analytics-eight.vercel.app",
    "https://plusone-frontend-mu.vercel.app",
]

CORS_ORIGIN_REGEX = (
    r"https?://localhost(:\d+)?"
    r"|https://[a-z0-9-]+(?:\.[a-z0-9-]+)*\.vercel\.app"
    r"|https://[a-z0-9-]+\.onrender\.com"
    r"|https://[a-z0-9-]+(?:\.up)?\.railway\.app"
    r"|chrome-extension://[a-z0-9]+"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["Content-Length"],
    max_age=600,
)

# Register all route modules
app.include_router(health.router,       prefix="/api",             tags=["Health"])
app.include_router(leagues.router,      prefix="/api/leagues",     tags=["Leagues"])
app.include_router(teams.router,        prefix="/api/teams",       tags=["Teams"])
app.include_router(matches.router,      prefix="/api/matches",     tags=["Matches"])
app.include_router(standings.router,    prefix="/api/standings",   tags=["Standings"])
app.include_router(squad_stats.router,  prefix="/api/squad-stats", tags=["Squad Stats"])
app.include_router(venue_stats.router,  prefix="/api/venue-stats", tags=["Venue Stats"])
app.include_router(player_stats.router, prefix="/api/players",     tags=["Players"])
app.include_router(sync.router,         prefix="/api/sync",        tags=["Sync"])
app.include_router(sync_enrichment.router, prefix="/api/sync",     tags=["Sync Validation"])
app.include_router(cleanup.router,      prefix="/api/cleanup",     tags=["Cleanup"])
app.include_router(auth.router,                                     tags=["Auth"])
app.include_router(predictions.router,    prefix="/api/predictions",    tags=["Predictions"])
app.include_router(prediction_log.router, prefix="/api/prediction-log", tags=["Prediction Log"])
app.include_router(markets.router,        prefix="/api/markets",        tags=["Markets"])
app.include_router(performance.router,    prefix="/api/performance",    tags=["Performance"])
app.include_router(feedback.router,       prefix="/api/feedback",       tags=["Feedback"])
app.include_router(settings.router,       prefix="/api/settings",       tags=["Settings"])
app.include_router(prediction_ask.router, prefix="/api/predict",        tags=["Prediction Ask"])

if _soccerdata_available:
    app.include_router(soccerdata_sync.router, prefix="/api/soccerdata", tags=["Soccerdata Sync"])


# ─── Chrome/Chromium startup diagnostic ───────────────────────────────────────
# Runs once at boot. Check Railway deploy logs for "=== Chrome diagnostic ===" to
# see exactly what Chrome binary is (or isn't) available in this container.
_diag = _logging.getLogger("chrome_diag")

def _log_chrome_env() -> None:
    _diag.info("=== Chrome diagnostic ===")
    _diag.info("CHROME_BIN env     : %s", os.environ.get("CHROME_BIN", "(not set)"))
    _diag.info("CHROMIUM_BIN env   : %s", os.environ.get("CHROMIUM_BIN", "(not set)"))
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        found = _shutil.which(name)
        _diag.info("shutil.which(%-22r): %s", name, found or "(not found)")
    for cmd in ("chromium", "chromium-browser", "google-chrome"):
        try:
            out = _sp.check_output(["which", cmd], text=True, stderr=_sp.DEVNULL).strip()
            _diag.info("shell which %-20s: %s", cmd, out)
        except Exception:
            _diag.info("shell which %-20s: (not found)", cmd)
    _diag.info("PATH: %s", os.environ.get("PATH", "(not set)"))
    _diag.info("=== end Chrome diagnostic ===")

_log_chrome_env()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 4000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
