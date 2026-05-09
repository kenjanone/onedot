import json
import re
import time
import threading
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Any
from database import get_connection
from routes.deps import require_admin

_sync_log = logging.getLogger(__name__)

from functools import wraps

def retry_on_db_lock_errors(max_retries=6, base_delay=1.5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_str = str(e).lower()
                    # Check for deadlocks or statement timeouts
                    if "deadlock" in err_str or "timeout" in err_str or "lock" in err_str:
                        last_err = e
                        if attempt < max_retries - 1:
                            time.sleep(base_delay + base_delay * attempt)
                            continue
                    # It's an unrelated DB error or we exhausted retries
                    raise e
            raise last_err
        return wrapper
    return decorator

def safe_num(val):
    """Safely convert FBref values to a number, returning None for non-numeric."""
    if val is None:
        return None
    s = str(val).strip().replace(",", "").replace("%", "").replace("N/A", "").replace("nan", "")
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None


def safe_text(val):
    """Extract plain text from a value that may be a dict/link object or plain string."""
    if val is None:
        return ""
    if isinstance(val, dict):
        return str(val.get("text", val.get("name", ""))).strip()
    s = str(val).strip()
    if s.startswith("{") and ("'text'" in s or '"text"' in s):
        try:
            import ast
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                return str(d.get("text", d.get("name", ""))).strip()
        except Exception:
            pass
    return s


def trunc(val, max_len: int):
    """Cap a string to max_len to respect VARCHAR column limits."""
    if val is None:
        return None
    s = str(val).strip()
    return s[:max_len] if s else None


def safe_age_int(val):
    """Convert FBref age/birth_year to an integer."""
    if val is None:
        return None
    s = safe_text(val) if isinstance(val, dict) else str(val).strip()
    s = s.replace(",", "").strip()
    s = s.split("-")[0].split(".")[0].strip()
    try:
        return int(s) if s else None
    except (ValueError, TypeError):
        return None


router = APIRouter()


from routes.prediction_log import do_evaluate_predictions

_evaluate_lock = threading.Lock()

def _auto_evaluate_predictions(conn) -> int:
    """
    Grade all unevaluated prediction_log rows whose match is now complete.
    Called automatically after each sync so performance metrics stay current.
    Returns the number of rows updated.
    """
    with _evaluate_lock:
        try:
            updated = do_evaluate_predictions(conn)
            conn.commit()
            return updated
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Auto-evaluate failed: %s", e)
            try: conn.rollback()
            except Exception: pass
            return 0
_recalibrate_lock = threading.Lock()

def _auto_recalibrate_bg():
    """
    Fire-and-forget: refit the ML feedback calibrator from prediction_log.
    Runs in a background daemon thread so the sync response is not delayed.
    After calibration, also trigger DC live-calibration from dc_correct data.
    """
    if not _recalibrate_lock.acquire(blocking=False):
        _sync_log.debug("Auto-recalibrate already running. Skipping this trigger.")
        return
    try:
        from ml.feedback_calibrator import recalibrate_with_feedback
        recalibrate_with_feedback()
        _sync_log.info("Auto-recalibrate (ML) completed after sync.")
    except Exception as exc:
        _sync_log.warning("Auto-recalibrate (ML) failed: %s", exc)
    try:
        from ml.dc_engine import get_dc_predictor
        dc = get_dc_predictor()
        if dc is not None and dc.fitted:
            n = dc.fit_calibrator_from_log()
            if n:
                _sync_log.info("Auto-recalibrate (DC) completed on %d samples.", n)
    except Exception as exc:
        _sync_log.warning("Auto-recalibrate (DC) failed: %s", exc)
    try:
        from ml.market_recalibrator import recalibrate_markets_from_log
        recalibrate_markets_from_log()
        _sync_log.info("Auto-recalibrate (markets) completed after sync.")
    except Exception as exc:
        _sync_log.warning("Auto-recalibrate (markets) failed: %s", exc)
    finally:
        _recalibrate_lock.release()

# ── Auto-retrain ─────────────────────────────────────────────────────────────
# Triggers a full ML ensemble retrain when enough new completed matches have
# accumulated since the last training run, without ever blocking a sync response.

_retrain_lock = threading.Lock()
_MIN_NEW_MATCHES_FOR_RETRAIN = 50   # retrain when this many new completed matches
                                     # have accumulated since the last training run
_MIN_RETRAIN_INTERVAL_DAYS   = 7    # never retrain more frequently than once per week

def _auto_retrain_bg():
    """
    Fire-and-forget: retrain the ML ensemble when enough new completed matches
    have accumulated since the last training run.

    Gates (both must pass before retraining fires):
      1. At least _MIN_RETRAIN_INTERVAL_DAYS since last successful train.
      2. At least _MIN_NEW_MATCHES_FOR_RETRAIN new completed matches in the DB
         vs n_samples in the current trained model.

    This guarantees the model stays current without running on every sync.
    """
    if not _retrain_lock.acquire(blocking=False):
        _sync_log.debug("Auto-retrain already running. Skipping this trigger.")
        return
    try:
        import datetime
        from ml.prediction_engine import train_model
        from ml.prediction_engine import _meta as _engine_meta

        # ── Gate 1: minimum interval since last train ─────────────────────────
        trained_at_str = _engine_meta.get("trained_at")
        if trained_at_str:
            try:
                trained_at = datetime.datetime.fromisoformat(
                    trained_at_str.replace("Z", "+00:00")
                )
                days_since = (
                    datetime.datetime.now(datetime.timezone.utc) - trained_at
                ).days
                if days_since < _MIN_RETRAIN_INTERVAL_DAYS:
                    _sync_log.debug(
                        "Auto-retrain skipped: only %d days since last train (min %d).",
                        days_since, _MIN_RETRAIN_INTERVAL_DAYS,
                    )
                    return
            except Exception:
                pass  # if date parse fails, proceed to gate 2

        # ── Gate 2: enough new completed matches since last train ─────────────
        n_trained = int(_engine_meta.get("n_samples") or 0)
        try:
            from database import get_connection as _gc
            _conn = _gc()
            _cur  = _conn.cursor()
            _cur.execute(
                "SELECT COUNT(*) AS n FROM matches WHERE home_score IS NOT NULL"
            )
            _row = _cur.fetchone()
            _conn.close()
            n_total = int(_row["n"] or 0) if _row else 0
        except Exception as exc:
            _sync_log.warning("Auto-retrain gate-2 check failed: %s", exc)
            return

        new_matches = n_total - n_trained
        if new_matches < _MIN_NEW_MATCHES_FOR_RETRAIN:
            _sync_log.debug(
                "Auto-retrain skipped: only %d new completed matches since last "
                "train (need %d).",
                new_matches, _MIN_NEW_MATCHES_FOR_RETRAIN,
            )
            return

        _sync_log.info(
            "Auto-retrain triggered: %d new completed matches since last train "
            "(%d total, %d trained on).",
            new_matches, n_total, n_trained,
        )
        result = train_model()
        if result.get("success"):
            _sync_log.info(
                "Auto-retrain completed: %d matches trained, CV accuracy %.1f%%.",
                result.get("matches_trained", 0),
                (result.get("cv_accuracy") or 0) * 100,
            )
        else:
            _sync_log.warning(
                "Auto-retrain completed with error: %s", result.get("error", "unknown")
            )
    except Exception as exc:
        _sync_log.exception("Auto-retrain thread error: %s", exc)
    finally:
        _retrain_lock.release()

class TableData(BaseModel):
    headers: List[str] = []
    rows: List[List[Any]] = []
    rowCount: Optional[int] = None

class SyncPayload(BaseModel):
    league: str
    season: str
    tables: Optional[List[TableData]] = None
    fixtures: Optional[List[dict]] = None
    stats: Optional[List[dict]] = None
    player_stats: Optional[List[dict]] = None
    playerStats: Optional[List[dict]] = None
    team_logos: Optional[dict] = None

def tables_to_fixtures(tables):
    # FBref sends day-of-week as text abbreviations: "Sat", "Sun", "Mon" etc.
    # Convert to ISO weekday integer (1=Mon … 7=Sun) so safe_num() doesn't
    # silently discard them. Previously ALL dayofweek values were NULL in DB.
    _DOW = {
        "mon": 1, "tue": 2, "wed": 3, "thu": 4,
        "fri": 5, "sat": 6, "sun": 7,
    }

    result = []
    for table in tables:
        headers = [h.strip().lower() for h in table.headers]
        for row in table.rows:
            if len(row) < 3:
                continue
            r = dict(zip(headers, row))
            home = safe_text(r.get("home_team", r.get("home", r.get("home team", ""))))
            away = safe_text(r.get("away_team", r.get("away", r.get("away team", ""))))
            if not home or not away or home.lower() in ("home", "home_team", ""):
                continue

            # dayofweek: map FBref text abbreviation → ISO integer (was always NULL before)
            raw_dow = safe_text(r.get("dayofweek", r.get("day", "")))
            dow_int = _DOW.get(raw_dow.lower()[:3], None) if raw_dow else None

            # gameweek: numeric only; round stored separately as free-text
            gw_raw = safe_text(r.get("gameweek", r.get("wk", "")))
            gw_num = safe_num(gw_raw)

            result.append({
                "home_team":  home,
                "away_team":  away,
                "date":       safe_text(r.get("date", r.get("dates", ""))),
                "start_time": safe_text(r.get("start_time", r.get("time", ""))),
                "score":      trunc(safe_text(r.get("score", "")), 30),
                "gameweek":   gw_num,
                "dayofweek":  dow_int,
                "venue":      safe_text(r.get("venue", "")),
                "attendance": safe_num(r.get("attendance", None)),
                "referee":    safe_text(r.get("referee", "")),
                "round":      trunc(safe_text(r.get("round", r.get("wk", r.get("gameweek", "")))), 100),
            })
    return result


def tables_to_squad_stats(tables):
    result = []
    for table in tables:
        headers = [h.strip().lower() for h in table.headers]
        for row in table.rows:
            if len(row) < 2:
                continue
            r = dict(zip(headers, row))
            team = safe_text(r.get("squad", r.get("team", "")))
            if not team or team.lower() in ("squad", "team", ""):
                continue
            extra = {k: v for k, v in r.items() if k not in ("squad", "team")}
            result.append({
                "team": team,
                "players_used": r.get("# pl", r.get("players used", r.get("players_used", None))),
                "avg_age": r.get("age", r.get("avg age", None)),
                "possession": r.get("poss", r.get("possession", None)),
                "games": r.get("mp", r.get("games", None)),
                "games_starts": r.get("starts", r.get("games_starts", None)),
                "minutes": r.get("min", r.get("minutes", None)),
                "minutes_90s": r.get("90s", r.get("minutes_90s", None)),
                "goals": r.get("gls", r.get("goals", None)),
                "assists": r.get("ast", r.get("assists", None)),
                "standard_stats": extra,
            })
    return result


def tables_to_player_stats(tables):
    result = []
    for table in tables:
        headers = [h.strip().lower() for h in table.headers]
        for row in table.rows:
            if len(row) < 2:
                continue
            r = dict(zip(headers, row))
            name = safe_text(r.get("player", ""))
            if not name or name.lower() in ("player", ""):
                continue
            extra = {k: v for k, v in r.items() if k not in ("player",)}
            raw_nat = safe_text(r.get("nationality", r.get("nation", "")) or "").strip()
            nationality = raw_nat.split()[-1] if raw_nat else ""
            result.append({
                "player":        name,
                "nationality":   trunc(nationality, 10),
                "position":      trunc(safe_text(r.get("position", r.get("pos", ""))), 20),
                "team":          safe_text(r.get("team", r.get("squad", ""))),
                "age":           safe_age_int(r.get("age", None)),
                "birth_year":    safe_age_int(r.get("birth_year", r.get("born", None))),
                "games":         safe_num(r.get("games", r.get("mp", None))),
                "games_starts":  safe_num(r.get("games_starts", r.get("starts", None))),
                "minutes":       safe_num(r.get("minutes", r.get("min", None))),
                "minutes_90s":   safe_num(r.get("minutes_90s", r.get("90s", None))),
                "goals":         safe_num(r.get("goals", r.get("gls", None))),
                "assists":       safe_num(r.get("assists", r.get("ast", None))),
                "standard_stats": extra,
            })
    return result


def tables_to_home_away_stats(tables):
    """Parse FBref home/away split table (Table 2 on stats pages).
    Returns two dicts per team: one for 'home' and one for 'away'.
    FBref column pattern: home_games, home_wins, home_ties, home_losses,
    home_goals_for, home_goals_against, home_goal_diff, home_points,
    and same with 'away_' prefix.
    """
    result = []
    for table in tables:
        headers = [h.strip().lower() for h in table.headers]
        for row in table.rows:
            r = dict(zip(headers, row))
            team = safe_text(r.get("team", r.get("squad", "")))
            if not team or team.lower() in ("team", "squad", ""):
                continue
            # Home row
            result.append({
                "team": team,
                "venue": "home",
                "games":          safe_num(r.get("home_games",         r.get("home_mp", None))),
                "wins":           safe_num(r.get("home_wins",          r.get("home_w",  None))),
                "draws":          safe_num(r.get("home_ties",          r.get("home_d",  None))),
                "losses":         safe_num(r.get("home_losses",        r.get("home_l",  None))),
                "goals_for":      safe_num(r.get("home_goals_for",     r.get("home_gf", None))),
                "goals_against":  safe_num(r.get("home_goals_against", r.get("home_ga", None))),
                "goal_diff":      safe_num(r.get("home_goal_diff",     r.get("home_gd", None))),
                "points":         safe_num(r.get("home_points",        r.get("home_pts",None))),
            })
            # Away row
            result.append({
                "team": team,
                "venue": "away",
                "games":          safe_num(r.get("away_games",         r.get("away_mp", None))),
                "wins":           safe_num(r.get("away_wins",          r.get("away_w",  None))),
                "draws":          safe_num(r.get("away_ties",          r.get("away_d",  None))),
                "losses":         safe_num(r.get("away_losses",        r.get("away_l",  None))),
                "goals_for":      safe_num(r.get("away_goals_for",     r.get("away_gf", None))),
                "goals_against":  safe_num(r.get("away_goals_against", r.get("away_ga", None))),
                "goal_diff":      safe_num(r.get("away_goal_diff",     r.get("away_gd", None))),
                "points":         safe_num(r.get("away_points",        r.get("away_pts",None))),
            })
    return result


def detect_table_type(table):
    headers_lower = [h.strip().lower() for h in table.headers]
    headers_set = set(headers_lower)

    # -- Home/Away split table (Table 2 on FBref stats pages) ---
    if ("home_games" in headers_set or "home_wins" in headers_set) and ("rank" in headers_set or "rk" in headers_set) and ("team" in headers_set or "squad" in headers_set):
        return "standings_home_away"

    # -- Standings ---
    has_rank  = "rank" in headers_set or "rk" in headers_set
    has_pts   = "points" in headers_set or "pts" in headers_set
    has_squad = "team" in headers_set or "squad" in headers_set
    has_wins  = "wins" in headers_set or "w" in headers_set
    if has_rank and has_pts and has_squad and has_wins:
        return "standings"

    # -- Fixtures ---
    has_home_team = "home_team" in headers_set or "home" in headers_set
    has_date      = "date" in headers_set
    has_score     = "score" in headers_set
    if has_home_team or (has_date and has_score):
        return "fixtures"

    # -- Player stats ---
    if "player" in headers_set:
        return "player_stats"

    return "squad_stats"


def get_or_create(cur, table, unique_cols, extra_cols={}):
    where = " AND ".join(f"{k}=%s" for k in unique_cols)
    cur.execute(f"SELECT id FROM {table} WHERE {where}", list(unique_cols.values()))
    row = cur.fetchone()
    if row:
        return row["id"]
    all_cols = {**unique_cols, **extra_cols}
    cols = ", ".join(all_cols.keys())
    placeholders = ", ".join(["%s"] * len(all_cols))
    cur.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) RETURNING id", list(all_cols.values()))
    return cur.fetchone()["id"]


def get_or_create_league(cur, name):
    clean = name.strip()
    if clean.isdigit():
        raise ValueError(f"Invalid league name '{clean}'")
    # Reject OpenAPI/Swagger placeholder values that indicate an unfilled template
    _PLACEHOLDER_NAMES = frozenset({
        "string", "league", "leaguename", "example", "test",
        "placeholder", "your_league", "insert_league",
    })
    if clean.lower() in _PLACEHOLDER_NAMES:
        raise ValueError(
            f"Invalid league name '{clean}' — looks like an unfilled API placeholder. "
            f"Pass a real league name such as 'Premier League' or 'Bundesliga'."
        )
    cur.execute("SELECT id FROM leagues WHERE name ILIKE %s LIMIT 1", (clean,))
    row = cur.fetchone()
    if row:
        return row["id"]
    normalized = clean.title()
    cur.execute("""
        INSERT INTO leagues (name) VALUES (%s)
        ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name
        RETURNING id
    """, (normalized,))
    return cur.fetchone()["id"]


def get_or_create_season(cur, name):
    clean = name.strip()
    cur.execute("SELECT id FROM seasons WHERE name ILIKE %s LIMIT 1", (clean,))
    row = cur.fetchone()
    if row:
        return row["id"]
    cur.execute("""
        INSERT INTO seasons (name) VALUES (%s)
        ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name
        RETURNING id
    """, (clean,))
    return cur.fetchone()["id"]


def _normalize_fbref_team_name(name: str) -> str:
    """
    FBref decorates team names with 2-3 char ISO country codes in two ways:

    PREFIX (own-stats rows on UEFA pages):
      'eng Arsenal'      -> 'Arsenal'
      'gr  AEK Athens'   -> 'AEK Athens'

    SUFFIX (opponent rows / some UEFA fixture pages):
      'AEK Athens gr'    -> 'AEK Athens'
      'APOEL FC cy'      -> 'APOEL FC'
      'Aberdeen sct'     -> 'Aberdeen'
      'Ararat-Armenia am'-> 'Ararat-Armenia'

    Also handles 'cc vs TeamName' and plain 'vs TeamName' patterns.

    We deliberately avoid stripping suffix tokens that are part of the
    canonical team name (e.g. "Athletic Club", "Real Madrid", "AC Milan")
    by only removing a trailing token when it is a known 2-3 letter
    lowercase-only ISO country/region code AND the remaining name is
    at least 2 characters long.
    """
    s = name.strip()

    # "cc vs TeamName"
    m = re.match(r'^[a-z]{2,3}\s+vs\s+(.+)$', s)
    if m:
        return m.group(1).strip()

    # PREFIX: "cc TeamName" where cc is 2-3 lowercase letters
    m = re.match(r'^([a-z]{2,3})\s+(.+)$', s)
    if m:
        s = m.group(2).strip()

    # Strip leading vs/at
    s = re.sub(r'^(?:vs\s+|at\s+)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'(?:\s+at|\s+vs)$', '', s, flags=re.IGNORECASE)
    s = s.strip()

    # SUFFIX: "TeamName cc" where cc is 2-3 lowercase letters only
    # e.g. "AEK Athens gr", "APOEL FC cy", "Aberdeen sct"
    # Guard: remaining name must be >= 2 chars so we don't over-strip
    m = re.match(r'^(.+?)\s+([a-z]{2,3})$', s)
    if m:
        remainder = m.group(1).strip()
        suffix    = m.group(2)
        # Only strip if the suffix is all-lowercase (country code),
        # not a legitimate capitalised abbreviation like "FC", "AC", "SC"
        if suffix == suffix.lower() and len(remainder) >= 2:
            s = remainder

    return s


def get_or_create_team(cur, name, league_id):
    """
    Always stores teams with CLEAN names (no FBref country suffixes).
    Lookup order:
      1. Exact clean name match in THIS league (fastest path, no ambiguity)
      2. Exact clean name match in ANY league (UEFA cross-league teams).
         Prefer the record with the LOWEST league_id (domestic leagues are
         inserted first and therefore have lower IDs), then shortest name
         as a tiebreaker. This makes the lookup deterministic even when a
         team exists in multiple league rows (e.g. Man City under PL AND UCL).
      3. Dirty suffix variant in DB → rename it and return its id.
         Same deterministic ordering applied.
      4. Insert brand-new clean record.
    Any dirty record found in step 3 is renamed on the spot so the DB
    self-heals on every scrape, even without running the migration SQL.
    """
    raw   = safe_text(name) or name.strip()
    clean = _normalize_fbref_team_name(raw)
    if not clean:
        return None

    # Apply shared alias map — same canonical resolver used by sync_enrichment.py
    # This ensures both FBref and Football-Data paths produce identical team names.
    # e.g. FBref sends "Nott'ham Forest" → alias → "Nottingham Forest"
    try:
        from routes.sync_enrichment import TEAM_NAME_ALIASES
        clean = TEAM_NAME_ALIASES.get(clean.strip().lower(), clean)
    except ImportError:
        try:
            from api.routes.sync_enrichment import TEAM_NAME_ALIASES
            clean = TEAM_NAME_ALIASES.get(clean.strip().lower(), clean)
        except ImportError:
            pass  # silently continue without alias map

    # 1. Exact clean match within this league (fastest path, no ambiguity)
    cur.execute(
        "SELECT id FROM teams WHERE name ILIKE %s AND league_id = %s LIMIT 1",
        (clean, league_id)
    )
    row = cur.fetchone()
    if row:
        return row["id"]

    # 2. Exact clean match across all leagues (UEFA teams stored under domestic league).
    #    BUG FIX: was plain LIMIT 1 with no ORDER BY — PostgreSQL returns rows in
    #    undefined physical order, causing non-deterministic resolution when a team
    #    exists under multiple leagues (e.g. Man City under PL AND UCL).
    #    Fix: ORDER BY league_id ASC (domestic leagues = lower IDs inserted first)
    #    then LENGTH(name) ASC as a tiebreaker (prefer shorter canonical name).
    cur.execute(
        "SELECT id FROM teams WHERE name ILIKE %s ORDER BY league_id ASC, LENGTH(name) ASC LIMIT 1",
        (clean,)
    )
    row = cur.fetchone()
    if row:
        return row["id"]

    # 3. Dirty suffix variant exists in DB — rename it and return its id.
    #    Covers cases like DB has "AEK Athens gr" but we're looking up "AEK Athens".
    #    Also covers the reverse: we received "AEK Athens gr" and DB has "AEK Athens".
    #    We match on: name starts with clean AND the remainder is a known suffix pattern.
    #    BUG FIX: same deterministic ordering as step 2.
    cur.execute(
        "SELECT id, name FROM teams WHERE name ILIKE %s ORDER BY league_id ASC, LENGTH(name) ASC LIMIT 1",
        (f"{clean} %",)   # "AEK Athens %" matches "AEK Athens gr"
    )
    row = cur.fetchone()
    if row:
        if row["name"] != clean:
            # Self-heal: rename dirty record to clean name
            cur.execute("UPDATE teams SET name = %s WHERE id = %s", (clean, row["id"]))
        return row["id"]

    # Also check if the name we received was clean but DB has dirty version
    # e.g. we received "Aberdeen" but DB has "Aberdeen sct" — within this league
    cur.execute(
        "SELECT id, name FROM teams WHERE name ILIKE %s AND league_id = %s ORDER BY LENGTH(name) ASC LIMIT 1",
        (f"{clean} %", league_id)
    )
    row = cur.fetchone()
    if row:
        if row["name"] != clean:
            cur.execute("UPDATE teams SET name = %s WHERE id = %s", (clean, row["id"]))
        return row["id"]

    # 4. Brand-new team — insert with clean name only
    cur.execute("""
        INSERT INTO teams (name, league_id) VALUES (%s, %s)
        ON CONFLICT (name, league_id) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
    """, (clean, league_id))
    return cur.fetchone()["id"]


def parse_score(score_raw):
    if not score_raw or str(score_raw).strip() in ("", "nan", "None"):
        return None, None
    # Strip any parenthetical suffix like "(aet)", "(5-3 pen)", then split on dash variant
    cleaned = re.sub(r"\s*\(.*?\)", "", str(score_raw)).strip()
    for sep in ["–", "-", "—"]:
        if sep in cleaned:
            parts = cleaned.split(sep)
            try:
                return int(parts[0].strip()), int(parts[1].strip())
            except (ValueError, IndexError):
                return None, None
    return None, None


def parse_date(raw):
    if not raw or str(raw).strip() in ("", "nan", "None"):
        return None
    s = str(raw).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s[:10]


@router.get("/status")
def sync_status():
    """Return per-league sync history from scrape_log + live row counts from DB."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        # ── Sync log history (requires scrape_log table) ───────────────────────
        # If the table doesn't exist yet (fresh DB), skip gracefully.
        log_rows = []
        try:
            cur.execute("""
                SELECT
                    l.name            AS league,
                    s.name            AS season,
                    sl.page_type,
                    SUM(sl.rows_inserted) AS rows,
                    MAX(sl.scraped_at)    AS last_sync
                FROM scrape_log sl
                JOIN leagues  l ON l.id = sl.league_id
                JOIN seasons  s ON s.id = sl.season_id
                GROUP BY l.name, s.name, sl.page_type
                ORDER BY l.name, sl.page_type
            """)
            log_rows = cur.fetchall()
        except Exception:
            # scrape_log table may not exist on fresh deployments — safe to ignore
            conn.rollback()

        # ── Live counts per league from key tables ─────────────────────────────
        cur.execute("""
            SELECT l.name AS league,
                (SELECT COUNT(m.id) FROM matches m WHERE m.league_id = l.id) AS fixtures,
                (SELECT COUNT(tvs.id) FROM team_venue_stats tvs WHERE tvs.league_id = l.id) AS home_away_rows,
                (SELECT COUNT(st.id) FROM league_standings st WHERE st.league_id = l.id) AS standings_rows
            FROM leagues l
            ORDER BY l.name
        """)
        live_rows = cur.fetchall()

        # ── Build structured response ──────────────────────────────────────────
        by_league = {}
        for r in log_rows:
            lg = r["league"]
            if lg not in by_league:
                by_league[lg] = {"league": lg, "season": r["season"], "log": [], "live": {}}
            by_league[lg]["log"].append({
                "type": r["page_type"],
                "rows": r["rows"],
                "last_sync": r["last_sync"].isoformat() if r["last_sync"] else None
            })

        for r in live_rows:
            lg = r["league"]
            if lg not in by_league:
                by_league[lg] = {"league": lg, "season": None, "log": [], "live": {}}
            by_league[lg]["live"] = {
                "fixtures": r["fixtures"],
                "home_away_rows": r["home_away_rows"],
                "standings_rows": r["standings_rows"]
            }

        return {"success": True, "leagues": list(by_league.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.post("/all")
@retry_on_db_lock_errors()
def sync_all(payload: SyncPayload, _admin: dict = Depends(require_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Self-healing: ensure enrichment_predicted_outcome column exists.
        # This column was added after initial release; older DBs may be missing it.
        try:
            cur.execute("""
                ALTER TABLE prediction_log
                ADD COLUMN IF NOT EXISTS enrichment_predicted_outcome TEXT
            """)
            conn.commit()
        except Exception:
            conn.rollback()  # Column may already exist or table locked — safe to ignore

        league_id = get_or_create_league(cur, payload.league)
        season_id = get_or_create_season(cur, payload.season)
        fixtures_list = payload.fixtures or []
        stats_list = payload.stats or []
        players_list = payload.playerStats or payload.player_stats or []
        standings_list = []
        ha_split_list  = []  # Home/Away split table (Table 2 on FBref stats pages)
        if payload.tables:
            for t in payload.tables:
                ttype = detect_table_type(t)
                if ttype == "standings":
                    standings_list.extend(tables_to_standings([t]))
                elif ttype == "fixtures":
                    fixtures_list.extend(tables_to_fixtures([t]))
                elif ttype == "player_stats":
                    players_list.extend(tables_to_player_stats([t]))
                elif ttype == "standings_home_away":
                    ha_split_list.extend(tables_to_home_away_stats([t]))
                else:
                    stats_list.extend(tables_to_squad_stats([t]))
        fx  = _insert_fixtures(cur, league_id, season_id, payload.league, fixtures_list)
        st  = _insert_squad_stats(cur, league_id, season_id, stats_list)
        pl  = _insert_player_stats(cur, league_id, season_id, players_list)
        sd  = _insert_standings(cur, league_id, season_id, standings_list)
        ha  = _insert_home_away_stats(cur, league_id, season_id, ha_split_list)
        logos = 0
        if payload.team_logos:
            logos = _update_team_logos(cur, league_id, payload.team_logos)

        total_rows = fx + st + pl + sd + ha

        # ── Empty-payload guard ────────────────────────────────────────────────
        # When FBref rate-limits the extension (429) or the page loads without
        # its stats tables (Bot-block / JS-only render), the scraper sends an
        # empty tables list.  Previously this was silently committed as a
        # "successful" sync with 0 rows, polluting the scrape_log and masking
        # failed scrapes.  Now we abort the commit and return a distinct warning
        # so the extension can flag the URL for retry.
        if total_rows == 0:
            conn.rollback()
            _sync_log.warning(
                "sync_all: empty payload for league=%s season=%s — no data written. "
                "Likely a rate-limited or failed scrape.",
                payload.league, payload.season
            )
            return {
                "success": False,
                "warning": "empty_payload",
                "detail": (
                    "No rows were extracted from the scraped page. "
                    "This is usually caused by FBref rate-limiting (429) or a "
                    "page that loaded without its stats tables. "
                    "Re-scrape this URL after a short delay."
                ),
                "fixtures_inserted": 0, "stats_inserted": 0,
                "players_inserted": 0, "standings_inserted": 0,
            }
        # ── End empty-payload guard ────────────────────────────────────────────

        conn.commit()
        log_scrape(cur, league_id, season_id, "sync_all", total_rows, 0)
        conn.commit()
        # Auto-evaluate predictions whose matches just received scores
        evaluated = _auto_evaluate_predictions(conn)
        # Auto-recalibrate both ML and DC engines if new grades came in
        if evaluated > 0:
            threading.Thread(target=_auto_recalibrate_bg, daemon=True,
                             name="auto-recalibrate").start()
        # Auto-retrain the ML ensemble when enough new matches have accumulated.
        # Internal gates prevent this from firing more than once per week or
        # before 50+ new completed matches are available.
        threading.Thread(target=_auto_retrain_bg, daemon=True,
                         name="auto-retrain").start()
        return {"success": True, "fixtures_inserted": fx, "stats_inserted": st, "players_inserted": pl, "standings_inserted": sd, "home_away_inserted": ha, "logos_updated": logos, "predictions_evaluated": evaluated, "recalibration_triggered": evaluated > 0}

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.post("/fixtures")
@retry_on_db_lock_errors()
def sync_fixtures(payload: SyncPayload, _admin: dict = Depends(require_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        league_id = get_or_create_league(cur, payload.league)
        season_id = get_or_create_season(cur, payload.season)
        rows = payload.fixtures or []
        if payload.tables:
            rows.extend(tables_to_fixtures(payload.tables))
        inserted = _insert_fixtures(cur, league_id, season_id, payload.league, rows)
        conn.commit()
        # Auto-evaluate predictions whose matches just received scores
        evaluated = _auto_evaluate_predictions(conn)
        if evaluated > 0:
            threading.Thread(target=_auto_recalibrate_bg, daemon=True,
                             name="auto-recalibrate").start()
        threading.Thread(target=_auto_retrain_bg, daemon=True,
                         name="auto-retrain").start()
        return {"success": True, "matches_inserted": inserted, "predictions_evaluated": evaluated, "recalibration_triggered": evaluated > 0}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.post("/stats")
@retry_on_db_lock_errors()
def sync_stats(payload: SyncPayload, _admin: dict = Depends(require_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        league_id = get_or_create_league(cur, payload.league)
        season_id = get_or_create_season(cur, payload.season)
        rows = payload.stats or []
        if payload.tables:
            rows.extend(tables_to_squad_stats(payload.tables))
        inserted = _insert_squad_stats(cur, league_id, season_id, rows)
        conn.commit()
        return {"success": True, "stats_inserted": inserted}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.post("/player-stats")
@retry_on_db_lock_errors()
def sync_player_stats(payload: SyncPayload, _admin: dict = Depends(require_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        league_id = get_or_create_league(cur, payload.league)
        season_id = get_or_create_season(cur, payload.season)
        rows = payload.player_stats or payload.playerStats or []
        if payload.tables:
            rows.extend(tables_to_player_stats(payload.tables))
        inserted = _insert_player_stats(cur, league_id, season_id, rows)
        conn.commit()
        return {"success": True, "players_inserted": inserted}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


def tables_to_standings(tables):
    result = []
    for table in tables:
        headers = [h.strip().lower() for h in table.headers]
        for row in table.rows:
            if len(row) < 3:
                continue
            r = dict(zip(headers, row))
            team = safe_text(r.get("squad", r.get("team", "")))
            if not team or team.lower() in ("squad", "team", ""):
                continue
            result.append({
                "rank":          safe_num(r.get("rk", r.get("rank", r.get("pos", r.get("#", None))))),
                "team":          team,
                "games":         safe_num(r.get("mp", r.get("games", r.get("pld", None)))),
                "wins":          safe_num(r.get("w", r.get("wins", None))),
                "ties":          safe_num(r.get("d", r.get("draws", r.get("ties", None)))),
                "losses":        safe_num(r.get("l", r.get("losses", None))),
                "goals_for":     safe_num(r.get("gf", r.get("goals_for", None))),
                "goals_against": safe_num(r.get("ga", r.get("goals_against", None))),
                "goal_diff":     safe_num(r.get("gd", r.get("goal_diff", None))),
                "points":        safe_num(r.get("pts", r.get("points", r.get("pt", None)))),
                "points_avg":    safe_num(r.get("pts/g", r.get("pts_avg", r.get("points_avg", None)))),
            })
    return result


def _insert_fixtures(cur, league_id, season_id, league_name, fixtures):
    count = 0
    for f in fixtures:
        home = str(f.get("home_team", "")).strip()
        away = str(f.get("away_team", "")).strip()
        if not home or not away:
            continue
        home_id = get_or_create_team(cur, home, league_id)
        away_id = get_or_create_team(cur, away, league_id)
        home_score, away_score = parse_score(f.get("score"))
        match_date = parse_date(f.get("date"))
        cur.execute("""
            INSERT INTO matches (league_id, season_id, home_team_id, away_team_id,
                gameweek, dayofweek, match_date, start_time, home_score, away_score, score_raw,
                attendance, venue, referee, round)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (home_team_id, away_team_id, match_date) DO UPDATE SET
                -- Fix corrupt rows that had league_id/season_id=None from old imports
                league_id  = COALESCE(matches.league_id, EXCLUDED.league_id),
                season_id  = COALESCE(matches.season_id, EXCLUDED.season_id),
                -- ONLY overwrite score when the incoming row carries a real score.
                -- A re-sync of a future-fixture page (score=NULL) must never erase
                -- a previously stored result. This was the root cause of 3,121 0-0 rows.
                home_score = CASE
                               WHEN EXCLUDED.home_score IS NOT NULL THEN EXCLUDED.home_score
                               ELSE matches.home_score
                             END,
                away_score = CASE
                               WHEN EXCLUDED.away_score IS NOT NULL THEN EXCLUDED.away_score
                               ELSE matches.away_score
                             END,
                score_raw  = CASE
                               WHEN EXCLUDED.score_raw IS NOT NULL AND EXCLUDED.score_raw != ''
                               THEN EXCLUDED.score_raw
                               ELSE matches.score_raw
                             END,
                -- Only flip is_played to TRUE when we're providing a real score;
                -- never flip it back to FALSE from a blank resync.
                is_played  = CASE
                               WHEN EXCLUDED.home_score IS NOT NULL THEN TRUE
                               ELSE matches.is_played
                             END,
                attendance = EXCLUDED.attendance,
                venue      = COALESCE(NULLIF(EXCLUDED.venue, ''), matches.venue),
                referee    = COALESCE(NULLIF(EXCLUDED.referee, ''), matches.referee),
                updated_at = NOW()
        """, (
            league_id, season_id, home_id, away_id,
            safe_num(f.get("gameweek")),    safe_num(f.get("dayofweek")),
            match_date,                     safe_text(f.get("start_time", "")) or None,
            home_score, away_score,         safe_text(f.get("score", "")),
            safe_num(f.get("attendance")),  safe_text(f.get("venue", "")),
            safe_text(f.get("referee", "")), safe_text(f.get("round", ""))
        ))
        count += 1
    return count


def _insert_squad_stats(cur, league_id, season_id, stats_rows):
    count = 0
    for row in stats_rows:
        team_raw = safe_text(row.get("team", ""))
        if not team_raw:
            continue
        # Detect FBref "against" split — handles both domestic ("vs Arsenal")
        # and UEFA-prefixed ("eng vs Arsenal") patterns
        is_against = team_raw.startswith("vs ") or bool(re.match(r'^[a-z]{2,3}\s+vs\s+', team_raw))
        split = "against" if is_against else "for"
        team_id = get_or_create_team(cur, team_raw, league_id)  # normalization happens inside
        cur.execute("""
            INSERT INTO team_squad_stats
                (team_id, league_id, season_id, split, players_used, avg_age, possession,
                 games, games_starts, minutes, minutes_90s, goals, assists,
                 standard_stats, goalkeeping, shooting, playing_time, misc_stats)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (team_id, season_id, split) DO UPDATE SET
                goals=EXCLUDED.goals, assists=EXCLUDED.assists,
                standard_stats=EXCLUDED.standard_stats,
                scraped_at=NOW()
        """, (
            team_id, league_id, season_id, split,
            safe_num(row.get("players_used")), safe_num(row.get("avg_age")), safe_num(row.get("possession")),
            safe_num(row.get("games")), safe_num(row.get("games_starts")), safe_num(row.get("minutes")), safe_num(row.get("minutes_90s")),
            safe_num(row.get("goals")), safe_num(row.get("assists")),
            json.dumps(row.get("standard_stats") or {}),
            json.dumps(row.get("goalkeeping") or {}),
            json.dumps(row.get("shooting") or {}),
            json.dumps(row.get("playing_time") or {}),
            json.dumps(row.get("misc_stats") or {}),
        ))
        count += 1
    return count


def _insert_home_away_stats(cur, league_id, season_id, rows):
    """Insert/update home and away venue stats per team into team_venue_stats."""
    count = 0
    for row in rows:
        team_name = row.get("team", "")
        if not team_name:
            continue
        team_id = get_or_create_team(cur, team_name, league_id)
        cur.execute("""
            INSERT INTO team_venue_stats
                (team_id, league_id, season_id, venue,
                 games, wins, draws, losses, goals_for, goals_against, goal_diff, points)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (team_id, season_id, venue) DO UPDATE SET
                league_id     = EXCLUDED.league_id,
                games         = EXCLUDED.games,
                wins          = EXCLUDED.wins,
                draws         = EXCLUDED.draws,
                losses        = EXCLUDED.losses,
                goals_for     = EXCLUDED.goals_for,
                goals_against = EXCLUDED.goals_against,
                goal_diff     = EXCLUDED.goal_diff,
                points        = EXCLUDED.points,
                updated_at    = NOW()
        """, (
            team_id, league_id, season_id, row.get("venue"),
            safe_num(row.get("games")),
            safe_num(row.get("wins")),
            safe_num(row.get("draws")),
            safe_num(row.get("losses")),
            safe_num(row.get("goals_for")),
            safe_num(row.get("goals_against")),
            safe_num(row.get("goal_diff")),
            safe_num(row.get("points")),
        ))
        count += 1
    return count


def _insert_player_stats(cur, league_id, season_id, players):
    count = 0
    for p in players:
        name = str(p.get("player", "")).strip()
        if not name or name.lower() in ("player", ""):
            continue
        team_name = str(p.get("team", "")).strip()
        team_id = None
        if team_name and league_id:
            team_id = get_or_create_team(cur, team_name, league_id)
        cur.execute("""
            INSERT INTO player_stats
                (player_name, nationality, position, team_id, season_id,
                 age, birth_year, games, games_starts, minutes, minutes_90s,
                 goals, assists, standard_stats)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (player_name, team_id, season_id) DO UPDATE SET
                goals=EXCLUDED.goals, assists=EXCLUDED.assists,
                standard_stats=EXCLUDED.standard_stats,
                scraped_at=NOW()
        """, (
            name,
            safe_text(p.get("nationality", "")),
            safe_text(p.get("position", "")),
            team_id, season_id,
            safe_age_int(p.get("age")),
            safe_age_int(p.get("birth_year")),
            safe_num(p.get("games")),
            safe_num(p.get("games_starts")),
            safe_num(p.get("minutes")),
            safe_num(p.get("minutes_90s")),
            safe_num(p.get("goals")),
            safe_num(p.get("assists")),
            json.dumps(p.get("standard_stats") or {})
        ))
        count += 1
    return count


def _insert_standings(cur, league_id, season_id, rows):
    count = 0
    for row in rows:
        team_name = row.get("team", "")
        if not team_name:
            continue
        team_id = get_or_create_team(cur, team_name, league_id)
        cur.execute("""
            INSERT INTO league_standings
                (league_id, season_id, team_id, rank, games, wins, ties, losses,
                 goals_for, goals_against, goal_diff, points, points_avg)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (team_id, league_id, season_id) DO UPDATE SET
                rank          = EXCLUDED.rank,
                games         = EXCLUDED.games,
                wins          = EXCLUDED.wins,
                ties          = EXCLUDED.ties,
                losses        = EXCLUDED.losses,
                goals_for     = EXCLUDED.goals_for,
                goals_against = EXCLUDED.goals_against,
                goal_diff     = EXCLUDED.goal_diff,
                points        = EXCLUDED.points,
                points_avg    = EXCLUDED.points_avg,
                scraped_at    = NOW()
        """, (
            league_id, season_id, team_id,
            row.get("rank"), row.get("games"),
            row.get("wins"), row.get("ties"), row.get("losses"),
            row.get("goals_for"), row.get("goals_against"), row.get("goal_diff"),
            row.get("points"), row.get("points_avg")
        ))
        count += 1
    return count


def _update_standings_home_away(cur, league_id, season_id, rows):
    for r in rows:
        team_name = r["team"]
        split_json = json.dumps(r["split"])
        cur.execute("""
            UPDATE league_standings ls
            SET    home_away_split = %s
            FROM   teams t
            WHERE  t.id = ls.team_id
              AND  ls.league_id = %s
              AND  ls.season_id = %s
              AND  LOWER(t.name) LIKE LOWER(%s)
        """, (split_json, league_id, season_id, f"%{team_name}%"))

def _update_team_logos(cur, league_id, team_logos):
    """Update team logos from the scraper mapping."""
    count = 0
    for team_name, logo_url in team_logos.items():
        if not team_name or not logo_url:
            continue
        try:
            # We use an ILIKE match on name and league_id to find the team
            cur.execute("""
                UPDATE teams
                SET logo_url = %s
                WHERE league_id = %s AND name ILIKE %s
                  AND (logo_url IS NULL OR logo_url != %s)
            """, (logo_url, league_id, f"%{team_name}%", logo_url))
            if cur.rowcount > 0:
                count += cur.rowcount
        except Exception:
            pass
    return count

def log_scrape(cur, league_id, season_id, page_type, inserted, updated):
    try:
        cur.execute("""
            INSERT INTO scrape_log (league_id, season_id, page_type, rows_inserted, rows_updated)
            VALUES (%s, %s, %s, %s, %s)
        """, (league_id, season_id, page_type, inserted, updated))
    except Exception:
        pass
