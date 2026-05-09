"""
Prediction Ask — /api/predict/ask
===================================
LLM-powered Q&A about any consensus match prediction.

Strategy:
  1. Fetch full consensus prediction (all 3 engines + markets + best bets)
  2. Fetch rich DB context from ALL relevant tables:
       - matches            → recent form, H2H, season stats (with season filter + fallback)
       - league_standings   → current standings (season + league filtered)
       - team_squad_stats   → possession, shooting %, save %, squad depth
       - team_venue_stats   → home/away goal splits and xG
       - player_stats       → top scorers / assisters
       - player_injuries    → current injuries and suspensions
       - team_clubelo       → ELO ratings and match-specific win probabilities
       - match_odds         → bookmaker implied probabilities (B365 + Over/Under)
  3. Ask the LLM to:
       a) Analyse the raw DB data independently
       b) Compare its own reading against the 3 engine predictions
       c) Agree or challenge the consensus, with reasoning
       d) Then answer the user's specific question
  4. Try the chosen provider first, fall back to the other
  5. Return answer + which provider/model was used

Environment variables (set in Railway):
  GROQ_API_KEY       — from console.groq.com
  GEMINI_API_KEY     — from aistudio.google.com

Optional overrides (no redeploy needed):
  GROQ_MODEL         — default: llama-3.3-70b-versatile
  GEMINI_MODEL       — default: gemini-2.0-flash

Available Groq models (all free):
  llama-3.3-70b-versatile   <- smartest, recommended
  llama-3.1-8b-instant      <- fastest
  mixtral-8x7b-32768        <- good for long context

Available Gemini models (free tier):
  gemini-2.0-flash          <- fast, good quality
  gemini-2.0-flash-lite     <- fastest, lower quality
"""

import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal

from database import get_connection

log = logging.getLogger(__name__)
router = APIRouter()

# API keys
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Default models (overridable via env vars without redeploying)
DEFAULT_GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Available models catalogue (shown to frontend for selection UI)
AVAILABLE_MODELS = {
    "groq": [
        {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B  - Smartest",  "free": True},
        {"id": "llama-3.1-8b-instant",    "label": "Llama 3.1 8B   - Fastest",   "free": True},
        {"id": "mixtral-8x7b-32768",      "label": "Mixtral 8x7B   - Long ctx",  "free": True},
    ],
    "gemini": [
        {"id": "gemini-2.0-flash",      "label": "Gemini 2.0 Flash      - Fast & smart", "free": True},
        {"id": "gemini-2.0-flash-lite", "label": "Gemini 2.0 Flash Lite - Fastest",      "free": True},
    ],
}

GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

SYSTEM_PROMPT = """You are PlusOne AI, a sharp football analyst embedded in a prediction app.

If match context is provided below, you have full match context: recent form, H2H record, \
season stats, standings, squad data, venue splits, top scorers, injuries, ClubElo ELO ratings, \
bookmaker odds, and outputs from three prediction engines.

RULES — follow strictly:
- Answer the user directly. No opening filler or preamble.
- Handle casual chat warmly. If the user just says "thanks", "great", or "hello", acknowledge it naturally and concisely without forcing match statistics into the response.
- If the user asks a football query and match data is provided, support your analysis with 2-3 specific numbers from the data.
- If engines disagree, note it in one sentence only.
- Plain text only. No markdown, no asterisks, no bullet points.
- Hard limit: 120 words. If the question needs a list, use numbered lines.
- Never open with "Based on..." or "I think..." — state the answer directly.

Match data:
{context}

User question: {question}"""



class AskRequest(BaseModel):
    question:     str
    match_id:     Optional[int] = None
    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None
    league_id:    Optional[int] = None
    season_id:    Optional[int] = None
    provider:     Optional[Literal["groq", "gemini", "auto"]] = "auto"
    model:        Optional[str] = None
    # Multi-turn conversation history: list of {"role": "user"|"assistant", "content": str}
    history:      Optional[list] = []
    # Attachments (images or text files)
    file_base64:  Optional[str] = None
    file_mime:    Optional[str] = None


# ─── DB Context Fetcher ────────────────────────────────────────────────────────

def _fetch_db_context(cur, home_team_id: int, away_team_id: int,
                      season_id: int = None, league_id: int = None,
                      match_id: int = None) -> dict:
    """
    Fetch the full rich DB context the engines use, from ALL relevant tables:
      matches, league_standings, team_squad_stats, team_venue_stats,
      player_stats, player_injuries, team_clubelo, match_odds.

    season_id is used for season-scoped queries. If the current season has
    fewer than 3 completed matches, we automatically fall back to include the
    previous season so the AI always has real data to work with.
    """

    # ── 1. Recent form (last 6 completed matches, any season) ──────────────
    def recent_results(tid: int, n: int = 6):
        try:
            cur.execute("""
                SELECT
                    m.match_date,
                    CASE WHEN m.home_team_id = %(t)s THEN at2.name ELSE ht2.name END AS opponent,
                    CASE WHEN m.home_team_id = %(t)s THEN 'H' ELSE 'A'           END AS venue,
                    CASE WHEN m.home_team_id = %(t)s THEN m.home_score ELSE m.away_score END AS ts,
                    CASE WHEN m.home_team_id = %(t)s THEN m.away_score ELSE m.home_score END AS os,
                    l.name AS league_name
                FROM matches m
                JOIN teams ht2 ON ht2.id = m.home_team_id
                JOIN teams at2 ON at2.id = m.away_team_id
                LEFT JOIN leagues l ON l.id = m.league_id
                WHERE (m.home_team_id = %(t)s OR m.away_team_id = %(t)s)
                  AND m.home_score IS NOT NULL
                ORDER BY m.match_date DESC
                LIMIT %(n)s
            """, {"t": tid, "n": n})
            out = []
            for r in cur.fetchall():
                ts = r["ts"] or 0; os_ = r["os"] or 0
                out.append({
                    "date":     str(r["match_date"]) if r["match_date"] else None,
                    "opponent": r["opponent"],
                    "venue":    r["venue"],
                    "league":   r.get("league_name") or "",
                    "score":    f"{ts}-{os_}",
                    "result":   "W" if ts > os_ else ("L" if ts < os_ else "D"),
                })
            return out
        except Exception as e:
            log.warning("recent_results failed: %s", e)
            try: cur.connection.rollback()
            except Exception: pass
            return []

    # ── 2. Season stats — current season first, fallback to broader range ──
    def season_stats(tid: int, sid: int = None):
        """
        Stats filtered to the current season. If fewer than 3 games found
        (early season / brand-new season), automatically widens to include
        the previous season as well, so the AI never sees empty data.
        """
        try:
            base_sql = """
                SELECT
                    COUNT(*) AS played,
                    SUM(CASE WHEN (home_team_id=%(t)s AND home_score>away_score)
                              OR  (away_team_id=%(t)s AND away_score>home_score) THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN home_score=away_score THEN 1 ELSE 0 END) AS draws,
                    SUM(CASE WHEN (home_team_id=%(t)s AND home_score<away_score)
                              OR  (away_team_id=%(t)s AND away_score<home_score) THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN home_team_id=%(t)s THEN home_score ELSE away_score END) AS gf,
                    SUM(CASE WHEN home_team_id=%(t)s THEN away_score ELSE home_score END) AS ga,
                    SUM(CASE WHEN (home_team_id=%(t)s AND away_score=0)
                              OR  (away_team_id=%(t)s AND home_score=0) THEN 1 ELSE 0 END) AS cs
                FROM matches
                WHERE (home_team_id=%(t)s OR away_team_id=%(t)s)
                  AND home_score IS NOT NULL
                  {season_filter}
            """

            def _run(extra_filter, params):
                cur.execute(base_sql.format(season_filter=extra_filter), params)
                r = cur.fetchone()
                return r if (r and r["played"]) else None

            r = None
            note = "all seasons"

            # Try current season
            if sid:
                r = _run("AND season_id = %(s)s", {"t": tid, "s": sid})
                if r and r["played"]:
                    note = "current season"

            # If fewer than 3 games in current season, include previous season too
            if (not r or r["played"] < 3) and sid:
                r2 = _run(
                    "AND season_id IN (%(s)s, %(s)s - 1)",
                    {"t": tid, "s": sid}
                )
                if r2 and r2["played"] >= (r["played"] if r else 0):
                    r = r2
                    note = "current + previous season"

            # Last resort: all seasons (always has data)
            if not r or not r["played"]:
                r = _run("", {"t": tid})
                note = "all seasons (fallback)"

            if not r or not r["played"]:
                return {}

            p = r["played"]
            return {
                "played":            p,
                "wins":              r["wins"]   or 0,
                "draws":             r["draws"]  or 0,
                "losses":            r["losses"] or 0,
                "goals_for":         r["gf"]     or 0,
                "goals_against":     r["ga"]     or 0,
                "avg_goals_for":     round((r["gf"] or 0) / p, 2),
                "avg_goals_against": round((r["ga"] or 0) / p, 2),
                "clean_sheets":      r["cs"]     or 0,
                "data_scope":        note,
            }
        except Exception as e:
            log.warning("season_stats failed: %s", e)
            try: cur.connection.rollback()
            except Exception: pass
            return {}

    # ── 3. Head-to-head (last 8 meetings, all time) ────────────────────────
    def h2h(htid: int, atid: int, n: int = 8):
        try:
            cur.execute("""
                SELECT
                    m.match_date,
                    ht4.name AS home_team, at4.name AS away_team,
                    m.home_team_id, m.home_score, m.away_score,
                    l.name AS league_name
                FROM matches m
                JOIN teams ht4 ON ht4.id = m.home_team_id
                JOIN teams at4 ON at4.id = m.away_team_id
                LEFT JOIN leagues l ON l.id = m.league_id
                WHERE ((m.home_team_id=%(h)s AND m.away_team_id=%(a)s)
                    OR (m.home_team_id=%(a)s AND m.away_team_id=%(h)s))
                  AND m.home_score IS NOT NULL
                ORDER BY m.match_date DESC
                LIMIT %(n)s
            """, {"h": htid, "a": atid, "n": n})
            rows = cur.fetchall()
            hw = dr = aw = 0
            matches = []
            for r in rows:
                hs = r["home_score"] or 0; as_ = r["away_score"] or 0
                if hs > as_:
                    winner = r["home_team"]
                    if r["home_team_id"] == htid: hw += 1
                    else: aw += 1
                elif hs < as_:
                    winner = r["away_team"]
                    if r["home_team_id"] == atid: hw += 1
                    else: aw += 1
                else:
                    winner = "Draw"; dr += 1
                matches.append({
                    "date":   str(r["match_date"]) if r["match_date"] else None,
                    "home":   r["home_team"],
                    "away":   r["away_team"],
                    "score":  f"{hs}-{as_}",
                    "winner": winner,
                    "league": r.get("league_name") or "",
                })
            return {"summary": {"home_wins": hw, "draws": dr, "away_wins": aw}, "matches": matches}
        except Exception as e:
            log.warning("h2h failed: %s", e)
            try: cur.connection.rollback()
            except Exception: pass
            return {}

    # ── 4. League standings (season + league filtered, with fallback) ───────
    def standings(tid: int, sid: int = None, lid: int = None):
        try:
            params: dict = {"t": tid}
            filters = ["ls.team_id = %(t)s"]
            if sid:
                filters.append("ls.season_id = %(s)s"); params["s"] = sid
            if lid:
                filters.append("ls.league_id = %(l)s"); params["l"] = lid
            where = " AND ".join(filters)
            cur.execute(f"""
                SELECT ls.wins, ls.ties, ls.losses, ls.games,
                       ls.goals_for, ls.goals_against, ls.points, ls.rank,
                       ls.points_avg
                FROM league_standings ls
                WHERE {where}
                ORDER BY ls.season_id DESC LIMIT 1
            """, params)
            r = cur.fetchone()
            # Fallback: no season/league filter
            if not r:
                cur.execute("""
                    SELECT ls.wins, ls.ties, ls.losses, ls.games,
                           ls.goals_for, ls.goals_against, ls.points, ls.rank,
                           ls.points_avg
                    FROM league_standings ls
                    WHERE ls.team_id = %s
                    ORDER BY ls.season_id DESC LIMIT 1
                """, (tid,))
                r = cur.fetchone()
            if not r:
                return {}
            return {
                "rank":          r["rank"],
                "points":        r["points"],
                "points_avg":    r.get("points_avg"),
                "played":        r["games"],
                "wins":          r["wins"],
                "draws":         r["ties"],
                "losses":        r["losses"],
                "goals_for":     r["goals_for"],
                "goals_against": r["goals_against"],
                "goal_diff":     (r["goals_for"] or 0) - (r["goals_against"] or 0),
            }
        except Exception as e:
            log.warning("standings failed: %s", e)
            try: cur.connection.rollback()
            except Exception: pass
            return {}

    # ── 5. Squad stats (possession, shooting %, save %) ────────────────────
    def squad_stats(tid: int, sid: int = None):
        """
        team_squad_stats splits: 'for' (attacking) and 'against' (defensive).
        'overall' does NOT exist. goals/assists columns are NULL — they live
        inside the standard_stats/shooting/goalkeeping JSONB columns.
        season_id = 1 is current (2025-26). Falls back across seasons if empty.
        """
        try:
            params: dict = {"t": tid}
            season_filter = "AND tss.season_id = %(s)s" if sid else ""
            if sid:
                params["s"] = sid
            # Fetch 'for' split (attacking stats) — this is the primary row
            params["sp"] = "for"
            cur.execute(f"""
                SELECT tss.possession, tss.avg_age, tss.players_used,
                       tss.games, tss.minutes_90s,
                       tss.shooting, tss.goalkeeping, tss.misc_stats,
                       tss.standard_stats, tss.playing_time
                FROM team_squad_stats tss
                WHERE tss.team_id = %(t)s
                  AND tss.split = %(sp)s
                  {season_filter}
                ORDER BY tss.season_id DESC LIMIT 1
            """, params)
            r_for = cur.fetchone()
            # If nothing for this season, try without season filter
            if not r_for and sid:
                cur.execute("""
                    SELECT tss.possession, tss.avg_age, tss.players_used,
                           tss.games, tss.minutes_90s,
                           tss.shooting, tss.goalkeeping, tss.misc_stats,
                           tss.standard_stats, tss.playing_time
                    FROM team_squad_stats tss
                    WHERE tss.team_id = %s AND tss.split = 'for'
                    ORDER BY tss.season_id DESC LIMIT 1
                """, (tid,))
                r_for = cur.fetchone()
            # Also fetch 'against' split for GK/defensive stats
            params2 = {"t": tid, "sp": "against"}
            sf2 = "AND tss.season_id = %(s)s" if sid else ""
            if sid: params2["s"] = sid
            cur.execute(f"""
                SELECT tss.goalkeeping, tss.shooting
                FROM team_squad_stats tss
                WHERE tss.team_id = %(t)s
                  AND tss.split = %(sp)s
                  {sf2}
                ORDER BY tss.season_id DESC LIMIT 1
            """, params2)
            r_ag = cur.fetchone()
            if not r_for:
                return {}
            def _jget(blob, *keys):
                if not isinstance(blob, dict): return None
                for k in keys:
                    v = blob.get(k)
                    if v is not None:
                        try: return float(v)
                        except (TypeError, ValueError): pass
                return None
            sh  = r_for.get("shooting")  or {}
            ss  = r_for.get("standard_stats") or {}
            ms  = r_for.get("misc_stats") or {}
            gk_ag = (r_ag.get("goalkeeping") if r_ag else None) or {}
            return {
                "possession":          r_for.get("possession"),
                "avg_age":             r_for.get("avg_age"),
                "players_used":        r_for.get("players_used"),
                "games":               r_for.get("games"),
                # Goals / assists come from standard_stats JSONB
                "goals":               _jget(ss, "goals"),
                "assists":             _jget(ss, "assists"),
                "goals_per90":         _jget(ss, "goals_per90"),
                "assists_per90":       _jget(ss, "assists_per90"),
                # Shooting from shooting JSONB
                "shots_on_target_pct": _jget(sh, "shots_on_target_pct"),
                "goals_per_shot":      _jget(sh, "goals_per_shot"),
                "shots_per90":         _jget(sh, "shots_per90"),
                # GK from against split goalkeeping JSONB
                "save_pct":            _jget(gk_ag, "gk_save_pct"),
                "clean_sheet_pct":     _jget(gk_ag, "gk_clean_sheets_pct"),
                # Discipline
                "yellow_cards":        _jget(ms, "cards_yellow") or _jget(ms, "yellow_cards"),
                "red_cards":           _jget(ms, "cards_red")    or _jget(ms, "red_cards"),
            }
        except Exception as e:
            log.warning("squad_stats failed: %s", e)
            try: cur.connection.rollback()
            except Exception: pass
            return {}

    # ── 6. Venue stats — home/away splits ──────────────────────────────────
    def venue_stats(tid: int, sid: int = None):
        try:
            params: dict = {"t": tid}
            season_filter = "AND tvs.season_id = %(s)s" if sid else ""
            if sid:
                params["s"] = sid
            cur.execute(f"""
                SELECT tvs.venue, tvs.games, tvs.wins, tvs.draws, tvs.losses,
                       tvs.goals_for, tvs.goals_against,
                       tvs.goal_diff, tvs.points
                FROM team_venue_stats tvs
                WHERE tvs.team_id = %(t)s
                  AND tvs.games > 0
                  {season_filter}
                ORDER BY tvs.season_id DESC, tvs.venue
            """, params)
            rows = cur.fetchall()
            result = {}
            for r in rows:
                split = r.get("venue") or r.get("split")  # column is 'venue' in DB
                if split not in ("home", "away"):
                    continue
                g = r["games"] or 1
                result[split] = {
                    "games":         r["games"],
                    "wins":          r["wins"],
                    "draws":         r.get("draws") or 0,
                    "losses":        r["losses"],
                    "goals_for":     r["goals_for"],
                    "goals_against": r["goals_against"],
                    "goal_diff":     r.get("goal_diff"),
                    "points":        r.get("points"),
                    "avg_gf":        round((r["goals_for"]  or 0) / g, 2),
                    "avg_ga":        round((r["goals_against"] or 0) / g, 2),
                    "win_rate":      round((r["wins"] or 0) / g, 2),
                }
            return result
        except Exception as e:
            log.warning("venue_stats failed: %s", e)
            try: cur.connection.rollback()
            except Exception: pass
            return {}

    # ── 7. Top scorers ──────────────────────────────────────────────────────
    def top_scorers(tid: int, sid: int = None):
        """
        player_stats.season_id is NULL for all rows — cannot filter by it.
        Uses the v_top_scorers view (which already joins team name + season)
        then falls back to raw player_stats without season filter.
        """
        try:
            # Try the view first — it resolves team name so we match by team_id via join
            cur.execute("""
                SELECT vts.player_name, vts.goals, vts.assists, vts.minutes, vts.games
                FROM v_top_scorers vts
                JOIN teams t ON t.name = vts.team
                WHERE t.id = %s
                ORDER BY vts.goals DESC, vts.assists DESC
                LIMIT 5
            """, (tid,))
            rows = cur.fetchall()
            if not rows:
                # Fallback: direct player_stats without season filter (season_id is NULL)
                cur.execute("""
                    SELECT ps.player_name, ps.goals, ps.assists, ps.minutes, ps.games
                    FROM player_stats ps
                    WHERE ps.team_id = %s
                      AND ps.goals IS NOT NULL
                    ORDER BY ps.goals DESC, ps.assists DESC
                    LIMIT 5
                """, (tid,))
                rows = cur.fetchall()
            return [
                {"name": r["player_name"], "goals": r["goals"],
                 "assists": r.get("assists"), "minutes": r.get("minutes"),
                 "games": r.get("games")}
                for r in rows
            ]
        except Exception as e:
            log.warning("top_scorers failed: %s", e)
            try: cur.connection.rollback()
            except Exception: pass
            return []

    # ── 8. Current injuries / suspensions ──────────────────────────────────
    def injuries(tid: int):
        """
        player_injuries has: id, team_id, player_name, injury_type,
        return_date, raw_data (jsonb), scraped_at. NO status column.
        Pulls latest batch; falls back to all-time most recent if none in 30d.
        """
        try:
            cur.execute("""
                SELECT pi.player_name, pi.injury_type, pi.return_date,
                       pi.raw_data, pi.scraped_at
                FROM player_injuries pi
                WHERE pi.team_id = %s
                  AND pi.scraped_at >= NOW() - INTERVAL '30 days'
                ORDER BY pi.scraped_at DESC
                LIMIT 8
            """, (tid,))
            rows = cur.fetchall()
            if not rows:
                cur.execute("""
                    SELECT pi.player_name, pi.injury_type, pi.return_date,
                           pi.raw_data, pi.scraped_at
                    FROM player_injuries pi
                    WHERE pi.team_id = %s
                    ORDER BY pi.scraped_at DESC
                    LIMIT 8
                """, (tid,))
                rows = cur.fetchall()
            result = []
            for r in rows:
                import json as _jj
                rd = r.get("raw_data") or {}
                if isinstance(rd, str):
                    try: rd = _jj.loads(rd)
                    except Exception: rd = {}
                inj_type = r.get("injury_type") or rd.get("Injury") or "Unknown"
                ret_date = r.get("return_date") or rd.get("Return") or None
                result.append({
                    "name":   r["player_name"],
                    "type":   inj_type if inj_type != "Unknown" else "injury/concern",
                    "return": str(ret_date) if ret_date and str(ret_date) != "EMPTY" else None,
                })
            return result
        except Exception as e:
            log.warning("injuries failed: %s", e)
            try: cur.connection.rollback()
            except Exception: pass
            return []

    # ── 9. ClubElo ratings (most recent available for each team) ───────────
    def clubelo(htid: int, atid: int):
        """
        Fetch latest ELO ratings + the match-specific win probability forecast
        from team_clubelo (the same data the DC engine blends with its Poisson).
        Falls back to latest available rating if no fixture-specific row exists.
        """
        try:
            result = {}

            def _latest_elo(tid: int, label: str):
                """
                team_clubelo: elo column is always NULL — ELO value is inside
                raw_data JSONB. Structure: {"raw": {"Elo": "1850", "Away": "...",
                "GD=0": "0.29", "GD=1": "0.24", ...}}.
                """
                cur.execute("""
                    SELECT elo, elo_date, raw_data
                    FROM team_clubelo
                    WHERE team_id = %s
                    ORDER BY elo_date DESC LIMIT 1
                """, (tid,))
                r = cur.fetchone()
                if not r:
                    return
                result[f"{label}_elo_date"] = str(r["elo_date"]) if r["elo_date"] else None
                # Parse raw_data for the actual ELO value
                import json as _jj2
                rd = r.get("raw_data") or {}
                if isinstance(rd, str):
                    try: rd = _jj2.loads(rd)
                    except Exception: rd = {}
                inner = rd.get("raw", rd)
                # ELO may be stored as "Elo" key, or fall back to r["elo"] if not NULL
                elo_val = None
                if isinstance(inner, dict):
                    elo_val = inner.get("Elo") or inner.get("elo")
                if elo_val is None and r.get("elo") is not None:
                    elo_val = r["elo"]
                if elo_val is not None:
                    try: result[f"{label}_elo"] = float(elo_val)
                    except (TypeError, ValueError): pass
                rd = r.get("raw_data") or {}
                if isinstance(rd, str):
                    import json as _json
                    try: rd = _json.loads(rd)
                    except Exception: rd = {}
                inner = rd.get("raw", rd)
                if inner:
                    def _pf(key):
                        v = inner.get(key)
                        try: return float(v) if v is not None else None
                        except (TypeError, ValueError): return None
                    # Rebuild 1X2 from goal-difference probability buckets
                    hw = sum(filter(None, [_pf(f"GD={i}") for i in range(1, 6)])) or 0
                    hw += _pf("GD>5") or 0
                    dr  = _pf("GD=0") or 0
                    aw  = sum(filter(None, [_pf(f"GD={-i}") for i in range(1, 6)])) or 0
                    aw += _pf("GD<-5") or 0
                    total = hw + dr + aw
                    if total > 0.5 and label == "home":   # only store once from home row
                        result["clubelo_home_win_pct"] = round(hw / total * 100, 1)
                        result["clubelo_draw_pct"]     = round(dr / total * 100, 1)
                        result["clubelo_away_win_pct"] = round(aw / total * 100, 1)

            _latest_elo(htid, "home")
            _latest_elo(atid, "away")

            if "home_elo" in result and "away_elo" in result:
                result["elo_gap"] = round(result["home_elo"] - result["away_elo"], 1)

            return result
        except Exception as e:
            log.warning("clubelo failed: %s", e)
            try: cur.connection.rollback()
            except Exception: pass
            return {}

    # ── 10. Bookmaker odds (B365 + Over/Under, latest scraped) ─────────────
    def match_odds(mid: int = None, htid: int = None, atid: int = None):
        """
        Fetch latest bookmaker odds from match_odds.
        Tries match_id lookup first; falls back to home+away team lookup.
        Converts raw B365 decimal odds to implied probability percentages.
        """
        try:
            row = None
            if mid:
                cur.execute("""
                    SELECT b365_home_win, b365_draw, b365_away_win, raw_data
                    FROM match_odds
                    WHERE match_id = %s
                    ORDER BY scraped_at DESC LIMIT 1
                """, (mid,))
                row = cur.fetchone()

            if not row and htid and atid:
                cur.execute("""
                    SELECT mo.b365_home_win, mo.b365_draw, mo.b365_away_win, mo.raw_data
                    FROM match_odds mo
                    JOIN matches m ON m.id = mo.match_id
                    WHERE (m.home_team_id = %s AND m.away_team_id = %s)
                      AND m.home_score IS NULL
                    ORDER BY mo.scraped_at DESC LIMIT 1
                """, (htid, atid))
                row = cur.fetchone()

            if not row:
                return {}

            def _implied(decimal_odds):
                """Convert decimal odds to implied probability %."""
                try:
                    v = float(decimal_odds)
                    return round((1.0 / v) * 100, 1) if v > 1.0 else None
                except (TypeError, ValueError, ZeroDivisionError):
                    return None

            hw_prob = _implied(row.get("b365_home_win"))
            dr_prob = _implied(row.get("b365_draw"))
            aw_prob = _implied(row.get("b365_away_win"))

            # Margin-normalised implied probs (remove the bookmaker's overround)
            margin = (hw_prob or 0) + (dr_prob or 0) + (aw_prob or 0)
            if margin > 0:
                hw_norm = round(hw_prob / margin * 100, 1) if hw_prob else None
                dr_norm = round(dr_prob / margin * 100, 1) if dr_prob else None
                aw_norm = round(aw_prob / margin * 100, 1) if aw_prob else None
            else:
                hw_norm = dr_norm = aw_norm = None

            result: dict = {
                "b365_home_win":     row.get("b365_home_win"),
                "b365_draw":         row.get("b365_draw"),
                "b365_away_win":     row.get("b365_away_win"),
                "implied_home_pct":  hw_prob,
                "implied_draw_pct":  dr_prob,
                "implied_away_pct":  aw_prob,
                "norm_home_pct":     hw_norm,
                "norm_draw_pct":     dr_norm,
                "norm_away_pct":     aw_norm,
                "overround_pct":     round(margin - 100, 1) if margin > 0 else None,
            }

            # Additional markets from raw_data JSON
            # raw_data keys (from scraper): B365H, B365D, B365A, BbMx>2.5,
            # BbMxAHH (Asian Handicap Home), AH (handicap size), etc.
            import json as _json2
            rd = row.get("raw_data") or {}
            if isinstance(rd, str):
                try: rd = _json2.loads(rd)
                except Exception: rd = {}
            if rd:
                def _sf(key):
                    try: return float(rd[key]) if rd.get(key) not in (None, "", "0", 0) else None
                    except (TypeError, ValueError): return None
                # Over/Under 2.5 — try multiple possible key names
                o25 = _sf("BbMx>2.5") or _sf("B365>2.5") or _sf("Bb1X2") or _sf("AvgC>2.5")
                u25 = _sf("BbMx<2.5") or _sf("B365<2.5") or _sf("AvgC<2.5")
                if o25:
                    result["over25_odds"]         = o25
                    result["implied_over25_pct"]  = round((1.0 / o25) * 100, 1)
                if u25:
                    result["under25_odds"]        = u25
                    result["implied_under25_pct"] = round((1.0 / u25) * 100, 1)
                # Asian handicap
                ah_size = _sf("AH") or _sf("AHh")
                ah_home = _sf("BbMxAHH") or _sf("LAHH") or _sf("AHH")
                ah_away = _sf("BbMxAHA") or _sf("LAHA") or _sf("AHA")
                if ah_size is not None: result["asian_hdcp"]    = ah_size
                if ah_home:            result["ah_home_odds"]  = ah_home
                if ah_away:            result["ah_away_odds"]  = ah_away
                # BTTS (if available)
                btts_y = _sf("BbMxBCH") or _sf("BTSSH")
                if btts_y: result["btts_yes_odds"] = btts_y

            return result
        except Exception as e:
            log.warning("match_odds failed: %s", e)
            try: cur.connection.rollback()
            except Exception: pass
            return {}

    # ── Assemble everything ─────────────────────────────────────────────────
    elo_data  = clubelo(home_team_id, away_team_id)
    odds_data = match_odds(mid=match_id, htid=home_team_id, atid=away_team_id)

    return {
        "home_form":         recent_results(home_team_id),
        "away_form":         recent_results(away_team_id),
        "home_season":       season_stats(home_team_id, season_id),
        "away_season":       season_stats(away_team_id, season_id),
        "h2h":               h2h(home_team_id, away_team_id),
        "home_standings":    standings(home_team_id, season_id, league_id),
        "away_standings":    standings(away_team_id, season_id, league_id),
        "home_squad":        squad_stats(home_team_id, season_id),
        "away_squad":        squad_stats(away_team_id, season_id),
        "home_venue":        venue_stats(home_team_id, season_id),
        "away_venue":        venue_stats(away_team_id, season_id),
        "home_top_scorers":  top_scorers(home_team_id, season_id),
        "away_top_scorers":  top_scorers(away_team_id, season_id),
        "home_injuries":     injuries(home_team_id),
        "away_injuries":     injuries(away_team_id),
        "clubelo":           elo_data,
        "odds":              odds_data,
    }


# ─── Context Builder ───────────────────────────────────────────────────────────

def _build_context(match_data: dict, db_ctx: dict) -> str:
    m         = match_data.get("match", {})
    con       = match_data.get("consensus", {})
    eng       = match_data.get("engines", {})
    mkt       = match_data.get("markets", {})
    bets      = match_data.get("best_bets", [])
    weights   = match_data.get("weights_used", {})
    agreement = match_data.get("agreement", "unknown")

    home = m.get("home_team", "Home")
    away = m.get("away_team", "Away")

    def pct(v):
        try: return f"{round(float(v) * 100, 1)}%"
        except: return "N/A"

    def fmt_engine(name, data):
        if not data: return f"  {name}: no data"
        return (f"  {name}: {pct(data.get('home_win'))} Home | "
                f"{pct(data.get('draw'))} Draw | "
                f"{pct(data.get('away_win'))} Away -> Predicts: {data.get('predicted_outcome', 'N/A')}")

    def fmt_form(form_list, team_name):
        if not form_list:
            return f"  {team_name}: no recent results available"
        lines = [f"  {team_name} last {len(form_list)} results:"]
        for f in form_list:
            league_tag = f" ({f.get('league','')})" if f.get("league") else ""
            lines.append(f"    {f.get('date','?')} [{f.get('venue','?')}] vs {f.get('opponent','?')}"
                         f"{league_tag} {f.get('score','?')} ({f.get('result','?')})")
        return "\n".join(lines)

    def fmt_stats(stats, team_name):
        if not stats:
            return f"  {team_name}: no season stats available"
        scope = f" [{stats.get('data_scope','')}]" if stats.get("data_scope") else ""
        return (f"  {team_name}{scope}: P{stats.get('played',0)} "
                f"W{stats.get('wins',0)} D{stats.get('draws',0)} L{stats.get('losses',0)} | "
                f"GF {stats.get('goals_for',0)} GA {stats.get('goals_against',0)} | "
                f"Avg scored {stats.get('avg_goals_for','?')} conceded {stats.get('avg_goals_against','?')} | "
                f"Clean sheets: {stats.get('clean_sheets',0)}")

    def fmt_standings(st, team_name):
        if not st:
            return f"  {team_name}: no standings data"
        pts_avg = f" | {st.get('points_avg'):.2f} pts/game" if st.get("points_avg") else ""
        return (f"  {team_name}: Rank {st.get('rank','?')} | "
                f"{st.get('points','?')} pts{pts_avg} | "
                f"W{st.get('wins',0)} D{st.get('draws',0)} L{st.get('losses',0)} | "
                f"GD {st.get('goal_diff','?')}")

    def fmt_h2h(h2h_data, home_name, away_name):
        if not h2h_data or not h2h_data.get("matches"):
            return "  No H2H data available"
        s = h2h_data.get("summary", {})
        lines = [f"  Last {len(h2h_data.get('matches',[]))} meetings: "
                 f"{home_name} wins {s.get('home_wins',0)} | "
                 f"Draws {s.get('draws',0)} | "
                 f"{away_name} wins {s.get('away_wins',0)}"]
        for match in h2h_data.get("matches", [])[:6]:
            league_tag = f" ({match.get('league','')})" if match.get("league") else ""
            lines.append(f"    {match.get('date','?')}{league_tag}: "
                         f"{match.get('home','?')} vs {match.get('away','?')} "
                         f"{match.get('score','?')} -> {match.get('winner','?')}")
        return "\n".join(lines)

    def fmt_squad(sq, team_name):
        if not sq:
            return f"  {team_name}: no squad stats available"
        parts = []
        if sq.get("possession") is not None:
            parts.append(f"Possession {sq['possession']}%")
        if sq.get("avg_age"):
            parts.append(f"Avg age {sq['avg_age']}")
        if sq.get("shots_on_target_pct") is not None:
            parts.append(f"SoT% {sq['shots_on_target_pct']}")
        if sq.get("goals_per_shot") is not None:
            parts.append(f"Goals/shot {sq['goals_per_shot']}")
        if sq.get("save_pct") is not None:
            parts.append(f"Save% {sq['save_pct']}")
        if sq.get("clean_sheet_pct") is not None:
            parts.append(f"CS% {sq['clean_sheet_pct']}")
        return f"  {team_name}: {' | '.join(parts)}" if parts else f"  {team_name}: data present but all null"

    def fmt_venue(venue_dict, team_name):
        if not venue_dict:
            return f"  {team_name}: no venue split data"
        lines = []
        for split in ("home", "away"):
            v = venue_dict.get(split)
            if not v:
                continue
            xg_tag = f" | xG for {v['xg_for']} against {v['xg_against']}" if v.get("xg_for") else ""
            lines.append(
                f"  {team_name} ({split.upper()}): {v['games']}G "
                f"W{v['wins']} D{v['draws']} L{v['losses']} | "
                f"Avg {v['avg_gf']}-{v['avg_ga']}{xg_tag}"
            )
        return "\n".join(lines) if lines else f"  {team_name}: no venue data"

    def fmt_scorers(scorers, team_name):
        if not scorers:
            return f"  {team_name}: no scorer data"
        parts = [f"{s['name']} {s['goals']}G/{s.get('assists','?')}A" for s in scorers]
        return f"  {team_name}: {', '.join(parts)}"

    def fmt_injuries(inj_list, team_name):
        if not inj_list:
            return f"  {team_name}: no injury/suspension concerns reported"
        parts = []
        for i in inj_list:
            inj_type = i.get("type") or "injury/concern"
            ret_str  = f", back {i['return']}" if i.get("return") else ""
            parts.append(f"{i['name']} ({inj_type}{ret_str})")
        return f"  {team_name}: {', '.join(parts)}"

    def fmt_elo(elo):
        if not elo:
            return "  No ClubElo data available"
        lines = []
        if elo.get("home_elo"):
            lines.append(f"  {home}: ELO {elo['home_elo']} (as of {elo.get('home_elo_date','?')})")
        if elo.get("away_elo"):
            lines.append(f"  {away}: ELO {elo['away_elo']} (as of {elo.get('away_elo_date','?')})")
        if elo.get("elo_gap") is not None:
            lines.append(f"  ELO gap (home - away): {elo['elo_gap']}")
        if elo.get("clubelo_home_win_pct") is not None:
            lines.append(f"  ClubElo forecast: {home} {elo['clubelo_home_win_pct']}% | "
                         f"Draw {elo['clubelo_draw_pct']}% | {away} {elo['clubelo_away_win_pct']}%")
        return "\n".join(lines) if lines else "  ClubElo: no data"

    def fmt_odds(odds):
        if not odds:
            return "  No bookmaker odds available"
        lines = []
        if odds.get("b365_home_win"):
            lines.append(
                f"  B365 odds: {home} {odds['b365_home_win']} | "
                f"Draw {odds['b365_draw']} | {away} {odds['b365_away_win']}"
            )
        if odds.get("norm_home_pct") is not None:
            lines.append(
                f"  Implied (margin-adjusted): {home} {odds['norm_home_pct']}% | "
                f"Draw {odds['norm_draw_pct']}% | {away} {odds['norm_away_pct']}%"
                + (f" (overround {odds['overround_pct']}%)" if odds.get("overround_pct") else "")
            )
        if odds.get("over25_odds"):
            lines.append(
                f"  Over 2.5 odds: {odds['over25_odds']} "
                f"({odds.get('implied_over25_pct','?')}% implied)"
            )
        if odds.get("under25_odds"):
            lines.append(
                f"  Under 2.5 odds: {odds['under25_odds']} "
                f"({odds.get('implied_under25_pct','?')}% implied)"
            )
        if odds.get("asian_hdcp") is not None:
            lines.append(f"  Asian handicap: {odds['asian_hdcp']} | "
                         f"Home {odds.get('ah_home_odds','?')} / Away {odds.get('ah_away_odds','?')}")
        if odds.get("btts_yes_odds"):
            lines.append(f"  BTTS Yes odds: {odds['btts_yes_odds']}")
        return "\n".join(lines) if lines else "  Odds fetched but no displayable fields"

    # ── Assemble the full context string ───────────────────────────────────
    lines = [
        "=" * 60,
        f"MATCH: {home} vs {away}",
        f"League: {m.get('league', 'Unknown')} | Season: {m.get('season', 'Unknown')}",
        "=" * 60,
        "",
        "━━━ SECTION 1: RAW DATABASE DATA ━━━",
        "",
        "RECENT FORM (last 6 matches, any competition):",
        fmt_form(db_ctx.get("home_form", []), home),
        fmt_form(db_ctx.get("away_form", []), away),
        "",
        "SEASON STATISTICS:",
        fmt_stats(db_ctx.get("home_season", {}), home),
        fmt_stats(db_ctx.get("away_season", {}), away),
        "",
        "LEAGUE STANDINGS:",
        fmt_standings(db_ctx.get("home_standings", {}), home),
        fmt_standings(db_ctx.get("away_standings", {}), away),
        "",
        "HEAD-TO-HEAD HISTORY (last 8 meetings):",
        fmt_h2h(db_ctx.get("h2h", {}), home, away),
        "",
        "SQUAD STATS (possession, shooting, goalkeeping):",
        fmt_squad(db_ctx.get("home_squad", {}), home),
        fmt_squad(db_ctx.get("away_squad", {}), away),
        "",
        "VENUE SPLITS (home/away performance):",
        fmt_venue(db_ctx.get("home_venue", {}), home),
        fmt_venue(db_ctx.get("away_venue", {}), away),
        "",
        "TOP SCORERS:",
        fmt_scorers(db_ctx.get("home_top_scorers", []), home),
        fmt_scorers(db_ctx.get("away_top_scorers", []), away),
        "",
        "INJURIES / SUSPENSIONS:",
        fmt_injuries(db_ctx.get("home_injuries", []), home),
        fmt_injuries(db_ctx.get("away_injuries", []), away),
        "",
        "CLUBELO RATINGS:",
        fmt_elo(db_ctx.get("clubelo", {})),
        "",
        "BOOKMAKER ODDS:",
        fmt_odds(db_ctx.get("odds", {})),
        "",
        "━━━ SECTION 2: ENGINE PREDICTIONS ━━━",
        "",
        "CONSENSUS (blended output):",
        f"  Predicted outcome: {con.get('predicted_outcome', 'N/A')}",
        f"  Confidence: {con.get('confidence', 'N/A')} ({con.get('confidence_score', 0):.1f}/100)",
        f"  Home Win: {pct(con.get('home_win'))} | Draw: {pct(con.get('draw'))} | Away Win: {pct(con.get('away_win'))}",
        f"  Engine agreement: {agreement.upper()}",
        "",
        "INDIVIDUAL ENGINES:",
        fmt_engine("Dixon-Coles (Poisson statistical)", eng.get("dc", {})),
        fmt_engine("ML Ensemble (XGBoost + RandomForest)", eng.get("ml", {})),
        fmt_engine("Legacy Heuristic (rule-based)", eng.get("legacy", {})),
        "",
        "ENGINE WEIGHTS:",
        f"  DC: {pct(weights.get('dc'))} | ML: {pct(weights.get('ml'))} | Legacy: {pct(weights.get('legacy'))}",
        f"  Source: {weights.get('source', 'unknown')}",
        "",
    ]

    if mkt:
        lines += [
            "MARKET PROBABILITIES (from prediction engine):",
            f"  xG: {home} {mkt.get('home_xg','?')} - {away} {mkt.get('away_xg','?')}",
            f"  BTTS Yes: {pct(mkt.get('btts_yes'))} | Over 2.5: {pct(mkt.get('over_2_5'))} | Under 2.5: {pct(mkt.get('under_2_5'))}",
            f"  Double Chance 1X: {pct(mkt.get('dc_1x'))} | X2: {pct(mkt.get('dc_x2'))}",
            "",
        ]

    if bets:
        lines.append("BEST BETS (by confidence):")
        for b in bets[:5]:
            lines.append(f"  [{b.get('tier','').upper()}] {b.get('pick','?')} - "
                         f"{b.get('probability_pct', 0):.1f}% | Fair odds: {b.get('fair_odds','?')}")
        lines.append("")

    return "\n".join(lines)


# ─── Prediction Data Fetcher ──────────────────────────────────────────────────

def _fetch_prediction_data(
    conn, match_id=None, home_team_id=None, away_team_id=None,
    league_id=None, season_id=None
) -> dict:
    cur = conn.cursor()
    stored = None
    try:
        if match_id:
            cur.execute("""
                SELECT pl.*, m.home_score, m.away_score,
                       ht.name as home_team, at.name as away_team,
                       l.name as league_name
                FROM prediction_log pl
                LEFT JOIN matches m ON m.id = pl.match_id
                LEFT JOIN teams ht ON ht.id = pl.home_team_id
                LEFT JOIN teams at ON at.id = pl.away_team_id
                LEFT JOIN leagues l ON l.id = pl.league_id
                WHERE pl.match_id = %s
                ORDER BY pl.predicted_at DESC LIMIT 1
            """, (match_id,))
        elif home_team_id and away_team_id:
            cur.execute("""
                SELECT pl.*, m.home_score, m.away_score,
                       ht.name as home_team, at.name as away_team,
                       l.name as league_name
                FROM prediction_log pl
                LEFT JOIN matches m ON m.id = pl.match_id
                LEFT JOIN teams ht ON ht.id = pl.home_team_id
                LEFT JOIN teams at ON at.id = pl.away_team_id
                LEFT JOIN leagues l ON l.id = pl.league_id
                WHERE pl.home_team_id = %s AND pl.away_team_id = %s
                ORDER BY pl.predicted_at DESC LIMIT 1
            """, (home_team_id, away_team_id))
        stored = cur.fetchone()
    except Exception as e:
        log.debug("Could not fetch stored prediction: %s", e)

    try:
        from ml.consensus_engine import run_consensus
        htid = home_team_id or (stored["home_team_id"] if stored else None)
        atid = away_team_id or (stored["away_team_id"] if stored else None)
        lid  = league_id   or (stored["league_id"]    if stored else None)
        sid  = season_id   or (stored["season_id"]    if stored else None)
        if htid and atid and lid and sid:
            result = run_consensus(htid, atid, lid, sid)
            if stored and stored.get("home_score") is not None:
                result["actual_score"]       = f"{stored['home_score']}-{stored['away_score']}"
                result["actual_outcome"]     = stored.get("actual")
                result["prediction_correct"] = stored.get("correct")
            return result
    except Exception as e:
        log.warning("live consensus failed in ask endpoint: %s", e)

    if stored:
        return {
            "match": {
                "home_team":    stored.get("home_team") or "Home",
                "away_team":    stored.get("away_team") or "Away",
                "league":       stored.get("league_name") or "",
                "season":       "",
                "home_team_id": stored.get("home_team_id"),
                "away_team_id": stored.get("away_team_id"),
            },
            "consensus": {
                "home_win":          stored.get("home_win_prob"),
                "draw":              stored.get("draw_prob"),
                "away_win":          stored.get("away_win_prob"),
                "predicted_outcome": stored.get("predicted_outcome"),
                "confidence":        stored.get("confidence"),
                "confidence_score":  stored.get("confidence_score"),
            },
            "engines": {}, "markets": {}, "best_bets": [],
            "weights_used": {}, "agreement": "unknown",
        }

    return {}


# ─── LLM Callers ──────────────────────────────────────────────────────────────

async def _ask_groq(context: str, question: str, model: str, history: list = None, file_base64: str = None, file_mime: str = None) -> str:
    import httpx
    import base64
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    
    final_question = question
    if file_base64 and file_mime and ("text/" in file_mime or "json" in file_mime or "csv" in file_mime):
        try:
            decoded_text = base64.b64decode(file_base64).decode("utf-8")
            final_question += f"\n\n[Attached File Content]:\n{decoded_text[:10000]}"
        except Exception as e:
            log.warning("Failed to decode text attachment for Groq: %s", e)
            
    system_msg = SYSTEM_PROMPT.format(context=context, question="")
    messages = [{"role": "system", "content": system_msg}]
    # Inject prior conversation turns for multi-turn context
    for turn in (history or []):
        role = turn.get("role") if turn.get("role") in ("user", "assistant") else "user"
        messages.append({"role": role, "content": turn.get("content", "")})
    # Current question as fresh user message
    messages.append({"role": "user", "content": final_question})
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def _ask_gemini(context: str, question: str, model: str, history: list = None, file_base64: str = None, file_mime: str = None) -> str:
    import httpx
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    # Gemini uses a flat prompt — prepend history as plain text turns
    history_text = ""
    for turn in (history or []):
        role_label = "User" if turn.get("role") == "user" else "Assistant"
        history_text += f"{role_label}: {turn.get('content', '')}\n"
    prompt = SYSTEM_PROMPT.format(context=context, question="")
    if history_text:
        prompt += f"\nConversation so far:\n{history_text}"
    prompt += f"\nUser question: {question}"
    
    parts = [{"text": prompt}]
    if file_base64 and file_mime:
        parts.append({
            "inline_data": {
                "mime_type": file_mime,
                "data": file_base64
            }
        })
        
    url = f"{GEMINI_BASE}/{model}:generateContent?key={GEMINI_API_KEY}"
    async with httpx.AsyncClient(timeout=35.0) as client:
        resp = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": parts}],
                "generationConfig": {"maxOutputTokens": 300, "temperature": 0.3},
            },
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/models")
async def list_models():
    """Return available providers and models so the frontend can build a selector."""
    return {
        "available": AVAILABLE_MODELS,
        "defaults": {
            "provider":     "groq",
            "groq_model":   DEFAULT_GROQ_MODEL,
            "gemini_model": DEFAULT_GEMINI_MODEL,
        },
        "configured": {
            "groq":   bool(GROQ_API_KEY),
            "gemini": bool(GEMINI_API_KEY),
        },
    }


@router.post("/debug-context")
async def debug_db_context(req: AskRequest):
    """
    Debug endpoint: returns the raw DB context that would be passed to the AI,
    without calling any LLM. Use this to diagnose missing data for a match.
    """
    conn = get_connection()
    try:
        match_data = _fetch_prediction_data(
            conn,
            match_id=req.match_id,
            home_team_id=req.home_team_id,
            away_team_id=req.away_team_id,
            league_id=req.league_id,
            season_id=req.season_id,
        )
        htid    = req.home_team_id or match_data.get("match", {}).get("home_team_id")
        atid    = req.away_team_id or match_data.get("match", {}).get("away_team_id")
        sid_ctx = req.season_id    or match_data.get("match", {}).get("season_id")
        lid_ctx = req.league_id    or match_data.get("match", {}).get("league_id")
        mid_ctx = req.match_id     or match_data.get("match", {}).get("id")
        cur = conn.cursor()
        db_ctx = (
            _fetch_db_context(cur, htid, atid, season_id=sid_ctx, league_id=lid_ctx, match_id=mid_ctx)
            if htid and atid else {}
        )
    finally:
        conn.close()

    return {
        "resolved_ids": {"home_team_id": htid, "away_team_id": atid,
                         "season_id": sid_ctx, "league_id": lid_ctx, "match_id": mid_ctx},
        "data_found": {
            "home_form_games":  len(db_ctx.get("home_form", [])),
            "away_form_games":  len(db_ctx.get("away_form", [])),
            "home_season_stats": bool(db_ctx.get("home_season")),
            "away_season_stats": bool(db_ctx.get("away_season")),
            "h2h_games":        len(db_ctx.get("h2h", {}).get("matches", [])),
            "home_standings":   bool(db_ctx.get("home_standings")),
            "away_standings":   bool(db_ctx.get("away_standings")),
            "home_squad_stats": bool(db_ctx.get("home_squad")),
            "away_squad_stats": bool(db_ctx.get("away_squad")),
            "home_venue_stats": bool(db_ctx.get("home_venue")),
            "away_venue_stats": bool(db_ctx.get("away_venue")),
            "home_top_scorers": len(db_ctx.get("home_top_scorers", [])),
            "away_top_scorers": len(db_ctx.get("away_top_scorers", [])),
            "home_injuries":    len(db_ctx.get("home_injuries", [])),
            "away_injuries":    len(db_ctx.get("away_injuries", [])),
            "clubelo":          bool(db_ctx.get("clubelo")),
            "odds":             bool(db_ctx.get("odds")),
        },
        "raw_context": db_ctx,
        "context_string_preview": _build_context(match_data, db_ctx)[:2000] if match_data else "",
    }


@router.post("/ask")
async def ask_prediction(req: AskRequest):
    """
    Answer any question about a match prediction.
    The LLM reads ALL available DB data (form, H2H, stats, standings, squad stats,
    venue splits, top scorers, injuries, ClubElo ratings, bookmaker odds), makes
    its own independent assessment, compares it against the three engine predictions,
    then directly answers the user's question.
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not GROQ_API_KEY and not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="No LLM API keys configured. Set GROQ_API_KEY or GEMINI_API_KEY.",
        )

    provider     = req.provider or "auto"
    groq_model   = req.model if (provider == "groq"   and req.model) else DEFAULT_GROQ_MODEL
    gemini_model = req.model if (provider == "gemini" and req.model) else DEFAULT_GEMINI_MODEL

    # If an image was provided but Groq is selected, automatically route to Gemini
    # because standard Groq text models won't accept image inputs.
    if req.file_mime and req.file_mime.startswith("image/"):
        if provider == "groq" and GEMINI_API_KEY:
            provider = "gemini"
        elif provider == "auto" and GEMINI_API_KEY:
            provider = "gemini"

    conn = get_connection()
    try:
        match_data = {}
        if req.match_id or (req.home_team_id and req.away_team_id):
            match_data = _fetch_prediction_data(
                conn,
                match_id=req.match_id,
                home_team_id=req.home_team_id,
                away_team_id=req.away_team_id,
                league_id=req.league_id,
                season_id=req.season_id,
            ) or {}

        htid    = req.home_team_id or match_data.get("match", {}).get("home_team_id")
        atid    = req.away_team_id or match_data.get("match", {}).get("away_team_id")
        sid_ctx = req.season_id    or match_data.get("match", {}).get("season_id")
        lid_ctx = req.league_id    or match_data.get("match", {}).get("league_id")
        mid_ctx = req.match_id     or match_data.get("match", {}).get("id")

        cur = conn.cursor()
        db_ctx = (
            _fetch_db_context(
                cur, htid, atid,
                season_id=sid_ctx,
                league_id=lid_ctx,
                match_id=mid_ctx,
            )
            if htid and atid else {}
        )

    finally:
        conn.close()

    context       = _build_context(match_data, db_ctx)
    answer        = None
    used_provider = None
    used_model    = None

    if provider == "groq":
        if not GROQ_API_KEY:
            raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured.")
        try:
            answer = await _ask_groq(context, req.question.strip(), groq_model, req.history or [], file_base64=req.file_base64, file_mime=req.file_mime)
            used_provider, used_model = "groq", groq_model
        except Exception as e:
            log.warning("Groq (%s) failed: %s - trying Gemini fallback", groq_model, e)
            if GEMINI_API_KEY:
                try:
                    answer = await _ask_gemini(context, req.question.strip(), gemini_model, req.history or [], file_base64=req.file_base64, file_mime=req.file_mime)
                    used_provider, used_model = "gemini", gemini_model
                except Exception as e2:
                    log.error("Gemini fallback also failed: %s", e2)

    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured.")
        try:
            answer = await _ask_gemini(context, req.question.strip(), gemini_model, req.history or [], file_base64=req.file_base64, file_mime=req.file_mime)
            used_provider, used_model = "gemini", gemini_model
        except Exception as e:
            log.warning("Gemini (%s) failed: %s - trying Groq fallback", gemini_model, e)
            if GROQ_API_KEY:
                try:
                    answer = await _ask_groq(context, req.question.strip(), groq_model, req.history or [], file_base64=req.file_base64, file_mime=req.file_mime)
                    used_provider, used_model = "groq", groq_model
                except Exception as e2:
                    log.error("Groq fallback also failed: %s", e2)

    else:  # auto
        if GROQ_API_KEY:
            try:
                answer = await _ask_groq(context, req.question.strip(), groq_model, req.history or [], file_base64=req.file_base64, file_mime=req.file_mime)
                used_provider, used_model = "groq", groq_model
            except Exception as e:
                log.warning("Groq (%s) failed: %s - falling back to Gemini", groq_model, e)
        if answer is None and GEMINI_API_KEY:
            try:
                answer = await _ask_gemini(context, req.question.strip(), gemini_model, req.history or [], file_base64=req.file_base64, file_mime=req.file_mime)
                used_provider, used_model = "gemini", gemini_model
            except Exception as e:
                log.error("Gemini (%s) also failed: %s", gemini_model, e)


    if not answer:
        raise HTTPException(status_code=502, detail="All LLM providers failed. Please try again.")

    return {
        "answer":   answer,
        "provider": used_provider,
        "model":    used_model,
        "match":    match_data.get("match", {}),
        "context_snapshot": {
            "consensus_outcome":  match_data.get("consensus", {}).get("predicted_outcome"),
            "confidence":         match_data.get("consensus", {}).get("confidence"),
            "agreement":          match_data.get("agreement"),
            "best_bets_count":    len(match_data.get("best_bets", [])),
            "has_market_data":    bool(match_data.get("markets")),
            "has_db_context":     bool(db_ctx),
            "home_form_games":    len(db_ctx.get("home_form", [])),
            "away_form_games":    len(db_ctx.get("away_form", [])),
            "h2h_games":          len(db_ctx.get("h2h", {}).get("matches", [])),
            "has_squad_stats":    bool(db_ctx.get("home_squad") or db_ctx.get("away_squad")),
            "has_venue_stats":    bool(db_ctx.get("home_venue") or db_ctx.get("away_venue")),
            "has_clubelo":        bool(db_ctx.get("clubelo")),
            "has_odds":           bool(db_ctx.get("odds")),
            "home_injuries":      len(db_ctx.get("home_injuries", [])),
            "away_injuries":      len(db_ctx.get("away_injuries", [])),
        },
    }
