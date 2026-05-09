"""
Batch Feature Engineering — Fast Training Path
================================================
Problem: build_training_dataset() in feature_engineering.py fires
~28 SQL round-trips per match. With 500 matches × 100 ms/query ≈ 60+ minutes.

Solution: Load ALL prediction-relevant tables in 6 bulk queries, then
compute 80+ features entirely in-memory per match. Training drops to ~30 s.

This is a STANDALONE ADDITION — feature_engineering.py is untouched.
Single-match prediction (called once per request) still uses the original
build_match_features() which is fast enough for that use-case.

Public API (mirrors feature_engineering.py):
  build_training_dataset_fast(cur, skip_errors=True)
    → X, y, match_ids, errors   (same shape as original)
"""

import math
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import linregress

log = logging.getLogger(__name__)


# ─── League-style defaults (used when a league has < 10 historical matches) ────
_LEAGUE_STYLE_DEFAULTS = {
    "league_home_win_rate":        0.44,
    "league_draw_rate":            0.25,
    "league_away_win_rate":        0.31,
    "league_goals_pg":             2.70,
    "league_btts_rate":            0.50,
    "league_home_advantage_score": 1.42,
}


# ─── Helpers (mirrors feature_engineering.py) ─────────────────────────────────

def _f(val, default=0.0):
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _safe_div(num, den, default=0.0):
    try:
        return float(num) / float(den) if den and float(den) != 0 else default
    except Exception:
        return default


def _j(d, key, default=0.0):
    if not isinstance(d, dict):
        return default
    return _f(d.get(key, default), default)


import datetime

def _to_date(d):
    if not d: return None
    if isinstance(d, str):
        try: return datetime.date.fromisoformat(d[:10])
        except: return None
    if hasattr(d, "date"): return d.date()
    return d

def _filter_before(matches, before_date):
    if not before_date: return matches
    bd = _to_date(before_date)
    if not bd: return matches
    return [m for m in matches if _to_date(m.get("match_date")) and _to_date(m.get("match_date")) < bd]


# ─── Data Cache ────────────────────────────────────────────────────────────────

class DataCache:
    """
    Bulk-loads every table needed for feature engineering in 6 SQL queries.
    After construction every feature computation is pure Python — zero DB calls.
    """

    def __init__(self, cur):
        self._load(cur)

    def _load(self, cur):
        log.info("DataCache: bulk-loading all tables (memory-capped)…")

        # Identify the 2 most recent season IDs — we only need recent data.
        # Older seasons are used for prev_season_form but not for features.
        cur.execute("SELECT id FROM seasons ORDER BY name DESC LIMIT 2")
        recent_season_ids = [r["id"] for r in cur.fetchall()]
        if not recent_season_ids:
            recent_season_ids = [0]   # safe fallback
        season_placeholder = ",".join("%s" for _ in recent_season_ids)

        # 1. League standings (current seasons only) ──────────────────────────
        cur.execute(
            f"SELECT * FROM league_standings WHERE season_id IN ({season_placeholder})",
            recent_season_ids,
        )
        self.standings: Dict[Tuple, dict] = {}
        by_ls: Dict[Tuple, List] = defaultdict(list)
        for r in cur.fetchall():
            r = dict(r)
            key = (r["team_id"], r["league_id"], r["season_id"])
            if key not in self.standings:
                self.standings[key] = r
            by_ls[(r["league_id"], r["season_id"])].append(r)

        # ── Cross-league standings fallback ──────────────────────────────
        # For cross-league teams (e.g. Aston Villa in Europa League),
        # their standings are stored under their domestic league.
        # Fallback: (team_id, season_id) → best domestic standings row.
        self.standings_fallback: Dict[Tuple, dict] = {}
        for (tid, lid, sid), row in self.standings.items():
            fb_key = (tid, sid)
            if fb_key not in self.standings_fallback:
                self.standings_fallback[fb_key] = row
            else:
                # Prefer the row with more games played (more complete data)
                existing_g = self.standings_fallback[fb_key].get("games", 0) or 0
                this_g = row.get("games", 0) or 0
                if this_g > existing_g:
                    self.standings_fallback[fb_key] = row

        # Pre-compute league averages from standings
        self.league_avgs: Dict[Tuple, dict] = {}
        for (lid, sid), teams in by_ls.items():
            gf_pgs, ga_pgs, pts = [], [], []
            for t in teams:
                g = max(_f(t.get("games")), 1)
                gf_pgs.append(_safe_div(_f(t.get("goals_for")), g))
                ga_pgs.append(_safe_div(_f(t.get("goals_against")), g))
                pts.append(_f(t.get("points_avg")))
            n = len(teams) or 1
            avg_gf = sum(gf_pgs) / n if gf_pgs else 1.35
            self.league_avgs[(lid, sid)] = {
                "avg_gf_pg":       avg_gf,
                "avg_ga_pg":       sum(ga_pgs) / n if ga_pgs else avg_gf,
                "avg_pts_avg":     sum(pts)    / n if pts    else 1.35,
                "n_teams":         n,
                # Venue fallbacks — will be overwritten if team_venue_stats loaded
                "avg_home_gf_pg":  avg_gf * 1.15,
                "avg_home_ga_pg":  avg_gf * 0.85,
                "avg_away_gf_pg":  avg_gf * 0.85,
                "avg_away_ga_pg":  avg_gf * 1.15,
            }

        # 2. Squad stats (current seasons only) ───────────────────────────────
        cur.execute(
            f"""
            SELECT team_id, season_id, split,
                   players_used, avg_age, possession, games, minutes_90s,
                   goals, assists, standard_stats, goalkeeping,
                   shooting, playing_time, misc_stats
            FROM team_squad_stats
            WHERE season_id IN ({season_placeholder})
            """,
            recent_season_ids,
        )
        self.squad: Dict[Tuple, dict] = {}
        for r in cur.fetchall():
            r = dict(r)
            key = (r["team_id"], r["season_id"], r["split"])
            if key not in self.squad:
                self.squad[key] = r

        # ── Cross-league squad stats fallback ─────────────────────────────
        # When a team plays in a competition different from their domestic
        # league (e.g. Aston Villa in Europa League), their squad_stats are
        # stored under their domestic season_id, not the European season_id.
        # This fallback index maps (team_id, split) → most recent squad row
        # so we can still provide squad features for cross-league matches.
        self.squad_fallback: Dict[Tuple, dict] = {}
        for (tid, sid, split), row in self.squad.items():
            fb_key = (tid, split)
            # Keep most recent (highest season_id = most recent season)
            if fb_key not in self.squad_fallback:
                self.squad_fallback[fb_key] = row
            else:
                existing_sid = self.squad_fallback[fb_key].get("season_id", 0)
                if sid > existing_sid:
                    self.squad_fallback[fb_key] = row

        # 3. Player injuries — active injuries per team ────────────────────
        cur.execute("""
            SELECT team_id, player_name, injury_type, return_date
            FROM player_injuries
            WHERE return_date IS NULL
               OR (return_date ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                   AND return_date::date >= CURRENT_DATE - INTERVAL '7 days')
        """)
        self.injuries: Dict[int, List] = defaultdict(list)
        for r in cur.fetchall():
            self.injuries[r["team_id"]].append(dict(r))

        # 4. Player stats — top 25 by minutes per team/season ──────────────
        # Sorted by minutes played so all positions (GK, DF, MF, FW) are
        # represented proportionally, not just top scorers.
        cur.execute(
            f"""
            SELECT DISTINCT ON (team_id, season_id, player_name)
                   team_id, season_id, player_name, position, goals, assists,
                   minutes, minutes_90s, age, standard_stats
            FROM player_stats
            WHERE minutes IS NOT NULL
              AND season_id IN ({season_placeholder})
            ORDER BY team_id, season_id, player_name, minutes DESC
            """,
            recent_season_ids,
        )
        # Keep top 25 by minutes per (team, season) — covers full first-team
        _player_raw: Dict[Tuple, List] = defaultdict(list)
        for r in cur.fetchall():
            r = dict(r)
            _player_raw[(r["team_id"], r["season_id"])].append(r)
        self.players: Dict[Tuple, List] = defaultdict(list)
        for k, rows in _player_raw.items():
            self.players[k] = sorted(rows, key=lambda x: _f(x.get("minutes")),
                                     reverse=True)[:25]

        # 4. Completed matches — rolling 4-year window.
        # QUALITY FILTER: require squad stats for at least ONE team (92.6% coverage)
        # OR standings for at least ONE team (broader coverage).
        # This gives ~38-42k training samples vs ~13k with the strict standings filter.
        # Matches where NEITHER team has any data are excluded (pure noise).
        # The batch_features fallback logic handles missing standings/squad gracefully.
        cur.execute("""
            SELECT m.id, m.home_team_id, m.away_team_id, m.season_id, m.league_id,
                   m.home_score, m.away_score, m.match_date
            FROM matches m
            WHERE m.home_score IS NOT NULL AND m.away_score IS NOT NULL
              AND m.match_date >= CURRENT_DATE - INTERVAL '4 years'
              AND (
                  EXISTS (
                      SELECT 1 FROM league_standings ls
                      WHERE ls.team_id = m.home_team_id
                        AND ls.season_id = m.season_id
                  )
                  OR EXISTS (
                      SELECT 1 FROM team_squad_stats tss
                      WHERE tss.team_id = m.home_team_id
                        AND tss.season_id = m.season_id
                  )
              )
              AND (
                  EXISTS (
                      SELECT 1 FROM league_standings ls
                      WHERE ls.team_id = m.away_team_id
                        AND ls.season_id = m.season_id
                  )
                  OR EXISTS (
                      SELECT 1 FROM team_squad_stats tss
                      WHERE tss.team_id = m.away_team_id
                        AND tss.season_id = m.season_id
                  )
              )
            ORDER BY m.match_date DESC
        """)
        self.all_matches: List[dict] = [dict(r) for r in cur.fetchall()]

        # Index by team — cap at 50 per team to bound memory
        self.matches_by_team: Dict[int, List] = defaultdict(list)
        for m in self.all_matches:
            if len(self.matches_by_team[m["home_team_id"]]) < 50:
                self.matches_by_team[m["home_team_id"]].append(m)
            if len(self.matches_by_team[m["away_team_id"]]) < 50:
                self.matches_by_team[m["away_team_id"]].append(m)

        # 5. Seasons ──────────────────────────────────────────────────────────
        cur.execute("SELECT id, name FROM seasons")
        seasons = [dict(r) for r in cur.fetchall()]
        self.season_name: Dict[int, str] = {r["id"]: r["name"] for r in seasons}

        # 6. Team venue stats (home/away splits) ──────────────────────────────
        # Wrapped in try/except: if team_venue_stats is empty or missing, we
        # fall back to the league-average estimates computed above (no crash).
        self.venue_stats: Dict[Tuple, dict] = {}
        try:
            cur.execute("""
                SELECT team_id, league_id, season_id, venue,
                       games, wins, draws, losses, goals_for, goals_against
                FROM team_venue_stats
                WHERE games > 0
            """)
            venue_by_league: Dict[Tuple, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
            for r in cur.fetchall():
                r = dict(r)
                self.venue_stats[(r["team_id"], r["season_id"], r["venue"])] = r
                venue_by_league[(r["league_id"], r["season_id"])][r["venue"]].append(r)

            # Overwrite league venue averages with real data
            for (lid, sid), by_venue in venue_by_league.items():
                if (lid, sid) not in self.league_avgs:
                    continue
                for venue, rows in by_venue.items():
                    gf_pgs = [_safe_div(_f(r.get("goals_for")), max(_f(r.get("games")), 1)) for r in rows]
                    ga_pgs = [_safe_div(_f(r.get("goals_against")), max(_f(r.get("games")), 1)) for r in rows]
                    n = len(rows) or 1
                    if venue == "home":
                        self.league_avgs[(lid, sid)]["avg_home_gf_pg"] = sum(gf_pgs) / n
                        self.league_avgs[(lid, sid)]["avg_home_ga_pg"] = sum(ga_pgs) / n
                    elif venue == "away":
                        self.league_avgs[(lid, sid)]["avg_away_gf_pg"] = sum(gf_pgs) / n
                        self.league_avgs[(lid, sid)]["avg_away_ga_pg"] = sum(ga_pgs) / n
        except Exception as _venue_err:
            log.warning("team_venue_stats unavailable (%s) — using league-average fallbacks.", _venue_err)
            self.venue_stats = {}

        # 8. Match Odds (Enrichment Data) ─────────────────────────────────────
        try:
            cur.execute("""
                SELECT match_id, b365_home_win, b365_draw, b365_away_win, raw_data
                FROM match_odds
            """)
            self.match_odds: Dict[int, dict] = {r["match_id"]: dict(r) for r in cur.fetchall()}
        except Exception as _odds_err:
            log.warning("match_odds unavailable (%s)", _odds_err)
            self.match_odds = {}

        # 9. Team ClubElo (Enrichment Data) ───────────────────────────────────
        try:
            cur.execute("""
                SELECT team_id, elo_date, elo, raw_data
                FROM team_clubelo
                ORDER BY team_id, elo_date DESC
            """)
            self.team_clubelo: Dict[int, List[dict]] = defaultdict(list)
            for r in cur.fetchall():
                self.team_clubelo[r["team_id"]].append(dict(r))
        except Exception as _elo_err:
            log.warning("team_clubelo unavailable (%s)", _elo_err)
            self.team_clubelo = defaultdict(list)

        log.info(
            "DataCache ready: %d standings, %d squad rows, %d player rows, %d matches, %d venue rows, %d odds.",
            len(self.standings), len(self.squad), sum(len(v) for v in self.players.values()),
            len(self.all_matches), len(self.venue_stats), len(self.match_odds)
        )

        # 7. Compute league-style statistics in-memory from already-loaded matches.
        # This reuses self.all_matches — no additional DB query needed.
        # Keyed by league_id → dict of 6 style features.
        self.league_style: Dict[int, dict] = {}
        _by_league: Dict[int, dict] = defaultdict(lambda: {
            "n": 0, "hw": 0, "d": 0, "aw": 0, "total_goals": 0.0, "btts": 0
        })
        for m in self.all_matches:
            lid = m.get("league_id")
            if lid is None:
                continue
            hs = _f(m.get("home_score")); as_ = _f(m.get("away_score"))
            bucket = _by_league[lid]
            bucket["n"] += 1
            bucket["total_goals"] += hs + as_
            if hs > as_:   bucket["hw"] += 1
            elif hs == as_: bucket["d"]  += 1
            else:           bucket["aw"] += 1
            if hs > 0 and as_ > 0: bucket["btts"] += 1

        for lid, b in _by_league.items():
            n = b["n"]
            if n < 10:
                self.league_style[lid] = dict(_LEAGUE_STYLE_DEFAULTS)
                continue
            hw_r = b["hw"] / n
            aw_r = b["aw"] / n
            self.league_style[lid] = {
                "league_home_win_rate":        round(hw_r,                   4),
                "league_draw_rate":            round(b["d"]  / n,            4),
                "league_away_win_rate":        round(aw_r,                   4),
                "league_goals_pg":             round(b["total_goals"] / n,   4),
                "league_btts_rate":            round(b["btts"] / n,          4),
                "league_home_advantage_score": round(
                    _safe_div(hw_r, aw_r, _LEAGUE_STYLE_DEFAULTS["league_home_advantage_score"]), 4
                ),
            }


# ─── Form (in-memory) ─────────────────────────────────────────────────────────

def _compute_form(cache: DataCache, team_id: int, venue: Optional[str] = None,
                  n: int = 5, before_date=None, league_id: int = None) -> dict:
    """
    Compute recent form for a team.

    When league_id is provided and the team has played 2+ matches in that
    competition, returns competition-specific form (e.g. Europa League form
    for a PL team playing in Europe). Falls back to all-competition form
    when insufficient competition-specific data exists.

    This prevents domestic form (e.g. Aston Villa struggling in PL) from
    contaminating European form where the same team may be thriving.
    """
    matches = cache.matches_by_team.get(team_id, [])
    matches = _filter_before(matches, before_date)
    if venue == "home":
        matches = [m for m in matches if m["home_team_id"] == team_id]
    elif venue == "away":
        matches = [m for m in matches if m["away_team_id"] == team_id]

    # Competition-specific form: if team has 2+ games in this competition,
    # use those instead of mixed all-competitions form.
    if league_id is not None:
        comp_matches = [m for m in matches if m.get("league_id") == league_id]
        if len(comp_matches) >= 2:
            matches = comp_matches

    matches = matches[:n]

    if not matches:
        return {"form_score": 0.5, "goals_scored_avg": 1.35,
                "goals_conceded_avg": 1.35, "win_streak": 0,
                "results": [], "last3_form_score": 0.5,
                "weighted_form_score": 0.5}

    pts_total = gf_total = ga_total = 0
    results, streak, streak_active = [], 0, True
    pts_last3 = 0
    weighted_pts = 0.0
    weighted_max = 0.0
    _decay = 0.8  # exponential decay — most recent game = 1.0, each prior game -20%
    for i, r in enumerate(matches):
        is_home = r["home_team_id"] == team_id
        gf = _f(r["home_score"] if is_home else r["away_score"])
        ga = _f(r["away_score"] if is_home else r["home_score"])
        gf_total += gf; ga_total += ga
        w = _decay ** i
        weighted_max += 3.0 * w
        if gf > ga:
            pts_total += 3; results.append("W")
            if streak_active: streak += 1
            if i < 3: pts_last3 += 3
            weighted_pts += 3.0 * w
        elif gf == ga:
            pts_total += 1; results.append("D"); streak_active = False
            if i < 3: pts_last3 += 1
            weighted_pts += 1.0 * w
        else:
            results.append("L"); streak_active = False

    n_actual  = len(matches)
    n_last3   = min(n_actual, 3)
    return {
        "form_score":          _safe_div(pts_total,    n_actual * 3,  0.5),
        "last3_form_score":    _safe_div(pts_last3,    n_last3  * 3,  0.5),
        "weighted_form_score": _safe_div(weighted_pts, weighted_max,  0.5),
        "goals_scored_avg":    _safe_div(gf_total,     n_actual,      1.2),
        "goals_conceded_avg":  _safe_div(ga_total,     n_actual,      1.2),
        "win_streak":          streak,
        "results":             results,
    }


# ─── Multi-window form stats (FootballGPT rolling aggregates) ─────────────────

def _compute_form_windows(
    cache: "DataCache",
    team_id: int,
    venue: Optional[str] = None,
    before_date=None,
    league_id: int = None,
) -> dict:
    """
    Compute rolling-window form statistics at three window sizes (3 / 5 / 7).

    For each window and each core metric (points, goals_scored, goals_conceded)
    this returns five statistics borrowed from the FootballGPT feature set:
      - mean   : average performance over the window
      - cv     : coefficient of variation (std / mean) — consistency / volatility
      - trend  : linear regression slope (positive = improving, negative = fading)
      - min    : floor performance in the window
      - max    : ceiling performance in the window

    Keys are prefixed form_w{N}_{metric}_{stat}, e.g.:
      form_w3_pts_mean, form_w7_gf_trend, form_w5_ga_cv …

    All values fall back to neutral defaults (0.0) when insufficient match data
    exists, so teams with < 3 completed fixtures still produce a full feature
    vector and no existing prediction breaks.

    This function is ADDITIVE — it never replaces _compute_form().  Both are
    called independently inside _build_team_features().
    """
    # ── Neutral fallback (returned when no data is available) ─────────────
    _NEUTRAL: dict = {}
    for _n in (3, 5, 7):
        for _metric in ("pts", "gf", "ga"):
            for _stat in ("mean", "cv", "trend", "min", "max"):
                _NEUTRAL[f"form_w{_n}_{_metric}_{_stat}"] = 0.0

    # ── Pull and filter matches (mirrors _compute_form logic exactly) ─────
    matches = cache.matches_by_team.get(team_id, [])
    matches = _filter_before(matches, before_date)

    if venue == "home":
        matches = [m for m in matches if m["home_team_id"] == team_id]
    elif venue == "away":
        matches = [m for m in matches if m["away_team_id"] == team_id]

    # Competition-specific form: same rule as _compute_form
    if league_id is not None:
        comp_matches = [m for m in matches if m.get("league_id") == league_id]
        if len(comp_matches) >= 2:
            matches = comp_matches

    # We only ever need the most recent 7 matches
    matches = matches[:7]

    if not matches:
        return dict(_NEUTRAL)

    # ── Build per-match pts / gf / ga lists (most-recent first) ──────────
    pts_list: list = []
    gf_list:  list = []
    ga_list:  list = []

    for r in matches:
        is_home = r["home_team_id"] == team_id
        gf = _f(r["home_score"] if is_home else r["away_score"])
        ga = _f(r["away_score"] if is_home else r["home_score"])
        if   gf > ga:  pts = 3
        elif gf == ga: pts = 1
        else:          pts = 0
        pts_list.append(pts)
        gf_list.append(gf)
        ga_list.append(ga)

    # ── Inner helper: 5-stat summary for a numeric list ───────────────────
    def _window_stats(values: list) -> dict:
        """
        Returns mean / cv / trend / min / max for a list of floats.
        trend is the linear regression slope fitted in chronological order
        (oldest first), so a positive slope means the metric is rising.
        """
        n = len(values)
        if n == 0:
            return {"mean": 0.0, "cv": 0.0, "trend": 0.0, "min": 0.0, "max": 0.0}

        arr = np.array(values, dtype=float)
        mean_val = float(np.mean(arr))
        std_val  = float(np.std(arr))

        # Coefficient of variation — small epsilon avoids divide-by-zero
        # when mean ≈ 0 (e.g. team on a losing streak with 0 pts per game).
        cv_val = std_val / (mean_val + 1e-6)

        # Trend: reverse list so index 0 = oldest, then fit slope.
        if n >= 2:
            chron = arr[::-1]
            slope, *_ = linregress(np.arange(n, dtype=float), chron)
            trend_val = float(slope)
        else:
            trend_val = 0.0

        return {
            "mean":  round(mean_val,              4),
            "cv":    round(cv_val,                4),
            "trend": round(trend_val,             4),
            "min":   round(float(np.min(arr)),    4),
            "max":   round(float(np.max(arr)),    4),
        }

    # ── Compute stats for each window × metric combination ────────────────
    feats: dict = {}
    for n in (3, 5, 7):
        slice_pts = pts_list[:n]
        slice_gf  = gf_list[:n]
        slice_ga  = ga_list[:n]

        # Compute regardless of whether len(slice) == n: if fewer matches
        # exist than the window size, _window_stats handles short arrays fine.
        for metric, data in (("pts", slice_pts), ("gf", slice_gf), ("ga", slice_ga)):
            stats = _window_stats(data)
            for stat_name, stat_val in stats.items():
                feats[f"form_w{n}_{metric}_{stat_name}"] = stat_val

    return feats


# ─── H2H (in-memory) ──────────────────────────────────────────────────────────

def _compute_h2h(cache: DataCache, home_team_id: int, away_team_id: int, n: int = 10, before_date=None) -> dict:
    rows = [
        m for m in cache.all_matches
        if (m["home_team_id"] == home_team_id and m["away_team_id"] == away_team_id)
        or (m["home_team_id"] == away_team_id and m["away_team_id"] == home_team_id)
    ]
    rows = _filter_before(rows, before_date)[:n]

    h_wins = d = a_wins = hg_tot = ag_tot = 0
    last_5 = []
    for r in rows:
        if r["home_team_id"] == home_team_id:
            hg, ag = _f(r["home_score"]), _f(r["away_score"])
        else:
            hg, ag = _f(r["away_score"]), _f(r["home_score"])
        hg_tot += hg; ag_tot += ag
        if hg > ag:   h_wins += 1; last_5.append("H")
        elif hg == ag: d += 1;     last_5.append("D")
        else:          a_wins += 1; last_5.append("A")

    total = h_wins + d + a_wins or 1
    return {
        "h2h_home_wins":    h_wins,
        "h2h_draws":        d,
        "h2h_away_wins":    a_wins,
        "h2h_home_win_pct": _safe_div(h_wins, total),
        "h2h_away_win_pct": _safe_div(a_wins, total),
        "h2h_home_gf_avg":  _safe_div(hg_tot, total),
        "h2h_away_gf_avg":  _safe_div(ag_tot, total),
        "h2h_last_5":       last_5[:5],
    }


# ─── Player features (in-memory) ─────────────────────────────────────────────

def _pos_bucket(pos: str) -> str:
    """Map FBref position string to GK / DF / MF / FW bucket."""
    p = str(pos or "").upper().strip()
    if not p:             return "UNK"
    if p.startswith("GK"): return "GK"
    # Primary position is first token before comma
    primary = p.split(",")[0].strip()
    if primary == "DF":   return "DF"
    if primary == "MF":   return "MF"
    if primary == "FW":   return "FW"
    # Fallback: scan for known tokens
    if "GK" in p:         return "GK"
    if "FW" in p:         return "FW"
    if "MF" in p:         return "MF"
    if "DF" in p:         return "DF"
    return "UNK"


def _build_player_features(cache: DataCache, team_id: int, season_id: int) -> dict:
    """
    Build positional player features from top-25-by-minutes squad.

    Replaces the old top-5-scorers approach with four positional buckets:
      ATTACK (FW):   goals, xG, attack depth, dependency
      MIDFIELD (MF): assists, progressive passes, chance creation
      DEFENCE (DF):  defensive solidity, squad continuity
      GK:            goalkeeper consistency
      SQUAD (all):   age profile, depth, minutes concentration
    All original feature names preserved for backward compatibility.
    New features added alongside — no deletions.
    """
    players = cache.players.get((team_id, season_id), [])

    # ── zero dict: all features (old + new) default to 0.0 ──────────────────
    zero = {k: 0.0 for k in [
        # ── original features (kept for backward compatibility) ───────────
        "top_scorer_goals", "goal_concentration", "n_scorers",
        "squad_depth_scorers", "top_assister_assists",
        "avg_goals_per_player", "scorer_avg_age",
        "top_scorer_goals_per90", "team_player_assists_pg",
        "attack_dependency",
        # ── new positional features ───────────────────────────────────────
        # Attack
        "avg_attack_goals_per90", "avg_attack_xg_per90",
        "attack_depth",
        # Midfield
        "avg_mid_assists_per90", "avg_mid_progressive_passes",
        "mid_depth",
        # Defence / GK
        "avg_def_minutes", "gk_games", "def_depth",
        # Squad-wide
        "squad_avg_age", "squad_depth",
        "minutes_concentration",
        # GK individual
        "gk_save_pct_player", "gk_clean_sheet_rate_player",
        "gk_goals_prevented", "gk_psxg_per90", "gk_mins",
    ]}
    if not players:
        return zero

    # ── bucket players by position ──────────────────────────────────────────
    buckets = {"GK": [], "DF": [], "MF": [], "FW": [], "UNK": []}
    for p in players:
        buckets[_pos_bucket(p.get("position", ""))].append(p)

    # ── helper: safe standard_stats JSONB access ─────────────────────────────
    def _ss(p, key, default=0.0):
        ss = p.get("standard_stats")
        if isinstance(ss, dict):
            return _f(ss.get(key), default)
        return default

    # ── squad-wide metrics (all 25 players) ──────────────────────────────────
    total_goals   = sum(_f(p.get("goals"))   for p in players)
    total_assists = sum(_f(p.get("assists"))  for p in players)
    total_minutes = sum(_f(p.get("minutes"))  for p in players) or 1
    scorers       = [p for p in players if _f(p.get("goals")) > 0]

    n_games = sum(
        1 for m in cache.matches_by_team.get(team_id, [])
        if m["season_id"] == season_id
    ) or 1

    # Minutes concentration: do top 3 players carry the whole team?
    top3_mins = sum(_f(p.get("minutes")) for p in players[:3])
    minutes_concentration = _safe_div(top3_mins, total_minutes, 0.0)

    # Squad depth: players who played 500+ minutes (meaningful contributors)
    squad_depth = sum(1 for p in players if _f(p.get("minutes")) >= 500)

    # Squad avg age (weighted by minutes — starters count more)
    ages_w = [(_f(p.get("age"), 26.0), _f(p.get("minutes"), 0.0)) for p in players]
    squad_avg_age = (_safe_div(sum(a*m for a,m in ages_w), total_minutes, 26.0))

    # ── original features (backward-compatible) ───────────────────────────────
    # Use all players sorted by minutes (not just attackers)
    top = players[0]  # highest-minutes player
    top_goals = _f(top.get("goals"))
    top_m90   = _f(top.get("minutes_90s")) or 1
    conc = _safe_div(top_goals, total_goals, 0.0)

    # Best scoring player (by goals, for top_scorer_goals backward compat)
    top_scorer = max(players, key=lambda p: _f(p.get("goals")))
    top_scorer_goals = _f(top_scorer.get("goals"))
    top_scorer_m90   = _f(top_scorer.get("minutes_90s")) or 1
    top_scorer_ss    = top_scorer.get("standard_stats") if isinstance(
                          top_scorer.get("standard_stats"), dict) else {}
    goal_conc = _safe_div(top_scorer_goals, total_goals, 0.0)

    # Best assisting player (by assists)
    top_assister = max(players, key=lambda p: _f(p.get("assists")))

    # ── ATTACK bucket features ────────────────────────────────────────────────
    attackers = buckets["FW"] or buckets["UNK"][:4]  # fallback if no FW tagged
    if attackers:
        atk_top = sorted(attackers, key=lambda p: _f(p.get("minutes")), reverse=True)[:4]
        avg_attack_goals_per90 = _safe_div(
            sum(_ss(p, "goals_per90") for p in atk_top), len(atk_top))
        avg_attack_xg_per90 = _safe_div(
            sum(_ss(p, "xg") / max(_f(p.get("minutes_90s")), 0.1) for p in atk_top),
            len(atk_top))
        attack_depth = sum(1 for p in attackers if _f(p.get("minutes")) >= 500)
    else:
        avg_attack_goals_per90 = _safe_div(total_goals, n_games * 11)
        avg_attack_xg_per90    = 0.0
        attack_depth           = 0.0

    # ── MIDFIELD bucket features ──────────────────────────────────────────────
    mids = buckets["MF"]
    if mids:
        mid_top = sorted(mids, key=lambda p: _f(p.get("minutes")), reverse=True)[:5]
        avg_mid_assists_per90 = _safe_div(
            sum(_ss(p, "assists_per90") for p in mid_top), len(mid_top))
        avg_mid_progressive_passes = _safe_div(
            sum(_ss(p, "progressive_passes") for p in mid_top), len(mid_top))
        mid_depth = sum(1 for p in mids if _f(p.get("minutes")) >= 500)
    else:
        avg_mid_assists_per90       = _safe_div(total_assists, n_games * 11)
        avg_mid_progressive_passes  = 0.0
        mid_depth                   = 0.0

    # ── DEFENCE / GK bucket features ─────────────────────────────────────────
    defenders = buckets["DF"]
    gks       = buckets["GK"]
    if defenders:
        def_top = sorted(defenders, key=lambda p: _f(p.get("minutes")), reverse=True)[:5]
        avg_def_minutes = _safe_div(
            sum(_f(p.get("minutes")) for p in def_top), len(def_top))
        def_depth = sum(1 for p in defenders if _f(p.get("minutes")) >= 500)
    else:
        avg_def_minutes = 0.0
        def_depth       = 0.0

    gk_games = _f(gks[0].get("games")) if gks else 0.0

    # ── Individual GK quality from player stats ─────────────────────────────
    # These complement the team-level gk_save_pct from squad_stats.
    # squad_stats gives aggregate; player stats gives primary keeper's individual stats.
    if gks:
        gk1 = sorted(gks, key=lambda p: _f(p.get("minutes")), reverse=True)[0]
        gk1_ss = gk1.get("standard_stats") if isinstance(
                     gk1.get("standard_stats"), dict) else {}
        gk_save_pct_player   = _f(gk1_ss.get("gk_save_pct"),    0.0)
        gk_clean_sheet_rate  = _f(gk1_ss.get("gk_clean_sheet_pct"), 0.0)
        gk_goals_prevented   = _f(gk1_ss.get("gk_goals_prevented"), 0.0)  # PSxG-GA
        gk_psxg_per90        = _f(gk1_ss.get("gk_psxg_per90"),    0.0)
        gk_mins              = _f(gk1.get("minutes"),              0.0)
    else:
        gk_save_pct_player = gk_clean_sheet_rate = 0.0
        gk_goals_prevented = gk_psxg_per90 = gk_mins = 0.0

    # ── assemble result ───────────────────────────────────────────────────────
    return {
        # ── original features (unchanged names) ───────────────────────────
        "top_scorer_goals":       top_scorer_goals,
        "goal_concentration":     goal_conc,
        "n_scorers":              _f(len(scorers)),
        "squad_depth_scorers":    _safe_div(len(scorers), len(players), 0.5),
        "top_assister_assists":   _f(top_assister.get("assists")),
        "avg_goals_per_player":   _safe_div(total_goals, len(players)),
        "scorer_avg_age":         (_safe_div(sum(_f(p.get("age")) for p in scorers),
                                             len(scorers), 26.0) if scorers else 26.0),
        "top_scorer_goals_per90": _j(top_scorer_ss, "goals_per90",
                                     _safe_div(top_scorer_goals, top_scorer_m90)),
        "team_player_assists_pg": _safe_div(total_assists, n_games),
        "attack_dependency":      1.0 if goal_conc > 0.40 else 0.0,
        # ── new positional features ───────────────────────────────────────
        "avg_attack_goals_per90":      avg_attack_goals_per90,
        "avg_attack_xg_per90":         avg_attack_xg_per90,
        "attack_depth":                float(attack_depth),
        "avg_mid_assists_per90":       avg_mid_assists_per90,
        "avg_mid_progressive_passes":  avg_mid_progressive_passes,
        "mid_depth":                   float(mid_depth),
        "avg_def_minutes":             avg_def_minutes,
        "gk_games":                    gk_games,
        "def_depth":                   float(def_depth),
        "squad_avg_age":               squad_avg_age,
        "squad_depth":                 float(squad_depth),
        "minutes_concentration":       minutes_concentration,
        # ── individual GK quality (primary keeper) ───────────────────────
        "gk_save_pct_player":          gk_save_pct_player,
        "gk_clean_sheet_rate_player":  gk_clean_sheet_rate,
        "gk_goals_prevented":          gk_goals_prevented,
        "gk_psxg_per90":               gk_psxg_per90,
        "gk_mins":                     gk_mins,
    }


# ─── Previous season (in-memory) ──────────────────────────────────────────────

_PREV_ZERO = {
    "prev_rank_norm": 0.5, "prev_points_avg": 0.0, "prev_wins_pct": 0.0,
    "prev_goals_for_pg": 1.35, "prev_goals_ag_pg": 1.35,
    "prev_goal_diff_pg": 0.0, "prev_season_found": 0.0,
}


def _build_prev_season_features(cache: DataCache, team_id: int, league_id: int) -> dict:
    # All seasons for this team+league sorted by season name descending
    candidates = sorted(
        [(sid, cache.season_name.get(sid, ""))
         for (tid, lid, sid) in cache.standings
         if tid == team_id and lid == league_id],
        key=lambda x: x[1], reverse=True,
    )
    if len(candidates) < 2:
        return _PREV_ZERO

    prev_sid = candidates[1][0]
    prev = cache.standings.get((team_id, league_id, prev_sid))
    if not prev:
        return _PREV_ZERO

    # Number of teams in that league/season (for rank normalisation)
    n = cache.league_avgs.get((league_id, prev_sid), {}).get("n_teams", 20)
    g = _f(prev.get("games")) or 1
    return {
        "prev_rank_norm":    1.0 - _safe_div(_f(prev.get("rank")) - 1, n - 1),
        "prev_points_avg":   _f(prev.get("points_avg")),
        "prev_wins_pct":     _safe_div(_f(prev.get("wins")),          g),
        "prev_goals_for_pg": _safe_div(_f(prev.get("goals_for")),     g),
        "prev_goals_ag_pg":  _safe_div(_f(prev.get("goals_against")), g),
        "prev_goal_diff_pg": _safe_div(_f(prev.get("goal_diff")),     g),
        "prev_season_found": 1.0,
    }


def _build_prev_season_form(cache: DataCache, team_id: int, before_date=None) -> dict:
    zero = {"prev_form_score": 0.5, "prev_gf_pg": 1.35,
            "prev_ga_pg": 1.35, "prev_match_wins_pct": 0.0}
    
    matches = cache.matches_by_team.get(team_id, [])
    matches = _filter_before(matches, before_date)

    # All distinct seasons for this team from match history (desc by name)
    seasons_used = sorted(
        set((m["season_id"], cache.season_name.get(m["season_id"], ""))
            for m in matches),
        key=lambda x: x[1], reverse=True,
    )
    if len(seasons_used) < 2:
        return zero

    prev_sid = seasons_used[1][0]
    prev_matches = [m for m in matches if m["season_id"] == prev_sid]
    if not prev_matches:
        return zero

    wins = pts = gf_t = ga_t = 0
    for m in prev_matches:
        is_home = m["home_team_id"] == team_id
        gf = _f(m["home_score"] if is_home else m["away_score"])
        ga = _f(m["away_score"] if is_home else m["home_score"])
        gf_t += gf; ga_t += ga
        if gf > ga:    wins += 1; pts += 3
        elif gf == ga: pts += 1

    n = len(prev_matches)
    return {
        "prev_form_score":      _safe_div(pts, n * 3, 0.5),
        "prev_gf_pg":           _safe_div(gf_t, n),
        "prev_ga_pg":           _safe_div(ga_t, n),
        "prev_match_wins_pct":  _safe_div(wins, n),
    }


# ─── Scoring patterns (in-memory) ─────────────────────────────────────────────

# ─── Scoring patterns (in-memory) ─────────────────────────────────────────────

def _build_scoring_patterns(cache: DataCache, team_id: int, season_id: int, before_date=None) -> dict:
    zero = {"goals_scored_variance": 0.0, "goals_conceded_variance": 0.0,
            "blank_rate": 0.0, "blowout_rate": 0.0,
            "defensive_collapse_rate": 0.0, "clean_sheet_rate": 0.0}
    matches = [m for m in cache.matches_by_team.get(team_id, [])
               if m["season_id"] == season_id]
    matches = _filter_before(matches, before_date)
    if not matches:
        return zero

    scored, conceded = [], []
    blanks = blowouts = collapses = clean_sheets = 0
    for m in matches:
        is_home = m["home_team_id"] == team_id
        gf = _f(m["home_score"] if is_home else m["away_score"])
        ga = _f(m["away_score"] if is_home else m["home_score"])
        scored.append(gf); conceded.append(ga)
        if gf == 0: blanks += 1
        if gf >= 3: blowouts += 1
        if ga == 0: clean_sheets += 1
        if ga >= 3: collapses += 1

    n = len(matches)
    mean_gf = sum(scored)   / n
    mean_ga = sum(conceded) / n
    var_gf  = sum((x - mean_gf) ** 2 for x in scored)   / n if n > 1 else 0.0
    var_ga  = sum((x - mean_ga) ** 2 for x in conceded) / n if n > 1 else 0.0
    return {
        "goals_scored_variance":   _f(var_gf),
        "goals_conceded_variance": _f(var_ga),
        "blank_rate":              _safe_div(blanks,      n),
        "blowout_rate":            _safe_div(blowouts,    n),
        "defensive_collapse_rate": _safe_div(collapses,   n),
        "clean_sheet_rate":        _safe_div(clean_sheets, n),
    }


# ─── Venue stats (in-memory) ──────────────────────────────────────────────────

def _build_venue_stats(cache: DataCache, team_id: int, season_id: int) -> dict:
    """
    Return home/away venue stats from the pre-loaded team_venue_stats cache.
    Keys: home_gf_pg, home_ga_pg, home_win_rate, home_games,
          away_gf_pg, away_ga_pg, away_win_rate, away_games
    """
    result = {}
    for venue, fallback_gf, fallback_ga in [
        ("home", 1.55, 1.15),
        ("away", 1.15, 1.55),
    ]:
        r = cache.venue_stats.get((team_id, season_id, venue))
        if r:
            g = _f(r.get("games")) or 1
            result[f"{venue}_gf_pg"]    = _safe_div(_f(r.get("goals_for")),    g, fallback_gf)
            result[f"{venue}_ga_pg"]    = _safe_div(_f(r.get("goals_against")), g, fallback_ga)
            result[f"{venue}_win_rate"] = _safe_div(_f(r.get("wins")),          g, 0.33)
            result[f"{venue}_games"]    = _f(r.get("games"))
        else:
            result[f"{venue}_gf_pg"]    = fallback_gf
            result[f"{venue}_ga_pg"]    = fallback_ga
            result[f"{venue}_win_rate"] = 0.33
            result[f"{venue}_games"]    = 0
    return result


# ─── Enrichment features (in-memory) ──────────────────────────────────────────

def _build_enrichment_features(cache: DataCache, match_id: Optional[int], match_date, home_team_id: int, away_team_id: int, league_id: int) -> dict:
    feats = {}
    
    # Missing Odds fallback values based on league baseline stats
    style = cache.league_style.get(league_id, dict(_LEAGUE_STYLE_DEFAULTS))
    btts_rate = style.get("league_btts_rate", 0.50)
    hw_rate   = style.get("league_home_win_rate", 0.44)
    aw_rate   = style.get("league_away_win_rate", 0.31)
    
    hw_sum = hw_rate + aw_rate + style.get("league_draw_rate", 0.25)
    
    # ── 1. Match Odds ──
    odds = cache.match_odds.get(match_id) if match_id else None
    if odds and odds.get("b365_home_win"):
        hw = _f(odds["b365_home_win"])
        dw = _f(odds["b365_draw"])
        aw = _f(odds["b365_away_win"])
        
        margin_1x2 = (1.0/hw + 1.0/dw + 1.0/aw) if (hw and dw and aw) else 1.0
        feats["odds_home_prob"] = (1.0/hw) / margin_1x2 if hw else hw_rate/hw_sum
        feats["odds_draw_prob"] = (1.0/dw) / margin_1x2 if dw else style.get("league_draw_rate",0.25)/hw_sum
        feats["odds_away_prob"] = (1.0/aw) / margin_1x2 if aw else aw_rate/hw_sum
        
        raw_odds = odds.get("raw_data") or {}
        o25 = _f(raw_odds.get("B365>2.5", raw_odds.get("Max>2.5", 0)))
        u25 = _f(raw_odds.get("B365<2.5", raw_odds.get("Max<2.5", 0)))
        margin_ou = (1.0/o25 + 1.0/u25) if (o25 > 0 and u25 > 0) else 1.0
        feats["odds_over_25_prob"]  = (1.0/o25) / margin_ou if o25 > 0 else btts_rate
        feats["odds_under_25_prob"] = (1.0/u25) / margin_ou if u25 > 0 else (1 - btts_rate)
        
        ah_size = _f(raw_odds.get("AHh", 0.0))
        ahh     = _f(raw_odds.get("B365AHH", 0))
        aha     = _f(raw_odds.get("B365AHA", 0))
        margin_ah = (1.0/ahh + 1.0/aha) if (ahh > 0 and aha > 0) else 1.0
        feats["odds_asian_hdcp_size"] = ah_size
        feats["odds_ah_home_prob"]    = (1.0/ahh) / margin_ah if ahh > 0 else 0.50
        feats["odds_ah_away_prob"]    = (1.0/aha) / margin_ah if aha > 0 else 0.50
    else:
        feats["odds_home_prob"]       = hw_rate/hw_sum
        feats["odds_draw_prob"]       = style.get("league_draw_rate",0.25)/hw_sum
        feats["odds_away_prob"]       = aw_rate/hw_sum
        feats["odds_over_25_prob"]    = btts_rate
        feats["odds_under_25_prob"]   = 1 - btts_rate
        feats["odds_asian_hdcp_size"] = 0.0
        feats["odds_ah_home_prob"]    = 0.50
        feats["odds_ah_away_prob"]    = 0.50

    # ── 2. Team ClubElo ──
    from datetime import date, datetime
    m_date = None
    if isinstance(match_date, str):
        try: m_date = datetime.fromisoformat(str(match_date)[:10]).date()
        except: pass
    elif hasattr(match_date, "date"): 
        m_date = match_date.date()
    elif isinstance(match_date, date):
        m_date = match_date
    
    def _get_elo(tid):
        history = cache.team_clubelo.get(tid, [])
        if not history: return 1500.0, 0.0, 0.0, 0.0
        # If no match_date provided (predicting upcoming), use latest
        if m_date is None: 
            row = history[0]
        else:
            # find closest snapshot where elo_date <= m_date
            row = next((r for r in history if r["elo_date"] and r["elo_date"] <= m_date), history[-1])
        e = _f(row["elo"]) if row["elo"] else 1500.0
        
        raw_json = row.get("raw_data") or {}
        raw_inner = raw_json.get("raw", {})
        h_prob = d_prob = a_prob = 0.0
        if raw_inner and "GD=0" in raw_inner:
            d_prob = _f(raw_inner.get("GD=0", 0.0))
            hw_sum_el = sum(_f(raw_inner.get(f"GD={i}", 0.0)) for i in range(1, 10))
            h_prob = hw_sum_el
            aw_sum_el = sum(_f(raw_inner.get(f"GD={i}", 0.0)) for i in range(-9, 0))
            a_prob = aw_sum_el
            
            # Normalize just like standalone enrichment script
            tot = h_prob + d_prob + a_prob
            if tot > 0:
                h_prob, d_prob, a_prob = h_prob/tot, d_prob/tot, a_prob/tot

        return e, h_prob, d_prob, a_prob

    h_elo, he_hw, he_dr, _ = _get_elo(home_team_id)
    a_elo, _, _, ae_aw     = _get_elo(away_team_id)
    
    feats["clubelo_home_rating"] = h_elo
    feats["clubelo_away_rating"] = a_elo
    feats["clubelo_gap"]         = h_elo - a_elo
    feats["clubelo_home_prob"]   = he_hw if he_hw > 0 else (hw_rate/hw_sum)
    feats["clubelo_draw_prob"]   = he_dr if he_dr > 0 else (style.get("league_draw_rate",0.25)/hw_sum)
    feats["clubelo_away_prob"]   = ae_aw if ae_aw > 0 else (aw_rate/hw_sum)
    
    return feats


# ─── Injury features ─────────────────────────────────────────────────────────

def _build_injury_features(cache: DataCache, team_id: int,
                            player_rows: List[dict]) -> dict:
    """
    Build injury-impact features from player_injuries table.

    Cross-references active injuries against the team's top-25 player list
    to estimate squad availability and key-player absence risk.

    Features:
      n_injuries          — raw count of active injured players
      injury_severity     — weighted count (muscle/ligament = 1.5, minor = 0.5)
      squad_availability  — % of usual starters expected to be available
      key_player_injured  — 1.0 if top-3-by-minutes player is injured
      gk_injured          — 1.0 if the first-choice GK is injured
      attack_injured      — count of injured FW/attackers
      mid_injured         — count of injured MF/midfielders
    """
    zero = {
        "n_injuries":         0.0,
        "injury_severity":    0.0,
        "squad_availability": 1.0,
        "key_player_injured": 0.0,
        "gk_injured":         0.0,
        "attack_injured":     0.0,
        "mid_injured":        0.0,
    }
    injuries = cache.injuries.get(team_id, [])
    if not injuries:
        return zero

    # Severity weights by injury type keyword
    def _severity(inj_type: str) -> float:
        t = str(inj_type or "").lower()
        if any(k in t for k in ("acl", "ligament", "fracture", "surgery", "torn")):
            return 2.0   # long-term
        if any(k in t for k in ("muscle", "hamstring", "thigh", "calf", "groin")):
            return 1.5   # medium-term
        if any(k in t for k in ("knock", "minor", "illness", "doubt", "fitness")):
            return 0.5   # short-term
        return 1.0       # unknown/other

    injured_names = {str(i["player_name"]).lower().strip() for i in injuries}
    n_inj = len(injuries)
    severity = sum(_severity(i.get("injury_type")) for i in injuries)

    # Cross-reference with squad (top 25 by minutes)
    key_injured   = 0.0
    gk_injured    = 0.0
    atk_injured   = 0.0
    mid_injured   = 0.0
    injured_mins  = 0.0
    total_mins    = sum(_f(p.get("minutes")) for p in player_rows) or 1

    for i, p in enumerate(player_rows):
        pname = str(p.get("player_name") or "").lower().strip()
        if pname not in injured_names:
            continue
        pos = _pos_bucket(p.get("position", ""))
        mins = _f(p.get("minutes"))
        injured_mins += mins
        if i < 3:              key_injured   = 1.0   # top-3 by minutes = key player
        if pos == "GK":        gk_injured    = 1.0
        if pos == "FW":        atk_injured  += 1.0
        if pos == "MF":        mid_injured  += 1.0

    squad_availability = max(0.0, 1.0 - _safe_div(injured_mins, total_mins))

    return {
        "n_injuries":         float(n_inj),
        "injury_severity":    severity,
        "squad_availability": squad_availability,
        "key_player_injured": key_injured,
        "gk_injured":         gk_injured,
        "attack_injured":     atk_injured,
        "mid_injured":        mid_injured,
    }


# ─── Full team feature builder (in-memory) ────────────────────────────────────

def _build_team_features(cache: DataCache, team_id: int,
                          league_id: int, season_id: int, before_date=None) -> dict:
    feats: dict = {}
    avgs = cache.league_avgs.get((league_id, season_id))
    if not avgs:
        # Cross-league: fall back to global average across all leagues this season
        all_avgs = [v for (lid, sid), v in cache.league_avgs.items() if sid == season_id]
        if all_avgs:
            avgs = {
                "avg_gf_pg": sum(a["avg_gf_pg"] for a in all_avgs) / len(all_avgs),
                "avg_ga_pg": sum(a["avg_ga_pg"] for a in all_avgs) / len(all_avgs),
                "n_teams":   max(a["n_teams"] for a in all_avgs),
            }
        else:
            avgs = {"avg_gf_pg": 1.3, "avg_ga_pg": 1.3, "n_teams": 20}
    st  = cache.standings.get((team_id, league_id, season_id))
    if not st:
        # Cross-league fallback: use domestic league standings for this season.
        # e.g. Aston Villa in Europa League → use their Premier League standings.
        st = cache.standings_fallback.get((team_id, season_id))
    n_teams = avgs.get("n_teams", 20)
    avg_gf  = avgs.get("avg_gf_pg", 1.3)
    avg_ga  = avgs.get("avg_ga_pg", 1.3)

    # 1. League standings
    if st:
        g      = _f(st.get("games")) or 1
        gf_pg  = _safe_div(_f(st.get("goals_for")),    g)
        ga_pg  = _safe_div(_f(st.get("goals_against")), g)
        gd_pg  = _safe_div(_f(st.get("goal_diff")),     g)
        feats["rank_norm"]       = 1.0 - _safe_div(_f(st.get("rank")) - 1, n_teams - 1)
        feats["points_avg"]      = _f(st.get("points_avg"))
        feats["wins_pct"]        = _safe_div(_f(st.get("wins")),   g)
        feats["draws_pct"]       = _safe_div(_f(st.get("ties")),   g)
        feats["losses_pct"]      = _safe_div(_f(st.get("losses")), g)
        feats["goals_for_pg"]    = gf_pg
        feats["goals_against_pg"]= ga_pg
        feats["goal_diff_pg"]    = gd_pg
        feats["attack_strength"] = _safe_div(gf_pg, avg_gf, 1.0)
        feats["defence_strength"]= _safe_div(avg_ga, ga_pg, 1.0)
        ha = st.get("home_away_split")
        ha = ha if isinstance(ha, dict) else {}
        feats["home_wins_pct"]     = _j(ha, "home_win_pct", feats["wins_pct"])
        feats["home_goals_for_pg"] = _j(ha, "home_gf_pg",   gf_pg)
        feats["home_goals_ag_pg"]  = _j(ha, "home_ga_pg",   ga_pg)
        feats["home_points_avg"]   = _j(ha, "home_pts_avg",  feats["points_avg"])
        feats["away_wins_pct"]     = _j(ha, "away_win_pct",  feats["wins_pct"])
        feats["away_goals_for_pg"] = _j(ha, "away_gf_pg",    gf_pg)
        feats["away_goals_ag_pg"]  = _j(ha, "away_ga_pg",    ga_pg)
        feats["away_points_avg"]   = _j(ha, "away_pts_avg",  feats["points_avg"])
    else:
        for k in ["rank_norm","points_avg","wins_pct","draws_pct","losses_pct",
                  "goals_for_pg","goals_against_pg","goal_diff_pg",
                  "attack_strength","defence_strength","home_wins_pct",
                  "home_goals_for_pg","home_goals_ag_pg","home_points_avg",
                  "away_wins_pct","away_goals_for_pg","away_goals_ag_pg","away_points_avg"]:
            feats[k] = 0.5 if "pct" in k or k == "rank_norm" else 0.0

    # 2. Squad stats — FOR split
    sq_for = cache.squad.get((team_id, season_id, "for"))
    if not sq_for:
        # Cross-league fallback: use domestic season squad stats when the
        # competition season_id doesn't match the team's stored squad stats.
        # e.g. Aston Villa's squad stats are stored under PL season_id,
        # but when they play Europa League a different season_id is used.
        sq_for = cache.squad_fallback.get((team_id, "for"))
    if sq_for:
        feats["possession"]   = _f(sq_for.get("possession"), 50.0)
        feats["avg_age"]      = _f(sq_for.get("avg_age"), 26.0)
        feats["players_used"] = _f(sq_for.get("players_used"), 20)
        m90 = _f(sq_for.get("minutes_90s")) or 1
        ss  = sq_for.get("standard_stats") or {}
        # Calculate per90 metrics strictly from the verified table totals rather than trusting the JSON scrape
        feats["goals_per90"]              = _safe_div(sq_for.get("goals"), sq_for.get("games"))
        feats["assists_per90"]            = _safe_div(sq_for.get("assists"), sq_for.get("games"))
        feats["goals_assists_per90"]      = feats["goals_per90"] + feats["assists_per90"]
        feats["goals_pens_per90"]         = _j(ss, "goals_pens_per90")
        feats["goals_assists_pens_per90"] = _j(ss, "goals_assists_pens_per90")
        feats["cards_yellow_per90"]       = _safe_div(_j(ss, "cards_yellow"), m90)
        feats["cards_red_per90"]          = _safe_div(_j(ss, "cards_red"),    m90)
        feats["pen_conversion"]           = _safe_div(_j(ss, "pens_made"),
                                                      _j(ss, "pens_att") or 1, 0.75)
        sh = sq_for.get("shooting") or {}
        feats["shots_per90"]                = _j(sh, "shots_per90")
        feats["shots_on_target_per90"]      = _j(sh, "shots_on_target_per90")
        feats["shots_on_target_pct"]        = _j(sh, "shots_on_target_pct")
        feats["goals_per_shot"]             = _j(sh, "goals_per_shot")
        feats["goals_per_shot_on_target"]   = _j(sh, "goals_per_shot_on_target")
        pt = sq_for.get("playing_time") or {}
        feats["points_per_game_pt"]  = _j(pt, "points_per_game")
        feats["plus_minus_per90"]    = _j(pt, "plus_minus_per90")
        on_gf = _j(pt, "on_goals_for"); on_ga = _j(pt, "on_goals_against")
        feats["on_field_goal_ratio"] = _safe_div(on_gf, on_gf + on_ga, 0.5)
        feats["squad_completeness"]  = _safe_div(_j(pt, "games_complete"),
                                                  _f(sq_for.get("games"), 1))
        ms = sq_for.get("misc_stats") or {}
        feats["fouls_per90"]         = _safe_div(_j(ms, "fouls"),         m90)
        feats["fouled_per90"]        = _safe_div(_j(ms, "fouled"),        m90)
        feats["offsides_per90"]      = _safe_div(_j(ms, "offsides"),      m90)
        feats["crosses_per90"]       = _safe_div(_j(ms, "crosses"),       m90)
        feats["interceptions_per90"] = _safe_div(_j(ms, "interceptions"), m90)
        feats["tackles_won_per90"]   = _safe_div(_j(ms, "tackles_won"),   m90)
        pens_con = _j(ms, "pens_conceded") or 1
        feats["pen_area_ratio"]      = _safe_div(_j(ms, "pens_won"), pens_con, 1.0)
        feats["own_goals"]           = _j(ms, "own_goals")
        feats["discipline_score"]    = (feats["cards_yellow_per90"] +
                                        feats["cards_red_per90"] * 3)
    else:
        for k in ["possession","avg_age","players_used","goals_per90","assists_per90",
                  "goals_assists_per90","goals_pens_per90","goals_assists_pens_per90",
                  "cards_yellow_per90","cards_red_per90","pen_conversion","shots_per90",
                  "shots_on_target_per90","shots_on_target_pct","goals_per_shot",
                  "goals_per_shot_on_target","points_per_game_pt","plus_minus_per90",
                  "on_field_goal_ratio","squad_completeness","fouls_per90","fouled_per90",
                  "offsides_per90","crosses_per90","interceptions_per90",
                  "tackles_won_per90","pen_area_ratio","own_goals","discipline_score"]:
            feats[k] = 0.0

    # 3. Squad stats — AGAINST split (goalkeeper / defensive)
    sq_ag = cache.squad.get((team_id, season_id, "against"))
    if not sq_ag:
        sq_ag = cache.squad_fallback.get((team_id, "against"))
    if sq_ag:
        gk = sq_ag.get("goalkeeping") or {}
        feats["gk_goals_ag_per90"]    = _j(gk, "gk_goals_against_per90", 1.3)
        feats["gk_shots_faced_per90"] = _j(gk, "gk_shots_on_target_against", 4.0)
        feats["gk_save_pct"]          = _j(gk, "gk_save_pct",          65.0)
        feats["gk_clean_sheets_pct"]  = _j(gk, "gk_clean_sheets_pct",  25.0)
        gk_g = _j(gk, "gk_games", 1.0) or 1
        feats["gk_win_rate"]          = _safe_div(_j(gk, "gk_wins"), gk_g)
        pen_att_gk = _j(gk, "gk_pens_att") or 1
        feats["gk_pen_save_pct"]      = _safe_div(_j(gk, "gk_pens_saved"), pen_att_gk)
        sh_ag = sq_ag.get("shooting") or {}
        feats["opp_shots_per90"]        = _j(sh_ag, "shots_per90")
        feats["opp_shots_on_tgt_per90"] = _j(sh_ag, "shots_on_target_per90")
        feats["opp_goals_per_shot"]     = _j(sh_ag, "goals_per_shot")
    else:
        for k in ["gk_goals_ag_per90","gk_shots_faced_per90","gk_save_pct",
                  "gk_clean_sheets_pct","gk_win_rate","gk_pen_save_pct",
                  "opp_shots_per90","opp_shots_on_tgt_per90","opp_goals_per_shot"]:
            feats[k] = 0.0

    # 4. Recent form
    form_all  = _compute_form(cache, team_id, None,   5, before_date, league_id=league_id)
    form_home = _compute_form(cache, team_id, "home", 5, before_date, league_id=league_id)
    form_away = _compute_form(cache, team_id, "away", 5, before_date, league_id=league_id)
    feats["form_score"]           = form_all["form_score"]
    feats["last3_form_score"]     = form_all["last3_form_score"]
    feats["weighted_form_score"]  = form_all["weighted_form_score"]
    # momentum: positive = improving form, negative = fading
    feats["form_momentum"]        = form_all["last3_form_score"] - form_all["form_score"]
    feats["form_gf_avg"]          = form_all["goals_scored_avg"]
    feats["form_ga_avg"]          = form_all["goals_conceded_avg"]
    feats["win_streak"]           = _f(form_all["win_streak"])
    feats["home_form_score"]      = form_home["form_score"]
    feats["away_form_score"]      = form_away["form_score"]

    # ── Multi-window rolling stats (FootballGPT additions) ────────────────
    # Appended after the existing form block — does NOT replace any of it.
    # Adds 45 all-venue + 45 home-venue + 45 away-venue = 135 new per-team
    # features.  _build_match_features() auto-generates the home_* / away_* /
    # diff_* versions for the full match vector, so nothing extra is needed
    # there for these window stats.
    fw_all  = _compute_form_windows(cache, team_id, None,   before_date, league_id=league_id)
    fw_home = _compute_form_windows(cache, team_id, "home", before_date, league_id=league_id)
    fw_away = _compute_form_windows(cache, team_id, "away", before_date, league_id=league_id)

    feats.update(fw_all)

    # Venue-specific window features stored under separate prefixes so that
    # _build_match_features() can later emit home_home_form_w* / away_away_form_w*
    # and their differentials without any extra code.
    for k, v in fw_home.items():
        feats[f"home_{k}"] = v
    for k, v in fw_away.items():
        feats[f"away_{k}"] = v

    # Days since last completed match — rest/fatigue signal.
    # Computed relative to before_date when building training data (avoids leakage),
    # or relative to today when predicting upcoming matches.
    try:
        team_matches_all = cache.matches_by_team.get(team_id, [])
        past_matches     = _filter_before(team_matches_all, before_date) if before_date else team_matches_all
        if past_matches:
            last_date = _to_date(past_matches[0].get("match_date"))
            ref_date  = _to_date(before_date) if before_date else datetime.date.today()
            if last_date and ref_date:
                feats["days_since_last_match"] = float((ref_date - last_date).days)
            else:
                feats["days_since_last_match"] = 7.0
        else:
            feats["days_since_last_match"] = 7.0
    except Exception:
        feats["days_since_last_match"] = 7.0

    # ELO-style strength proxy: blend rank, form and points
    # Normalised to [0, 1] — higher = stronger team right now
    feats["elo_proxy"] = (
        feats.get("rank_norm",   0.5) * 0.40
        + feats["form_score"]        * 0.35
        + min(feats.get("points_avg", 0.0) / 3.0, 1.0) * 0.25
    )

    feats["xg_estimate"]     = (
        feats.get("shots_on_target_per90", 0.0) * feats.get("goals_per_shot_on_target", 0.0)
        if feats.get("shots_on_target_per90")
        else feats.get("goals_per90", 1.2)
    )

    # 5. Player features
    player_rows = cache.players.get((team_id, season_id), [])
    feats.update(_build_player_features(cache, team_id, season_id))
    feats.update(_build_injury_features(cache, team_id, player_rows))

    # 6. Previous season
    feats.update(_build_prev_season_features(cache, team_id, league_id))
    feats.update(_build_prev_season_form(cache, team_id, before_date))

    # 7. Scoring patterns
    feats.update(_build_scoring_patterns(cache, team_id, season_id, before_date))

    # 8. Venue stats (home/away splits from team_venue_stats)
    feats.update(_build_venue_stats(cache, team_id, season_id))

    return feats


# ─── Match feature vector (in-memory) ─────────────────────────────────────────

def _get_domestic_league_id(cache: DataCache, team_id: int) -> Optional[int]:
    """
    Return the league_id of the team's primary domestic competition.
    Determined by which league has the most standing entries for this team
    (most games played = most likely their domestic league).
    Returns None if the team has no standings data at all.
    """
    best_lid   = None
    best_games = -1
    for (tid, lid, sid), row in cache.standings.items():
        if tid != team_id:
            continue
        g = int(_f(row.get("games"), 0))
        if g > best_games:
            best_games = g
            best_lid   = lid
    return best_lid


def _build_match_features(cache: DataCache, home_team_id: int, away_team_id: int,
                           league_id: int, season_id: int,
                           match_id: Optional[int] = None, match_date = None):
    """Same output format as feature_engineering.build_match_features()."""
    home_feats = _build_team_features(cache, home_team_id, league_id, season_id, match_date)
    away_feats = _build_team_features(cache, away_team_id, league_id, season_id, match_date)
    h2h        = _compute_h2h(cache, home_team_id, away_team_id, 10, match_date)
    enrichment = _build_enrichment_features(cache, match_id, match_date, home_team_id, away_team_id, league_id)

    vector: dict = {}
    for k, v in home_feats.items():
        vector[f"home_{k}"] = v
    for k, v in away_feats.items():
        vector[f"away_{k}"] = v
        if k in home_feats:
            vector[f"diff_{k}"] = home_feats[k] - v
    for k, v in h2h.items():
        if k != "h2h_last_5":
            vector[k] = _f(v)

    # ── League-style context (6 features) ──────────────────────────────────
    # Mirrors feature_engineering.compute_league_style() — same keys so the
    # trained model sees the same feature names at inference time.
    style = cache.league_style.get(league_id, dict(_LEAGUE_STYLE_DEFAULTS))
    vector.update(style)

    # ── League-relative team strength (4 features) ─────────────────────────
    lg_goals   = style["league_goals_pg"] or 2.70
    half_goals = lg_goals / 2
    vector["home_attack_rel_league"]  = _safe_div(
        home_feats.get("goals_for_pg", 0.0), half_goals, 1.0
    )
    vector["away_attack_rel_league"]  = _safe_div(
        away_feats.get("goals_for_pg", 0.0), half_goals, 1.0
    )
    vector["home_defence_rel_league"] = _safe_div(
        half_goals, home_feats.get("goals_against_pg", half_goals) or half_goals, 1.0
    )
    vector["away_defence_rel_league"] = _safe_div(
        half_goals, away_feats.get("goals_against_pg", half_goals) or half_goals, 1.0
    )

    vector["home_advantage"] = 1.0

    # ── ELO differential (1 strong feature) ──────────────────────────────
    # elo_diff > 0  → home team stronger, < 0 → away team stronger
    vector["elo_diff"] = home_feats.get("elo_proxy", 0.5) - away_feats.get("elo_proxy", 0.5)

    # ── Table position differential (FootballGPT addition) ────────────────
    # rank_norm: 1.0 = 1st place, 0.0 = last place (already in both feat dicts).
    # Positive diff → home team sits higher in the table than the away team.
    # Absolute value captures mismatch magnitude regardless of direction
    # (large value = big quality gap = outcome easier to predict).
    # Uses rank_norm which is computed in section 1 of _build_team_features()
    # from league_standings — zero extra queries, zero new data dependencies.
    _home_rank = home_feats.get("rank_norm", 0.5)
    _away_rank = away_feats.get("rank_norm", 0.5)
    vector["table_position_diff"]     = round(_home_rank - _away_rank, 4)
    vector["table_position_diff_abs"] = round(abs(_home_rank - _away_rank), 4)

    # ── Interaction features (6 derived, zero extra DB queries) ───────────
    # The model sees home_attack_strength and away_defence_strength as separate
    # inputs but not their product — the Dixon-Coles lambda proxy. These 6
    # interactions expose the multiplicative relationships the linear feature
    # space misses, without adding any new data source.

    # 1 & 2: Attack × opponent defence (DC lambda proxy — best single predictor)
    vector["home_attack_x_away_def"] = (
        home_feats.get("attack_strength", 1.0) * away_feats.get("defence_strength", 1.0)
    )
    vector["away_attack_x_home_def"] = (
        away_feats.get("attack_strength", 1.0) * home_feats.get("defence_strength", 1.0)
    )

    # 3: Form momentum differential — who is trending up right now
    vector["form_momentum_diff"] = (
        home_feats.get("form_momentum", 0.0) - away_feats.get("form_momentum", 0.0)
    )

    # 4: Points average differential — current season table quality gap
    vector["points_avg_diff"] = (
        home_feats.get("points_avg", 0.0) - away_feats.get("points_avg", 0.0)
    )

    # 5: Rest advantage — positive means away team had more days to recover
    vector["rest_advantage"] = (
        away_feats.get("days_since_last_match", 7.0)
        - home_feats.get("days_since_last_match", 7.0)
    )

    # 6: H2H recency-weighted home win rate (uses already-computed h2h dict)
    # Weight by number of H2H games so fixture-count uncertainty is reflected.
    h2h_n = h2h.get("h2h_home_wins", 0) + h2h.get("h2h_draws", 0) + h2h.get("h2h_away_wins", 0)
    h2h_weight = min(1.0, h2h_n / 5.0)   # fully trusted at 5+ H2H games
    vector["h2h_weighted_hw_pct"] = (
        h2h.get("h2h_home_win_pct", 0.44) * h2h_weight
        + 0.44 * (1.0 - h2h_weight)        # blend toward league-average prior
    )

    # ── Inject Enrichment Data (Odds & ClubElo) ──────────────────────────
    vector.update(enrichment)

    # ── Cross-league context features ────────────────────────────────────
    # Detects when teams are playing outside their domestic competition
    # (e.g. Europa League, Champions League, cross-border cup).
    # These features tell the model:
    #   - Whether this is a cross-league fixture at all
    #   - How strong each team's domestic league is relative to the other
    #   - Whether each team's domestic stats are from the same competition
    # Without these, the model naively compares e.g. Primeira Liga stats
    # directly against Premier League stats with no adjustment signal.
    home_domestic_lid = _get_domestic_league_id(cache, home_team_id)
    away_domestic_lid = _get_domestic_league_id(cache, away_team_id)

    home_is_cross = (home_domestic_lid is not None and home_domestic_lid != league_id)
    away_is_cross = (away_domestic_lid is not None and away_domestic_lid != league_id)
    is_cross_league = home_is_cross or away_is_cross

    vector["is_cross_league"]       = 1.0 if is_cross_league  else 0.0
    vector["home_is_cross_league"]  = 1.0 if home_is_cross    else 0.0
    vector["away_is_cross_league"]  = 1.0 if away_is_cross    else 0.0

    # Domestic league strength proxy: avg goals/game in each team's home league.
    # Higher = more open/attacking league. Helps compare teams from different leagues.
    home_dom_style = cache.league_style.get(home_domestic_lid, dict(_LEAGUE_STYLE_DEFAULTS)) if home_domestic_lid else dict(_LEAGUE_STYLE_DEFAULTS)
    away_dom_style = cache.league_style.get(away_domestic_lid, dict(_LEAGUE_STYLE_DEFAULTS)) if away_domestic_lid else dict(_LEAGUE_STYLE_DEFAULTS)

    vector["home_domestic_league_goals_pg"]      = _f(home_dom_style.get("league_goals_pg", 2.70))
    vector["away_domestic_league_goals_pg"]      = _f(away_dom_style.get("league_goals_pg", 2.70))
    vector["home_domestic_league_home_win_rate"] = _f(home_dom_style.get("league_home_win_rate", 0.44))
    vector["away_domestic_league_home_win_rate"] = _f(away_dom_style.get("league_home_win_rate", 0.44))

    # Difference in domestic league quality: positive = home team plays in stronger league
    vector["diff_domestic_league_goals_pg"] = (
        vector["home_domestic_league_goals_pg"] - vector["away_domestic_league_goals_pg"]
    )

    # When cross-league: re-normalise each team's attack strength to their OWN domestic league.
    # This replaces the single shared league_goals_pg baseline with per-team baselines,
    # correcting the distortion that caused Porto (weak league) to show 100% better stats.
    home_dom_half = (vector["home_domestic_league_goals_pg"] / 2) or 1.35
    away_dom_half = (vector["away_domestic_league_goals_pg"] / 2) or 1.35
    vector["home_attack_rel_domestic"]  = _safe_div(home_feats.get("goals_for_pg",     0.0), home_dom_half, 1.0)
    vector["away_attack_rel_domestic"]  = _safe_div(away_feats.get("goals_for_pg",     0.0), away_dom_half, 1.0)
    vector["home_defence_rel_domestic"] = _safe_div(home_dom_half, home_feats.get("goals_against_pg", home_dom_half) or home_dom_half, 1.0)
    vector["away_defence_rel_domestic"] = _safe_div(away_dom_half, away_feats.get("goals_against_pg", away_dom_half) or away_dom_half, 1.0)

    feature_names  = sorted(vector.keys())
    feature_values = [vector[k] for k in feature_names]
    return feature_values, feature_names, home_feats, away_feats, h2h


# ─── Public API ───────────────────────────────────────────────────────────────

def build_training_dataset_fast(cur, skip_errors: bool = True):
    """
    Drop-in replacement for feature_engineering.build_training_dataset().

    1. Bulk-loads all tables into a DataCache (~6 queries total).
    2. Iterates over completed matches and builds features in-memory.
    3. Returns X, y, match_ids, match_dates, league_ids, errors.
       match_dates is used for recency weighting.
       league_ids is used for league-tier weighting during training.

    Expected speed-up: 50–100× vs the original (seconds not minutes).
    """
    cache = DataCache(cur)

    # Fetch all training matches (already in cache, but we need the full list
    # with training labels — same query as the original)
    completed = [
        m for m in cache.all_matches
        if m.get("league_id") is not None and m.get("season_id") is not None
    ]
    log.info("Building features for %d completed matches…", len(completed))

    X, y, match_ids, match_dates, league_ids = [], [], [], [], []
    errors = 0

    for m in completed:
        try:
            fv, _, _, _, _ = _build_match_features(
                cache,
                m["home_team_id"], m["away_team_id"],
                m["league_id"],    m["season_id"],
                match_id=m["id"],  match_date=m.get("match_date")
            )
            hs = _f(m["home_score"]); as_ = _f(m["away_score"])
            label = 0 if hs > as_ else (1 if hs == as_ else 2)
            X.append(fv); y.append(label); match_ids.append(m["id"])
            match_dates.append(m.get("match_date"))
            league_ids.append(m.get("league_id"))
        except Exception:
            errors += 1
            if not skip_errors:
                raise

    log.info("Dataset built: %d samples, %d errors skipped.", len(X), errors)
    return X, y, match_ids, match_dates, league_ids, errors
