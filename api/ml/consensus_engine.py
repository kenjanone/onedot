"""
Dynamic Consensus Engine
========================
Aggregates three independent prediction engines into one synthesised output:
  1. DC Engine      — Dixon-Coles Poisson ensemble (dc_engine.py)
  2. ML Engine      — XGBoost + RandomForest ensemble (prediction_engine.py)
  3. Legacy Engine  — Heuristic rule-based predictor (predictions.py helpers)

Dynamic Weighting:
  Before blending, queries prediction_log for each engine's 30-day trailing
  accuracy.  Engines that have proved more accurate recently receive a higher
  weight in the blend.  Falls back to fixed defaults if < MIN_GRADED_ROWS rows
  have been graded.

Default weights (when insufficient history):
  DC 45% | ML 35% | Legacy 20%

Usage:
  from ml.consensus_engine import run_consensus
  result = run_consensus(home_team_id, away_team_id, league_id, season_id)
"""

import logging
import math
from typing import Optional

from database import get_connection

log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "dc":         0.45,
    "ml":         0.30,
    "enrichment": 0.00,
    "legacy":     0.25,
}

# Enrichment engine toggle — set True only when the enrichment engine is wired up
ENRICHMENT_ENABLED = False

# Minimum graded rows required before we trust historical weights
MIN_GRADED_ROWS = 10

# Trailing window for accuracy calculation (days)
ACCURACY_WINDOW_DAYS = 30

OUTCOME_LABELS = ["Home Win", "Draw", "Away Win"]

# ─── Lazy market-calculation imports (avoids circular import at module load) ───

def _get_market_calculator():
    from ml.market_calculator import compute_all_markets, blend_markets, select_best_bets, _override_1x2
    return compute_all_markets, blend_markets, select_best_bets, _override_1x2

def _get_market_weights():
    from ml.market_weights import fetch_per_market_weights
    return fetch_per_market_weights

def _get_recalibrator():
    from ml.market_recalibrator import get_market_recalibrator
    return get_market_recalibrator()


# ─── Legacy heuristic helpers (self-contained, no circular imports) ───────────

def _safe_div(a, b, default=0.0):
    try:
        return float(a) / float(b) if b else default
    except Exception:
        return default


def _run_legacy_engine(cur, home_team_id: int, away_team_id: int) -> dict:
    """
    Lightweight heuristic prediction.
    Inputs (all single SQL queries, no heavy feature engineering):
      1. league_standings  — win rate, goals/game  (original signals)
      2. team_squad_stats  — goals/game            (original signal)
      3. matches (last 5) — recent form pts/game   (NEW)
      4. matches (h2h 8)  — H2H win rate           (NEW)

    Blending weights keep this heuristic — no ML, just a strength ratio.
    """
    import numpy as np  # lazy import
    FALLBACK = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35,
                "predicted_outcome": "Home Win"}
    try:
        # ── 1. Standings ──────────────────────────────────────────────────────
        cur.execute("""
            SELECT ls.wins, ls.ties, ls.losses, ls.games,
                   ls.goals_for, ls.goals_against, ls.points
            FROM   league_standings ls
            WHERE  ls.team_id = %s
            ORDER  BY ls.season_id DESC LIMIT 1
        """, (home_team_id,))
        h_st = cur.fetchone()

        cur.execute("""
            SELECT ls.wins, ls.ties, ls.losses, ls.games,
                   ls.goals_for, ls.goals_against, ls.points
            FROM   league_standings ls
            WHERE  ls.team_id = %s
            ORDER  BY ls.season_id DESC LIMIT 1
        """, (away_team_id,))
        a_st = cur.fetchone()

        # ── 2. Squad stats ────────────────────────────────────────────────────
        cur.execute("""
            SELECT ts.goals, ts.games, ts.possession
            FROM   team_squad_stats ts
            WHERE  ts.team_id = %s AND ts.split = 'for'
            ORDER  BY ts.season_id DESC LIMIT 1
        """, (home_team_id,))
        h_sq = cur.fetchone()

        cur.execute("""
            SELECT ts.goals, ts.games, ts.possession
            FROM   team_squad_stats ts
            WHERE  ts.team_id = %s AND ts.split = 'for'
            ORDER  BY ts.season_id DESC LIMIT 1
        """, (away_team_id,))
        a_sq = cur.fetchone()

        # ── 3. Recent form — last 5 matches (NEW) ─────────────────────────────
        # Points per game from last 5: win=3, draw=1, loss=0
        # Gives a momentum signal the season standings can't capture.
        def _form_ppg(team_id: int) -> float:
            cur.execute("""
                SELECT home_team_id, away_team_id, home_score, away_score
                FROM   matches
                WHERE  (home_team_id = %s OR away_team_id = %s)
                  AND  home_score IS NOT NULL
                ORDER  BY match_date DESC
                LIMIT  5
            """, (team_id, team_id))
            rows = cur.fetchall()
            if not rows:
                return 1.2  # neutral default (40% win rate × 3pts)
            pts = 0
            for r in rows:
                if r["home_team_id"] == team_id:
                    gf, ga = r["home_score"], r["away_score"]
                else:
                    gf, ga = r["away_score"], r["home_score"]
                pts += 3 if gf > ga else (1 if gf == ga else 0)
            return pts / len(rows)  # 0–3 scale

        h_form = _form_ppg(home_team_id)
        a_form = _form_ppg(away_team_id)

        # ── 4. Head-to-head — last 8 meetings (NEW) ───────────────────────────
        # H2H home win rate as a nudge factor.
        # Low weight (10%) to avoid overfitting on small samples.
        cur.execute("""
            SELECT home_team_id, away_team_id, home_score, away_score
            FROM   matches
            WHERE  ((home_team_id = %s AND away_team_id = %s)
                 OR (home_team_id = %s AND away_team_id = %s))
              AND  home_score IS NOT NULL
            ORDER  BY match_date DESC
            LIMIT  8
        """, (home_team_id, away_team_id, away_team_id, home_team_id))
        h2h_rows = cur.fetchall()

        h2h_home_wins = h2h_draws = h2h_away_wins = 0
        for r in h2h_rows:
            # Normalise: always from the perspective of home_team_id
            if r["home_team_id"] == home_team_id:
                gf, ga = r["home_score"], r["away_score"]
            else:
                gf, ga = r["away_score"], r["home_score"]
            if gf > ga:   h2h_home_wins += 1
            elif gf == ga: h2h_draws    += 1
            else:          h2h_away_wins += 1

        h2h_total  = h2h_home_wins + h2h_draws + h2h_away_wins or 1
        h2h_h_rate = h2h_home_wins / h2h_total   # 0–1
        h2h_d_rate = h2h_draws     / h2h_total
        h2h_a_rate = h2h_away_wins / h2h_total

        # ── Base strength from original signals ───────────────────────────────
        h_gpg = _safe_div(h_sq["goals"] if h_sq else 0,
                           h_sq["games"] if h_sq else 1, 1.2)
        a_gpg = _safe_div(a_sq["goals"] if a_sq else 0,
                           a_sq["games"] if a_sq else 1, 1.0)
        h_wr  = _safe_div(h_st["wins"] if h_st else 0,
                           h_st["games"] if h_st else 1, 0.40)
        a_wr  = _safe_div(a_st["wins"] if a_st else 0,
                           a_st["games"] if a_st else 1, 0.35)

        # Normalise form to 0–1 scale (max possible = 3.0 pts/game)
        h_form_n = h_form / 3.0
        a_form_n = a_form / 3.0

        # ── Blended strength score ─────────────────────────────────────────────
        # Original: 0.6 × gpg + 0.4 × wr
        # New:      0.45 × gpg + 0.30 × wr + 0.25 × form  (form replaces some gpg weight)
        # H2H applied as a separate nudge AFTER normalisation (keeps it lightweight)
        h_str = 0.45 * h_gpg + 0.30 * h_wr + 0.25 * h_form_n
        a_str = 0.45 * a_gpg + 0.30 * a_wr + 0.25 * a_form_n
        total = h_str + a_str + 0.001

        # ── Draw rate ─────────────────────────────────────────────────────────
        h_ties = h_st["ties"] if h_st else 0
        a_ties = a_st["ties"] if a_st else 0
        h_g    = h_st["games"] if h_st else 1
        a_g    = a_st["games"] if a_st else 1

        base_draw_rate = (h_ties + a_ties) / (h_g + a_g) if (h_g + a_g) > 0 else 0.25
        # Blend with H2H draw rate when we have enough H2H data
        if h2h_total >= 4:
            base_draw_rate = 0.7 * base_draw_rate + 0.3 * h2h_d_rate
        base_draw_rate = max(0.15, min(0.35, base_draw_rate))

        strength_gap = abs(h_str - a_str) / total
        r_d = base_draw_rate * (1.0 - (strength_gap * 0.7))

        r_h = (h_str / total) + 0.04   # home advantage nudge
        r_a = (a_str / total) - 0.04

        scale = (1.0 - r_d) / (r_h + r_a) if (r_h + r_a) > 0 else 1.0
        r_h *= scale
        r_a *= scale

        # ── H2H nudge (10% weight, only when 4+ meetings exist) ───────────────
        if h2h_total >= 4:
            r_h = 0.90 * r_h + 0.10 * h2h_h_rate
            r_a = 0.90 * r_a + 0.10 * h2h_a_rate
            r_d = 0.90 * r_d + 0.10 * h2h_d_rate

        r_h = max(0.05, r_h)
        r_a = max(0.05, r_a)
        r_d = max(0.05, r_d)

        s      = r_h + r_a + r_d
        home_p = round(r_h / s, 4)
        away_p = round(r_a / s, 4)
        draw_p = round(r_d / s, 4)

        probs   = [home_p, draw_p, away_p]
        idx     = int(np.argmax(probs))
        outcome = OUTCOME_LABELS[idx]

        return {
            "home_win":          home_p,
            "draw":              draw_p,
            "away_win":          away_p,
            "predicted_outcome": outcome,
            "home_xg": round(min(max(h_gpg * 1.05, 0.30), 4.0), 2),
            "away_xg": round(min(max(a_gpg * 0.95, 0.30), 4.0), 2),
        }
    except Exception as exc:
        log.warning("Legacy engine error: %s", exc)
        return {**FALLBACK, "home_xg": 1.35, "away_xg": 1.10}


# ─── Dynamic weight computation ───────────────────────────────────────────────

def _fetch_dynamic_weights(cur) -> dict:
    """
    Compute per-engine consensus weights using SKILL-ADJUSTED accuracy.

    Raw accuracy rewards DC for guessing Home Win 85% of the time (since
    Home Wins happen ~44% of the time anyway). Instead we measure:
      skill  = accuracy - naive_baseline (always-Home-Win score)
      diversity = fraction of predictions that are NOT Home Win

    Weight = 70% skill + 30% diversity.
    This penalises engines that only guess Home Win and rewards engines
    that correctly identify Away Wins and Draws.
    Falls back to DEFAULT_WEIGHTS if not enough graded rows exist.
    """
    try:
        cur.execute(f"""
            SELECT
                COUNT(*)                                                     AS total,
                SUM(CASE WHEN dc_correct         THEN 1 ELSE 0 END)::float  AS dc_correct,
                SUM(CASE WHEN ml_correct         THEN 1 ELSE 0 END)::float  AS ml_correct,
                SUM(CASE WHEN enrichment_correct THEN 1 ELSE 0 END)::float  AS enrichment_correct,
                SUM(CASE WHEN legacy_correct     THEN 1 ELSE 0 END)::float  AS legacy_correct,
                SUM(CASE WHEN actual = 'Home Win' THEN 1 ELSE 0 END)::float AS actual_hw_count,
                SUM(CASE WHEN dc_predicted_outcome     != 'Home Win' THEN 1 ELSE 0 END)::float AS dc_non_hw,
                SUM(CASE WHEN ml_predicted_outcome     != 'Home Win' THEN 1 ELSE 0 END)::float AS ml_non_hw,
                SUM(CASE WHEN legacy_predicted_outcome != 'Home Win' THEN 1 ELSE 0 END)::float AS leg_non_hw
            FROM prediction_log
            WHERE correct IS NOT NULL
              AND evaluated_at >= NOW() - INTERVAL '{ACCURACY_WINDOW_DAYS} days'
              AND dc_predicted_outcome         IS NOT NULL
              AND ml_predicted_outcome         IS NOT NULL
              AND legacy_predicted_outcome     IS NOT NULL
        """)
        row = cur.fetchone()
        if not row:
            return DEFAULT_WEIGHTS.copy()

        total = int(row["total"] or 0)
        if total < MIN_GRADED_ROWS:
            log.info("Consensus: only %d graded rows → using default weights", total)
            return DEFAULT_WEIGHTS.copy()

        dc_acc     = float(row["dc_correct"]     or 0) / total
        ml_acc     = float(row["ml_correct"]     or 0) / total
        enr_acc    = 0.0
        legacy_acc = float(row["legacy_correct"] or 0) / total
        naive      = float(row["actual_hw_count"] or 0) / total

        # Skill = accuracy above the naive always-Home-Win baseline
        MIN_SKILL = 0.05
        dc_skill     = max(dc_acc     - naive, MIN_SKILL)
        ml_skill     = max(ml_acc     - naive, MIN_SKILL)
        enr_skill    = max(enr_acc    - naive, MIN_SKILL)
        legacy_skill = max(legacy_acc - naive, MIN_SKILL)

        # Diversity = fraction of non-Home-Win predictions
        dc_div     = float(row["dc_non_hw"]  or 0) / total
        ml_div     = float(row["ml_non_hw"]  or 0) / total
        leg_div    = float(row["leg_non_hw"] or 0) / total
        enr_div    = 0.20  # enrichment: assumed moderate diversity

        SKILL_W, DIV_W = 0.70, 0.30
        dc_score     = SKILL_W * dc_skill     + DIV_W * dc_div
        ml_score     = SKILL_W * ml_skill     + DIV_W * ml_div
        enr_score    = SKILL_W * enr_skill    + DIV_W * enr_div
        legacy_score = SKILL_W * legacy_skill + DIV_W * leg_div

        weight_sum = dc_score + ml_score + enr_score + legacy_score
        if weight_sum < 0.01:
            return DEFAULT_WEIGHTS.copy()

        weights = {
            "dc":         round(dc_score     / weight_sum, 4),
            "ml":         round(ml_score     / weight_sum, 4),
            "enrichment": round(enr_score    / weight_sum, 4),
            "legacy":     round(legacy_score / weight_sum, 4),
        }
        log.info(
            "Consensus: skill-adjusted weights from %d rows — "
            "DC=%.1f%% ML=%.1f%% ENR=%.1f%% Legacy=%.1f%% (naive=%.1f%%)",
            total, weights["dc"]*100, weights["ml"]*100,
            weights["enrichment"]*100, weights["legacy"]*100, naive*100,
        )
        return weights

    except Exception as exc:
        log.warning("Consensus: dynamic weight query failed (%s) → using defaults", exc)
        return DEFAULT_WEIGHTS.copy()


def _fetch_per_outcome_weights(cur) -> dict:
    """
    Query prediction_log for each engine's accuracy broken down by actual outcome
    (Home Win / Draw / Away Win). Returns per-outcome weight dicts so the
    consensus can trust whichever engine is historically best for each outcome.

    Returns:
        {
          "Home Win": {"dc": w, "ml": w, "enrichment": w, "legacy": w},
          "Draw":     {"dc": w, "ml": w, "enrichment": w, "legacy": w},
          "Away Win": {"dc": w, "ml": w, "enrichment": w, "legacy": w},
        }
    Falls back to None if not enough data (caller uses global weights instead).
    """
    try:
        cur.execute("""
            SELECT
                actual,
                COUNT(*) FILTER (WHERE dc_correct     IS NOT NULL) AS dc_n,
                SUM(CASE WHEN dc_correct     THEN 1 ELSE 0 END)::float AS dc_correct,
                COUNT(*) FILTER (WHERE ml_correct     IS NOT NULL) AS ml_n,
                SUM(CASE WHEN ml_correct     THEN 1 ELSE 0 END)::float AS ml_correct,
                COUNT(*) FILTER (WHERE legacy_correct IS NOT NULL) AS leg_n,
                SUM(CASE WHEN legacy_correct THEN 1 ELSE 0 END)::float AS leg_correct
            FROM prediction_log
            WHERE actual IS NOT NULL
              AND evaluated_at >= NOW() - INTERVAL '90 days'
            GROUP BY actual
            HAVING COUNT(*) >= 10
        """)
        rows = cur.fetchall()
        if not rows:
            return None

        result = {}
        for row in rows:
            outcome = str(row["actual"]).strip()  # "Home Win", "Draw", "Away Win"
            dc_n,  ml_n,  leg_n  = int(row["dc_n"] or 0), int(row["ml_n"] or 0), int(row["leg_n"] or 0)
            dc_acc  = float(row["dc_correct"]  or 0) / dc_n  if dc_n  > 0 else 0.0
            ml_acc  = float(row["ml_correct"]  or 0) / ml_n  if ml_n  > 0 else 0.0
            leg_acc = float(row["leg_correct"] or 0) / leg_n if leg_n > 0 else 0.0
            enr_acc = 0.0   # enrichment engine not yet per-outcome tracked

            # Apply a small floor (0.05) so no engine gets weight=0 and becomes
            # completely silenced — we still want some diversity in the blend.
            MIN_W = 0.05
            dc_acc  = max(dc_acc,  MIN_W)
            ml_acc  = max(ml_acc,  MIN_W)
            leg_acc = max(leg_acc, MIN_W)
            enr_acc = max(enr_acc, MIN_W)

            wsum = dc_acc + ml_acc + leg_acc + enr_acc
            result[outcome] = {
                "dc":         round(dc_acc  / wsum, 4),
                "ml":         round(ml_acc  / wsum, 4),
                "enrichment": round(enr_acc / wsum, 4),
                "legacy":     round(leg_acc / wsum, 4),
            }

        # Only return if we got all three outcome types
        if len(result) == 3:
            log.info(
                "Consensus per-outcome weights — HW: DC=%.0f%% ML=%.0f%% Leg=%.0f%% | "
                "D: DC=%.0f%% ML=%.0f%% Leg=%.0f%% | "
                "AW: DC=%.0f%% ML=%.0f%% Leg=%.0f%%",
                result.get("Home Win", {}).get("dc", 0)*100,
                result.get("Home Win", {}).get("ml", 0)*100,
                result.get("Home Win", {}).get("legacy", 0)*100,
                result.get("Draw", {}).get("dc", 0)*100,
                result.get("Draw", {}).get("ml", 0)*100,
                result.get("Draw", {}).get("legacy", 0)*100,
                result.get("Away Win", {}).get("dc", 0)*100,
                result.get("Away Win", {}).get("ml", 0)*100,
                result.get("Away Win", {}).get("legacy", 0)*100,
            )
            return result
        return None
    except Exception as exc:
        log.warning("Consensus: per-outcome weight query failed (%s)", exc)
        return None


# ─── Probability blending ─────────────────────────────────────────────────────

def _blend(dc: dict, ml: dict, enrichment: dict, legacy: dict,
           weights: dict, per_outcome_weights: dict = None) -> dict:
    """
    Weighted linear blend of four probability distributions.
    If per_outcome_weights is provided, uses separate engine weights for each
    outcome slot (Home Win / Draw / Away Win) so the best engine for each
    outcome type is trusted most. Falls back to global weights otherwise.
    """
    def _slot(outcome_key: str, engine_key: str) -> float:
        """Pick weight: per-outcome if available, else global."""
        if per_outcome_weights and outcome_key in per_outcome_weights:
            return per_outcome_weights[outcome_key].get(engine_key, 0.0)
        return weights.get(engine_key, 0.0)

    hw = (
        _slot("Home Win", "dc")         * dc["home_win"] +
        _slot("Home Win", "ml")         * ml["home_win"] +
        _slot("Home Win", "enrichment") * enrichment["home_win"] +
        _slot("Home Win", "legacy")     * legacy["home_win"]
    )
    dr = (
        _slot("Draw", "dc")         * dc["draw"] +
        _slot("Draw", "ml")         * ml["draw"] +
        _slot("Draw", "enrichment") * enrichment["draw"] +
        _slot("Draw", "legacy")     * legacy["draw"]
    )
    aw = (
        _slot("Away Win", "dc")         * dc["away_win"] +
        _slot("Away Win", "ml")         * ml["away_win"] +
        _slot("Away Win", "enrichment") * enrichment["away_win"] +
        _slot("Away Win", "legacy")     * legacy["away_win"]
    )

    total = hw + dr + aw or 1.0
    hw, dr, aw = hw / total, dr / total, aw / total

    return {
        "home_win": round(float(hw), 4),
        "draw":     round(float(dr), 4),
        "away_win": round(float(aw), 4),
    }


def _confidence_from_entropy(probs: dict) -> tuple:
    """Shannon-entropy-based confidence score (0–100) and label."""
    import numpy as np  # lazy import
    p = np.array([probs["home_win"], probs["draw"], probs["away_win"]])
    p = np.clip(p, 1e-10, 1.0)
    entropy     = float(-np.sum(p * np.log(p)))
    max_entropy = float(-np.log(1.0 / 3.0))
    score = round((1.0 - entropy / max_entropy) * 100, 1)
    if score >= 30:
        label = "High"
    elif score >= 15:
        label = "Medium"
    else:
        label = "Low"
    return score, label


def _agreement_level(dc_out: str, ml_out: str, enr_out: str, legacy_out: str) -> str:
    outcomes = [dc_out, ml_out, enr_out, legacy_out]
    unique = len(set(outcomes))
    if unique == 1:
        return "full"
    if unique == 2:
        # Check if 3 engines agree (majority) vs 2-2 tie
        from collections import Counter
        counts = Counter(outcomes).values()
        if max(counts) >= 3:
            return "majority"
    return "split"


# ─── Main public function ─────────────────────────────────────────────────────

def run_consensus(
    home_team_id: int,
    away_team_id: int,
    league_id:    int,
    season_id:    int,
) -> dict:
    """
    Run all three engines, blend results with dynamic weights, return a
    comprehensive consensus prediction.

    Return schema:
      consensus:   {home_win, draw, away_win, predicted_outcome,
                    confidence, confidence_score}
      engines:     {dc: {...}, ml: {...}, legacy: {...}}
      weights_used:{dc, ml, legacy, source}
      agreement:   "full" | "majority" | "split"
      markets:     {btts_yes, over_2_5, ...}
    """
    conn = get_connection()
    cur  = conn.cursor()

    errors = []

    try:
        # ── 1. Compute dynamic weights ─────────────────────────────────────────
        weights = _fetch_dynamic_weights(cur)
        weight_source = (
            "dynamic_historical"
            if weights != DEFAULT_WEIGHTS
            else "default_fallback"
        )

        # ── 2. Run DC engine ──────────────────────────────────────────────────
        try:
            from ml.dc_engine import predict_dc_match  # lazy import
            import numpy as np
            dc_raw = predict_dc_match(home_team_id, away_team_id, league_id=league_id)
            if "error" in dc_raw:
                raise RuntimeError(dc_raw["error"])
            dc_probs = {
                "home_win": float(dc_raw.get("calibrated", dc_raw.get("blended", {})).get("home_win", 0.35)),
                "draw":     float(dc_raw.get("calibrated", dc_raw.get("blended", {})).get("draw",     0.30)),
                "away_win": float(dc_raw.get("calibrated", dc_raw.get("blended", {})).get("away_win", 0.35)),
            }
            # Normalise
            s = sum(dc_probs.values()) or 1
            dc_probs = {k: round(v / s, 4) for k, v in dc_probs.items()}
            dc_outcome_idx = int(np.argmax([dc_probs["home_win"], dc_probs["draw"], dc_probs["away_win"]]))
            dc_outcome = OUTCOME_LABELS[dc_outcome_idx]
            # Extract DC xG for market computation
            _dc_bl = dc_raw.get("blended") or {}
            dc_xg_h = float(_dc_bl.get("exp_home_goals") or _dc_bl.get("home_xg") or 1.35)
            dc_xg_a = float(_dc_bl.get("exp_away_goals") or _dc_bl.get("away_xg") or 1.10)
        except Exception as exc:
            log.warning("Consensus: DC engine error: %s", exc)
            errors.append(f"dc: {exc}")
            dc_probs   = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            dc_outcome = "Home Win"
            dc_xg_h, dc_xg_a = 1.35, 1.10
            weights["dc"] = 0.0   # zero-weight broken engine

        # ── 3. Run ML engine ──────────────────────────────────────────────────
        try:
            import ml.prediction_engine as ml_engine  # lazy import
            ml_raw = ml_engine.predict_match(home_team_id, away_team_id,
                                             league_id, season_id)
            if "error" in ml_raw:
                raise RuntimeError(ml_raw["error"])
            ml_probs_inner = ml_raw.get("probabilities", {})
            ml_probs = {
                "home_win": float(ml_probs_inner.get("home_win", 0.35)),
                "draw":     float(ml_probs_inner.get("draw",     0.30)),
                "away_win": float(ml_probs_inner.get("away_win", 0.35)),
            }
            s = sum(ml_probs.values()) or 1
            ml_probs = {k: round(v / s, 4) for k, v in ml_probs.items()}
            ml_outcome = ml_raw.get("predicted_outcome", "Home Win")
            ml_match_info = ml_raw.get("match", {})
            home_name = ml_match_info.get("home_team", f"Team {home_team_id}")
            away_name = ml_match_info.get("away_team", f"Team {away_team_id}")
            league_name = ml_match_info.get("league", "")
            season_name = ml_match_info.get("season", "")
            expected_goals = ml_raw.get("expected_goals", {})
        except Exception as exc:
            log.warning("Consensus: ML engine error: %s", exc)
            errors.append(f"ml: {exc}")
            ml_probs   = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            ml_outcome = "Home Win"
            home_name  = f"Team {home_team_id}"
            away_name  = f"Team {away_team_id}"
            league_name = ""
            season_name = ""
            expected_goals = {}
            weights["ml"] = 0.0

        # ── 4. Enrichment engine — disabled until wired up ────────────────────
        if not ENRICHMENT_ENABLED:
            enr_probs   = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            enr_outcome = "\u2014"
            enr_features = {}
            weights["enrichment"] = 0.0
        else:
            try:
                from ml.enrichment_engine import predict_enrichment
                cur.execute(
                    "SELECT match_date FROM matches WHERE home_team_id=%s AND away_team_id=%s AND season_id=%s LIMIT 1",
                    (home_team_id, away_team_id, season_id),
                )
                mr = cur.fetchone()
                m_date = str(mr["match_date"]) if mr and mr["match_date"] else None
                enr_raw = predict_enrichment(home_team_id, away_team_id, m_date)
                if "error" in enr_raw:
                    raise RuntimeError(enr_raw["error"])
                enr_probs = {
                    "home_win": float(enr_raw.get("home_win", 0.35)),
                    "draw":     float(enr_raw.get("draw",     0.30)),
                    "away_win": float(enr_raw.get("away_win", 0.35)),
                }
                s = sum(enr_probs.values()) or 1
                enr_probs = {k: round(v / s, 4) for k, v in enr_probs.items()}
                enr_outcome  = enr_raw.get("predicted_outcome", "Home Win")
                enr_features = enr_raw.get("_features", {})
            except Exception as exc:
                log.warning("Consensus: Enrichment engine error: %s", exc)
                errors.append(f"enrichment: {exc}")
                enr_probs   = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
                enr_outcome = "Home Win"
                enr_features = {}
                weights["enrichment"] = 0.0

        # ── 5. Run Legacy engine ──────────────────────────────────────────────
        try:
            legacy_raw     = _run_legacy_engine(cur, home_team_id, away_team_id)
            legacy_probs   = {k: v for k, v in legacy_raw.items()
                              if k in ("home_win", "draw", "away_win")}
            legacy_outcome = legacy_raw.get("predicted_outcome", "Home Win")
            leg_xg_h = float(legacy_raw.get("home_xg", 1.35))
            leg_xg_a = float(legacy_raw.get("away_xg", 1.10))
        except Exception as exc:
            log.warning("Consensus: Legacy engine error: %s", exc)
            errors.append(f"legacy: {exc}")
            legacy_probs   = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            legacy_outcome = "Home Win"
            leg_xg_h, leg_xg_a = 1.35, 1.10
            weights["legacy"] = 0.0

        # ── 6. Re-normalise weights if any engine failed ─────────────────────
        w_sum = sum(weights.values())
        if w_sum < 0.01:
            weights = DEFAULT_WEIGHTS.copy()
            w_sum = 1.0
        weights = {k: round(v / w_sum, 4) for k, v in weights.items()}

        # ── 7. Per-outcome weights (trust best engine per outcome type) ────────
        per_outcome_w = _fetch_per_outcome_weights(cur)

        # ── 8. Blend ──────────────────────────────────────────────────────────
        blended = _blend(dc_probs, ml_probs, enr_probs, legacy_probs, weights,
                         per_outcome_weights=per_outcome_w)

        # ── 8. Consensus outcome + confidence ─────────────────────────────────
        import numpy as np  # lazy import (already imported above in DC block)
        idx = int(np.argmax([blended["home_win"], blended["draw"], blended["away_win"]]))
        consensus_outcome   = OUTCOME_LABELS[idx]
        confidence_score, confidence_label = _confidence_from_entropy(blended)
        agreement = _agreement_level(dc_outcome, ml_outcome, enr_outcome, legacy_outcome)

        # ── 8. Full market computation from per-engine xG ─────────────────────
        home_xg = float(expected_goals.get("home_xg", 1.35))
        away_xg = float(expected_goals.get("away_xg", 1.10))
        try:
            _calc, _mkt_blend, _bets, _ovr = _get_market_calculator()
            _fetch_pmw = _get_market_weights()

            # Apply league xG bias correction before computing markets
            recal = _get_recalibrator()
            if recal and recal.is_fitted:
                home_xg, away_xg = recal.calibrate_xg(home_xg, away_xg, league_id)
                dc_xg_h, dc_xg_a = recal.calibrate_xg(dc_xg_h, dc_xg_a, league_id)
                leg_xg_h, leg_xg_a = recal.calibrate_xg(leg_xg_h, leg_xg_a, league_id)

            # Per-engine full market sheets
            dc_markets  = _ovr(_calc(dc_xg_h,   dc_xg_a),   dc_probs)
            ml_markets  = _ovr(_calc(home_xg,   away_xg),   ml_probs)
            leg_markets = _ovr(_calc(leg_xg_h,  leg_xg_a),  legacy_probs)

            # Per-market adaptive weights
            _w3 = {"dc": weights.get("dc", 0.45), "ml": weights.get("ml", 0.30), "legacy": weights.get("legacy", 0.25)}
            per_mkt_w = _fetch_pmw(cur, _w3)

            # Blend
            consensus_markets = _mkt_blend(
                {"dc": dc_markets, "ml": ml_markets, "legacy": leg_markets},
                _w3, per_mkt_w,
            )

            # Isotonic calibration
            if recal and recal.is_fitted:
                consensus_markets = recal.calibrate(consensus_markets, engine="consensus", league_id=league_id)

            best_bets = _bets(consensus_markets, home_name, away_name)

            # Derive legacy btts/over for compatibility
            btts_yes = consensus_markets.get("btts_yes", 0.50)
            over_2_5 = consensus_markets.get("over_2_5", 0.50)
        except Exception as _mkt_exc:
            log.debug("Market computation failed in run_consensus: %s", _mkt_exc)
            btts_yes = round(max(0.0, min(1 - math.exp(-home_xg) - math.exp(-away_xg) + math.exp(-(home_xg + away_xg)), 1.0)), 4)
            over_2_5 = round(max(0.0, min(1 - sum((math.exp(-(home_xg + away_xg)) * ((home_xg + away_xg) ** k) / math.factorial(k)) for k in range(3)), 1.0)), 4)
            consensus_markets = {"btts_yes": btts_yes, "btts_no": round(1 - btts_yes, 4), "over_2_5": over_2_5, "under_2_5": round(1 - over_2_5, 4), "home_xg": round(home_xg, 2), "away_xg": round(away_xg, 2)}
            dc_markets = ml_markets = leg_markets = {}
            per_mkt_w = {}
            best_bets = []

        return {
            "match": {
                "home_team":    home_name,
                "away_team":    away_name,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "league":       league_name,
                "season":       season_name,
                "season_id":    season_id,    # needed by prediction_ask for DB context scoping
                "league_id":    league_id,    # needed by prediction_ask for standings filtering
            },
            "consensus": {
                "home_win":        blended["home_win"],
                "draw":            blended["draw"],
                "away_win":        blended["away_win"],
                "predicted_outcome": consensus_outcome,
                "confidence":      confidence_label,
                "confidence_score": confidence_score,
            },
            "engines": {
                "dc": {
                    "home_win":          dc_probs["home_win"],
                    "draw":              dc_probs["draw"],
                    "away_win":          dc_probs["away_win"],
                    "predicted_outcome": dc_outcome,
                },
                "ml": {
                    "home_win":          ml_probs["home_win"],
                    "draw":              ml_probs["draw"],
                    "away_win":          ml_probs["away_win"],
                    "predicted_outcome": ml_outcome,
                },
                "enrichment": {
                    "home_win":          enr_probs["home_win"],
                    "draw":              enr_probs["draw"],
                    "away_win":          enr_probs["away_win"],
                    "predicted_outcome": enr_outcome,
                    "diagnostics":       enr_features,
                },
                "legacy": {
                    "home_win":          legacy_probs["home_win"],
                    "draw":              legacy_probs["draw"],
                    "away_win":          legacy_probs["away_win"],
                    "predicted_outcome": legacy_outcome,
                },
            },
            "weights_used": {**weights, "source": weight_source},
            "agreement":    agreement,
            # Full blended market sheet (all engines, all markets)
            "markets":           consensus_markets,
            "engine_markets":    {"dc": dc_markets, "ml": ml_markets, "legacy": leg_markets},
            "per_market_weights": per_mkt_w,
            "best_bets":          best_bets,
            # Individual engine picks — stored in prediction_log for grading
            "_engine_picks": {
                "dc":         dc_outcome,
                "ml":         ml_outcome,
                "enrichment": enr_outcome,
                "legacy":     legacy_outcome,
            },
            **({"_errors": errors} if errors else {}),
        }

    finally:
        conn.close()

def upcoming_consensus_fast(league_id: int = None, limit: int = 50) -> list:
    """
    Bulk-predicts upcoming matches for the Consensus Engine at sub-second speeds.
    Bypasses the 30-second loop limitation by instantiating the DB DataCache exactly ONCE
    and sharing it across all native engines for the entire fixture set.
    """
    conn = get_connection()
    cur  = conn.cursor()
    errors = []
    
    try:
        # Compute dynamic weights once — save a pristine copy before the loop
        weights = _fetch_dynamic_weights(cur)
        weight_source = "dynamic_historical" if weights != DEFAULT_WEIGHTS else "default_fallback"
        global_weights = dict(weights)  # BUG FIX: never mutate this; copy per fixture

        # Bulk load fixtures
        query = """
            SELECT m.id, m.home_team_id, m.away_team_id,
                   m.league_id, m.season_id, m.match_date, m.gameweek,
                   ht.name AS home_name, at.name AS away_name,
                   ht.logo_url AS home_logo, at.logo_url AS away_logo
            FROM matches m
            JOIN teams ht ON ht.id = m.home_team_id
            JOIN teams at ON at.id = m.away_team_id
            WHERE m.home_score IS NULL AND m.match_date >= CURRENT_DATE
        """
        params = []
        if league_id:
            query += " AND m.league_id = %s"
            params.append(league_id)
        query += " ORDER BY m.match_date ASC LIMIT %s"
        params.append(limit)
        cur.execute(query, params)
        fixtures = [dict(r) for r in cur.fetchall()]

        if not fixtures:
            return []

        # Bulk load DB state exactly once
        from ml.batch_features import DataCache, _build_match_features
        import ml.prediction_engine as ml_engine
        
        ml_cache = DataCache(cur)
        ml_model = ml_engine._get_engine()

        # We need the DC engine in memory too
        from ml.dc_engine import predict_dc_match

        # Load market helpers once for the whole batch
        try:
            _calc, _mkt_blend, _bets, _ovr = _get_market_calculator()
            _fetch_pmw = _get_market_weights()
            _recal     = _get_recalibrator()
        except Exception:
            _calc = _mkt_blend = _bets = _ovr = _fetch_pmw = _recal = None

        results = []
        import numpy as np

        for fx in fixtures:
            # BUG FIX: start each fixture with a fresh copy of the global weights
            # so engine failures in one fixture don't bleed zero-weights into the next
            match_weights = dict(global_weights)

            match_payload = {
                "home_team":    fx["home_name"],
                "away_team":    fx["away_name"],
                "home_team_id": fx["home_team_id"],
                "away_team_id": fx["away_team_id"],
                "home_logo":    fx["home_logo"],
                "away_logo":    fx["away_logo"],
                "league":       "",
                "season":       "",
            }
            
            # --- DC
            try:
                dc_raw = predict_dc_match(fx["home_team_id"], fx["away_team_id"], league_id=fx.get("league_id"))
                if "error" in dc_raw: raise RuntimeError(dc_raw["error"])
                dc_probs = {
                    "home_win": float(dc_raw.get("calibrated", dc_raw.get("blended", {})).get("home_win", 0.35)),
                    "draw":     float(dc_raw.get("calibrated", dc_raw.get("blended", {})).get("draw",     0.30)),
                    "away_win": float(dc_raw.get("calibrated", dc_raw.get("blended", {})).get("away_win", 0.35)),
                }
                s = sum(dc_probs.values()) or 1
                dc_probs = {k: round(v / s, 4) for k, v in dc_probs.items()}
                dc_outcome = OUTCOME_LABELS[int(np.argmax([dc_probs["home_win"], dc_probs["draw"], dc_probs["away_win"]]))]
                _dc_bl = dc_raw.get("blended") or {}
                dc_xg_h = float(_dc_bl.get("exp_home_goals") or _dc_bl.get("home_xg") or 1.35)
                dc_xg_a = float(_dc_bl.get("exp_away_goals") or _dc_bl.get("away_xg") or 1.10)
            except Exception as e:
                dc_probs = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
                dc_outcome = "Home Win"
                dc_xg_h, dc_xg_a = 1.35, 1.10
                match_weights["dc"] = 0.0  # BUG FIX: was weights["dc"] = 0.0
                
            # --- ML Engine (Instant Cache Bypass)
            try:
                if not ml_model or not ml_model.is_trained:
                    raise RuntimeError("ML Model not trained.")
                fv, feat_names, home_feats, away_feats, h2h = _build_match_features(
                    ml_cache, fx["home_team_id"], fx["away_team_id"], fx["league_id"], fx["season_id"],
                    match_id=fx["id"], match_date=fx["match_date"]
                )
                raw_proba = ml_model.predict_proba(fv)
                calibrated = ml_engine._apply_calibration({
                    "home_win": raw_proba["home_win"],
                    "draw":     raw_proba["draw"],
                    "away_win": raw_proba["away_win"]
                })
                # Re-normalize just in case
                sm = sum(calibrated.values()) or 1
                ml_probs = {k: round(v/sm, 4) for k, v in calibrated.items()}
                ml_outcome = OUTCOME_LABELS[int(np.argmax([ml_probs["home_win"], ml_probs["draw"], ml_probs["away_win"]]))]
                
                home_xg = ml_engine._compute_venue_xg(home_feats, away_feats, "home")
                away_xg = ml_engine._compute_venue_xg(away_feats, home_feats, "away")
            except Exception as e:
                ml_probs = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
                ml_outcome = "Home Win"
                home_xg, away_xg = 1.35, 1.10
                match_weights["ml"] = 0.0  # BUG FIX: was weights["ml"] = 0.0

            # --- Enrichment (disabled until wired up)
            enr_probs   = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            enr_outcome = "\u2014"
            enr_features = {}
            match_weights["enrichment"] = 0.0  # BUG FIX: was weights["enrichment"] = 0.0
                
            # --- Legacy
            try:
                legacy_raw = _run_legacy_engine(cur, fx["home_team_id"], fx["away_team_id"])
                legacy_probs = {k: v for k, v in legacy_raw.items() if k in ("home_win", "draw", "away_win")}
                legacy_outcome = legacy_raw.get("predicted_outcome", "Home Win")
                leg_xg_h = float(legacy_raw.get("home_xg", 1.35))
                leg_xg_a = float(legacy_raw.get("away_xg", 1.10))
            except Exception as e:
                legacy_probs = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
                legacy_outcome = "Home Win"
                leg_xg_h, leg_xg_a = 1.35, 1.10
                match_weights["legacy"] = 0.0  # BUG FIX: was weights["legacy"] = 0.0
                
            # Re-normalize match_weights (already a fresh copy per fixture)
            w_sum = sum(v for k, v in match_weights.items() if k != "enrichment")
            if w_sum < 0.01:
                match_weights = dict(global_weights)
                w_sum = sum(match_weights.values()) or 1.0
            match_weights = {k: round(v / (sum(match_weights.values()) or 1), 4) for k, v in match_weights.items()}

            blended = _blend(dc_probs, ml_probs, enr_probs, legacy_probs, match_weights)
            idx = int(np.argmax([blended["home_win"], blended["draw"], blended["away_win"]]))
            consensus_outcome = OUTCOME_LABELS[idx]
            confidence_score, confidence_label = _confidence_from_entropy(blended)
            agreement = _agreement_level(dc_outcome, ml_outcome, enr_outcome, legacy_outcome)

            # Full market computation
            _w3 = {"dc": match_weights.get("dc", 0.45), "ml": match_weights.get("ml", 0.30), "legacy": match_weights.get("legacy", 0.25)}
            try:
                if _calc is not None:
                    if _recal and _recal.is_fitted:
                        home_xg, away_xg   = _recal.calibrate_xg(home_xg, away_xg, fx.get("league_id"))
                        dc_xg_h, dc_xg_a   = _recal.calibrate_xg(dc_xg_h, dc_xg_a, fx.get("league_id"))
                        leg_xg_h, leg_xg_a = _recal.calibrate_xg(leg_xg_h, leg_xg_a, fx.get("league_id"))
                    dc_mkts  = _ovr(_calc(dc_xg_h,  dc_xg_a),  dc_probs)
                    ml_mkts  = _ovr(_calc(home_xg,  away_xg),  ml_probs)
                    leg_mkts = _ovr(_calc(leg_xg_h, leg_xg_a), legacy_probs)
                    per_mkt_w    = _fetch_pmw(cur, _w3)
                    fx_markets   = _mkt_blend({"dc": dc_mkts, "ml": ml_mkts, "legacy": leg_mkts}, _w3, per_mkt_w)
                    if _recal and _recal.is_fitted:
                        fx_markets = _recal.calibrate(fx_markets, engine="consensus", league_id=fx.get("league_id"))
                    fx_best_bets = _bets(fx_markets, fx["home_name"], fx["away_name"])
                    fx_eng_mkts  = {"dc": dc_mkts, "ml": ml_mkts, "legacy": leg_mkts}
                else:
                    raise RuntimeError("market helpers not loaded")
            except Exception as _me:
                _h, _a = home_xg, away_xg
                _btts = round(max(0.0, min(1 - math.exp(-_h) - math.exp(-_a) + math.exp(-(_h + _a)), 1.0)), 4)
                _o25  = round(max(0.0, min(1 - sum((math.exp(-(_h+_a)) * ((_h+_a)**k) / math.factorial(k)) for k in range(3)), 1.0)), 4)
                fx_markets   = {"btts_yes": _btts, "btts_no": round(1-_btts,4), "over_2_5": _o25, "under_2_5": round(1-_o25,4), "home_xg": round(_h,2), "away_xg": round(_a,2)}
                fx_best_bets = []
                fx_eng_mkts  = {}
                per_mkt_w    = {}
            btts_yes = fx_markets.get("btts_yes", 0.50)
            over_2_5 = fx_markets.get("over_2_5", 0.50)
            
            results.append({
                "fixture_id": fx["id"],
                "match_date": str(fx["match_date"]) if fx["match_date"] else None,
                "gameweek":   fx["gameweek"],
                "match":      match_payload,
                "consensus": {
                    "home_win":          blended["home_win"],
                    "draw":              blended["draw"],
                    "away_win":          blended["away_win"],
                    "predicted_outcome": consensus_outcome,
                    "confidence":        confidence_label,
                    "confidence_score":  confidence_score,
                },
                "engines": {
                    "dc": {"home_win": dc_probs["home_win"], "draw": dc_probs["draw"], "away_win": dc_probs["away_win"], "predicted_outcome": dc_outcome},
                    "ml": {"home_win": ml_probs["home_win"], "draw": ml_probs["draw"], "away_win": ml_probs["away_win"], "predicted_outcome": ml_outcome},
                    "enrichment": {"home_win": enr_probs["home_win"], "draw": enr_probs["draw"], "away_win": enr_probs["away_win"], "predicted_outcome": enr_outcome, "diagnostics": enr_features},
                    "legacy": {"home_win": legacy_probs["home_win"], "draw": legacy_probs["draw"], "away_win": legacy_probs["away_win"], "predicted_outcome": legacy_outcome},
                },
                "weights_used": {**match_weights, "source": weight_source},
                "agreement": agreement,
                "markets":            fx_markets,
                "engine_markets":     fx_eng_mkts,
                "per_market_weights": per_mkt_w,
                "best_bets":          fx_best_bets,
                "_engine_picks": {
                    "dc":         dc_outcome,
                    "ml":         ml_outcome,
                    "enrichment": enr_outcome,
                    "legacy":     legacy_outcome,
                }
            })
            
        return results

    finally:
        conn.close()
