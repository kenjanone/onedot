"""
Market Grader
=============
Grades prediction_markets_log rows when match scores become available.
Called from do_evaluate_predictions() in routes/prediction_log.py after
the existing 1X2 grading block.

Idempotent — skips rows that already have evaluated_at set.
"""

import json
import logging
from typing import Optional

log = logging.getLogger(__name__)


def do_evaluate_market_predictions(conn) -> int:
    """
    Grade all un-evaluated prediction_markets_log rows whose match now has a score.
    Returns number of rows updated.
    """
    import psycopg2.extras
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT
                pml.id,
                pml.consensus_markets,
                pml.engine_predictions,
                m.home_score,
                m.away_score
            FROM prediction_markets_log pml
            JOIN matches m ON m.id = pml.match_id
            WHERE pml.evaluated_at IS NULL
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
        """)
        rows = cur.fetchall()
    except Exception as exc:
        log.debug("do_evaluate_market_predictions: query failed — %s", exc)
        return 0

    updated = 0
    for r in rows:
        try:
            actual_h    = int(r["home_score"])
            actual_a    = int(r["away_score"])
            total       = actual_h + actual_a
            both_scored = actual_h > 0 and actual_a > 0

            cons_mkt   = r["consensus_markets"] or {}
            engine_mkt = r["engine_predictions"] or {}

            # ── Market → actual boolean ──────────────────────────────────────
            actual_values = {
                "home_win":            actual_h > actual_a,
                "draw":                actual_h == actual_a,
                "away_win":            actual_h < actual_a,
                "dc_1x":               actual_h >= actual_a,
                "dc_x2":               actual_a >= actual_h,
                "dc_12":               actual_h != actual_a,
                "btts_yes":            both_scored,
                "btts_no":             not both_scored,
                "over_0_5":            total > 0,
                "over_1_5":            total > 1,
                "over_2_5":            total > 2,
                "over_3_5":            total > 3,
                "over_4_5":            total > 4,
                "under_0_5":           total == 0,
                "under_1_5":           total <= 1,
                "under_2_5":           total <= 2,
                "under_3_5":           total <= 3,
                "under_4_5":           total <= 4,
                "btts_yes_over_2_5":   both_scored and total > 2,
                "btts_yes_under_2_5":  both_scored and total <= 2,
                "btts_no_over_2_5":    (not both_scored) and total > 2,
                "btts_no_under_2_5":   (not both_scored) and total <= 2,
                "home_over_0_5":       actual_h > 0,
                "home_over_1_5":       actual_h > 1,
                "home_over_2_5":       actual_h > 2,
                "home_under_0_5":      actual_h == 0,
                "home_under_1_5":      actual_h <= 1,
                "home_under_2_5":      actual_h <= 2,
                "away_over_0_5":       actual_a > 0,
                "away_over_1_5":       actual_a > 1,
                "away_over_2_5":       actual_a > 2,
                "away_under_0_5":      actual_a == 0,
                "away_under_1_5":      actual_a <= 1,
                "away_under_2_5":      actual_a <= 2,
            }

            grades:       dict = {}
            brier_scores: dict = {}

            for mkt_key, actual_bool in actual_values.items():
                market_grades = {}
                market_brier  = {}
                actual_f      = 1.0 if actual_bool else 0.0

                # Consensus
                cons_prob = _safe_prob(cons_mkt, mkt_key)
                if cons_prob is not None:
                    market_grades["consensus"] = (cons_prob >= 0.5) == actual_bool
                    market_brier["consensus"]  = round((cons_prob - actual_f) ** 2, 4)

                # Per engine
                for eng in ("dc", "ml", "legacy"):
                    eng_prob = _safe_prob(engine_mkt.get(eng, {}), mkt_key)
                    if eng_prob is not None:
                        market_grades[eng] = (eng_prob >= 0.5) == actual_bool
                        market_brier[eng]  = round((eng_prob - actual_f) ** 2, 4)

                if market_grades:
                    grades[mkt_key]       = market_grades
                    brier_scores[mkt_key] = market_brier

            # ── Asian Handicap grading ────────────────────────────────────────
            gd = actual_h - actual_a
            for ah_label, vals in (cons_mkt.get("asian_handicap") or {}).items():
                try:
                    line = float(ah_label.replace("home", "").replace("+", ""))
                    effective = gd + line
                    if abs(effective) < 1e-9:
                        continue  # push — skip grading
                    ah_result = "home" if effective > 0 else "away"
                    for side in ("home", "away"):
                        p = vals.get(side)
                        if p is None:
                            continue
                        p = float(p)
                        actual_side = (ah_result == side)
                        actual_fah  = 1.0 if actual_side else 0.0
                        k = f"ah_{side}_{ah_label}"
                        grades[k]       = {"consensus": (p >= 0.5) == actual_side}
                        brier_scores[k] = {"consensus": round((p - actual_fah) ** 2, 4)}
                except Exception:
                    pass

            # ── Write back ───────────────────────────────────────────────────
            cur.execute("""
                UPDATE prediction_markets_log
                SET evaluated_at      = NOW(),
                    actual_home_goals = %s,
                    actual_away_goals = %s,
                    grades            = %s,
                    brier_scores      = %s
                WHERE id = %s
            """, (
                actual_h, actual_a,
                json.dumps(grades),
                json.dumps(brier_scores),
                r["id"],
            ))
            updated += 1

        except Exception as exc:
            log.debug("market_grader: skip row id=%s — %s", r.get("id"), exc)

    return updated


def _safe_prob(d: object, key: str) -> Optional[float]:
    """Safely extract a float in [0,1] from a dict."""
    if not isinstance(d, dict):
        return None
    v = d.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        return f if 0.0 <= f <= 1.0 else None
    except (TypeError, ValueError):
        return None
