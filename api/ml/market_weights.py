"""
Per-Market Dynamic Weight Computer
====================================
Queries prediction_markets_log.grades JSONB to compute per-engine accuracy
for each tracked market type, then returns market-specific weight dicts.

If < MIN_GRADED rows exist for a market, falls back to global engine weights.
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)

# Markets for which we learn per-market weights
TRACKED_MARKETS = [
    "btts_yes", "btts_no",
    "over_1_5", "over_2_5", "under_2_5", "under_3_5",
    "dc_1x", "dc_x2", "dc_12",
    "btts_yes_over_2_5",
    "home_over_1_5", "away_over_1_5",
]

MIN_GRADED = 15        # rows needed before trusting historical weights
WINDOW_DAYS = 90       # trailing window for accuracy computation


def fetch_per_market_weights(cur, global_weights: dict) -> dict:
    """
    For each tracked market, query prediction_markets_log to derive which
    engine is historically most accurate, then return normalised per-market
    weight dicts.

    Returns:
        {
            "btts_yes": {"dc": 0.48, "ml": 0.36, "legacy": 0.16},
            "over_2_5": {"dc": 0.41, "ml": 0.42, "legacy": 0.17},
            ...
        }
    Falls back to the caller's global_weights dict for any market that lacks
    enough history (< MIN_GRADED graded rows).
    """
    fallback = {
        "dc":     global_weights.get("dc",     0.45),
        "ml":     global_weights.get("ml",     0.30),
        "legacy": global_weights.get("legacy", 0.25),
    }
    result: dict = {}

    for mkt in TRACKED_MARKETS:
        try:
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE grades ? %(mk)s)                        AS n,
                    SUM((grades->%(mk)s->>'dc')::boolean::int)::float              AS dc_correct,
                    SUM((grades->%(mk)s->>'ml')::boolean::int)::float              AS ml_correct,
                    SUM((grades->%(mk)s->>'legacy')::boolean::int)::float          AS leg_correct
                FROM prediction_markets_log
                WHERE evaluated_at IS NOT NULL
                  AND grades ? %(mk)s
                  AND evaluated_at >= NOW() - INTERVAL %(win)s
                """,
                {"mk": mkt, "win": f"{WINDOW_DAYS} days"},
            )
            row = cur.fetchone()
            n = int(row["n"] or 0) if row else 0

            if n < MIN_GRADED:
                result[mkt] = dict(fallback)
                continue

            dc_acc  = float(row["dc_correct"]  or 0) / n
            ml_acc  = float(row["ml_correct"]  or 0) / n
            leg_acc = float(row["leg_correct"] or 0) / n

            # Floor: never fully silence any engine
            MIN_FLOOR = 0.05
            dc_acc  = max(dc_acc,  MIN_FLOOR)
            ml_acc  = max(ml_acc,  MIN_FLOOR)
            leg_acc = max(leg_acc, MIN_FLOOR)

            wsum = dc_acc + ml_acc + leg_acc
            result[mkt] = {
                "dc":     round(dc_acc  / wsum, 4),
                "ml":     round(ml_acc  / wsum, 4),
                "legacy": round(leg_acc / wsum, 4),
            }
            log.debug(
                "Per-market weights [%s, n=%d]: DC=%.1f%% ML=%.1f%% Leg=%.1f%%",
                mkt, n, result[mkt]["dc"] * 100, result[mkt]["ml"] * 100, result[mkt]["legacy"] * 100,
            )

        except Exception as exc:
            log.debug("fetch_per_market_weights(%s): %s", mkt, exc)
            result[mkt] = dict(fallback)

    return result
