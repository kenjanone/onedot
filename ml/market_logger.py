"""
Market Logger
=============
Persists consensus market predictions to prediction_markets_log.
Called from _log_prediction_to_db() after each consensus prediction.

Always wrapped in try/except — never propagates errors to the caller.
"""

import json
import logging
from typing import Optional

log = logging.getLogger(__name__)


def log_market_prediction(
    match_id:           Optional[int],
    home_team:          str,
    away_team:          str,
    league:             Optional[str],
    league_id:          Optional[int],
    match_date,
    season_id:          Optional[int],
    consensus_markets:  dict,
    engine_markets:     dict,
    weights_used:       dict,
    per_market_weights: dict,
    best_bets:          list,
) -> None:
    """
    Upsert one market prediction row.
    ON CONFLICT (match_id) updates the probability fields in-place but
    preserves evaluated_at / grades / brier_scores if already graded.

    Silently skips if match_id is None (custom fixtures without a DB record).
    """
    if match_id is None:
        return

    from database import get_connection
    conn = None
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO prediction_markets_log
                (match_id, home_team, away_team, league, league_id,
                 match_date, season_id,
                 consensus_markets, engine_predictions,
                 weights_used, per_market_weights, best_bets)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (match_id) DO UPDATE SET
                consensus_markets  = EXCLUDED.consensus_markets,
                engine_predictions = EXCLUDED.engine_predictions,
                weights_used       = EXCLUDED.weights_used,
                per_market_weights = EXCLUDED.per_market_weights,
                best_bets          = EXCLUDED.best_bets,
                -- preserve grading if already evaluated
                evaluated_at  = COALESCE(prediction_markets_log.evaluated_at, NULL),
                grades        = COALESCE(prediction_markets_log.grades, NULL),
                brier_scores  = COALESCE(prediction_markets_log.brier_scores, NULL),
                actual_home_goals = COALESCE(prediction_markets_log.actual_home_goals, NULL),
                actual_away_goals = COALESCE(prediction_markets_log.actual_away_goals, NULL)
        """, (
            int(match_id),
            str(home_team), str(away_team),
            str(league)   if league    else None,
            int(league_id) if league_id else None,
            match_date,
            int(season_id) if season_id else None,
            json.dumps(consensus_markets),
            json.dumps(engine_markets),
            json.dumps(weights_used),
            json.dumps(per_market_weights),
            json.dumps(best_bets),
        ))
        conn.commit()
    except Exception as exc:
        log.debug("log_market_prediction: match_id=%s — %s", match_id, exc)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
