"""
PlusOne Cache Warmer
======================
Precomputes upcoming consensus predictions in the background so the first
user request after a sync or retrain is always a cache hit.

Called by JobQueue at Priority.WARM_CACHE (P3) — runs after:
  1. sync_all completes with new rows
  2. RETRAIN_ML completes successfully
  3. RECALIBRATE_DC completes
  4. App startup (via FastAPI on_event)

Complexity:
  warm_upcoming_cache() runs the same prediction pipeline as the route,
  so its complexity is identical. The gain is that the O(n) computation
  happens once in the background rather than on every user request.
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)


def warm_upcoming_cache(league_id: Optional[int] = None, limit: int = 30) -> dict:
    """
    Compute upcoming consensus predictions and store in cache.
    Mirrors the formatting logic of GET /api/markets/upcoming.
    Returns {warmed: int} on success, {error: str} on failure.
    """
    try:
        from ml.cache import _cache
        from ml.consensus_engine import upcoming_consensus_fast

        log.info("CacheWarmer: computing predictions (league=%s, limit=%d)…",
                 league_id, limit)

        raw_results = upcoming_consensus_fast(league_id, limit)
        results = _format_results(raw_results)
        response = {
            "count":       len(results),
            "predictions": results,
            "engine":      "consensus",
            "cached":      True,
        }
        _cache.set_upcoming(league_id, limit, response)
        log.info("CacheWarmer: stored %d predictions in cache.", len(results))
        return {"warmed": len(results)}

    except Exception as exc:
        log.warning("CacheWarmer failed: %s", exc)
        return {"error": str(exc)}


def _format_results(raw_results: list) -> list:
    """
    Formats raw upcoming_consensus_fast() output into the API response shape.
    Extracted here (rather than repeating it in routes/markets.py) so the
    warmer and the route produce byte-for-byte identical responses.
    """
    results = []
    for pred in raw_results:
        try:
            fx        = pred["match"]
            fx_id     = pred["fixture_id"]
            consensus = pred["consensus"]
            engines   = pred["engines"]
            markets   = pred.get("markets", {})

            hw   = consensus.get("home_win", 0.33)
            dr   = consensus.get("draw",     0.33)
            aw   = consensus.get("away_win", 0.34)
            lead = max(hw, dr, aw)
            outcome    = consensus.get("predicted_outcome", "Home Win")
            confidence = consensus.get("confidence", "Medium")

            xg_h   = float(markets.get("home_xg", 0))
            xg_a   = float(markets.get("away_xg", 0))
            pred_h = max(0, round(xg_h))
            pred_a = max(0, round(xg_a))

            weight_map = pred.get("weights_used", {})
            valid_eng  = {k: v for k, v in weight_map.items()
                          if k in ("dc", "ml", "legacy", "enrichment")}
            champ_name = max(valid_eng, key=valid_eng.get) if valid_eng else "ml"
            champ_probs = engines.get(champ_name, {
                "home_win": hw, "draw": dr, "away_win": aw
            })

            results.append({
                "predicted_outcome": outcome,
                "confidence":        confidence,
                "confidence_score":  round(lead, 4),
                "probabilities": {
                    "home_win": round(hw, 4),
                    "draw":     round(dr, 4),
                    "away_win": round(aw, 4),
                },
                "expected_goals": {
                    "home_xg":         round(xg_h, 2),
                    "away_xg":         round(xg_a, 2),
                    "predicted_score": f"{pred_h}-{pred_a}",
                },
                "match": {
                    "match_id":      fx_id,
                    "home_team":     fx["home_team"],
                    "away_team":     fx["away_team"],
                    "home_team_id":  fx["home_team_id"],
                    "away_team_id":  fx["away_team_id"],
                    "home_logo":     fx.get("home_logo"),
                    "away_logo":     fx.get("away_logo"),
                    "league":        fx.get("league", ""),
                    "league_id":     fx.get("league_id", 0),
                    "date":          pred["match_date"],
                    "gameweek":      pred["gameweek"],
                    "season":        fx.get("season", ""),
                },
                "model_breakdown": engines,
                "weights":         weight_map,
                "engine":          "consensus",
            })
        except Exception as exc:
            log.debug("CacheWarmer skip fixture: %s", exc)
            continue
    return results
