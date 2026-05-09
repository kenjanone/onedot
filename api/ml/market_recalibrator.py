"""
Market Recalibrator
====================
Per-market isotonic regression calibrators + per-league xG bias correctors.

Trained from prediction_markets_log grading data after each sync.
Applied during consensus market computation to improve all market probabilities.

Calibration details:
  - For each (market_key, engine) pair: fit IsotonicRegression on
    (predicted_prob → actual_bool) from graded prediction_markets_log rows.
  - xG bias: per league, learn mean(actual_total_goals - predicted_total_xg)
    over the last 60 days. Apply with 30% damping so we don't over-correct.

Thread-safe via threading.Lock.
"""

import logging
import threading
from typing import Optional

log = logging.getLogger(__name__)

# Markets to calibrate
TRACKED_MARKETS = [
    "btts_yes", "btts_no",
    "over_1_5", "over_2_5", "over_3_5",
    "under_1_5", "under_2_5", "under_3_5",
    "dc_1x", "dc_x2", "dc_12",
    "btts_yes_over_2_5", "btts_no_under_2_5",
    "home_over_0_5", "home_over_1_5",
    "away_over_0_5", "away_over_1_5",
]

ENGINES      = ["dc", "ml", "legacy", "consensus"]
MIN_SAMPLES  = 20      # minimum graded rows before fitting a calibrator
XG_DAMPING   = 0.30   # how much of the bias to apply (0 = none, 1 = full)
WINDOW_DAYS  = 120     # training window

# O/U pairs that must sum to 1.0 after calibration
_OU_PAIRS = [
    ("over_0_5", "under_0_5"), ("over_1_5", "under_1_5"),
    ("over_2_5", "under_2_5"), ("over_3_5", "under_3_5"),
    ("over_4_5", "under_4_5"),
    ("home_over_0_5", "home_under_0_5"), ("home_over_1_5", "home_under_1_5"),
    ("home_over_2_5", "home_under_2_5"),
    ("away_over_0_5", "away_under_0_5"), ("away_over_1_5", "away_under_1_5"),
    ("away_over_2_5", "away_under_2_5"),
    ("btts_yes", "btts_no"),
    ("dc_1x", None), ("dc_x2", None), ("dc_12", None),  # no pair renorm needed
]


class MarketRecalibrator:
    def __init__(self):
        self._calibrators: dict = {}   # {(market_key, engine): IsotonicRegression}
        self._xg_bias:     dict = {}   # {league_id: {"home": float, "away": float}}
        self._lock               = threading.Lock()
        self.is_fitted           = False
        self.last_fit_result:    dict = {}

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, conn=None) -> dict:
        """
        Train calibrators from prediction_markets_log. Thread-safe.
        Accepts an optional DB connection (for testing); opens one otherwise.
        """
        try:
            from sklearn.isotonic import IsotonicRegression
            import numpy as np
        except ImportError:
            log.warning("scikit-learn not available — market calibrators skipped.")
            return {"fitted": False, "error": "sklearn missing"}

        from database import get_connection
        _conn = conn or get_connection()
        cur   = _conn.cursor()
        fitted_count = 0
        errors = []

        try:
            new_cals: dict = {}
            new_bias: dict = {}

            # ── Per-market, per-engine calibrators ──────────────────────────
            for mkt in TRACKED_MARKETS:
                for eng in ENGINES:
                    try:
                        if eng == "consensus":
                            cur.execute("""
                                SELECT
                                    (consensus_markets->>%(mk)s)::float  AS pred_prob,
                                    (grades->%(mk)s->>'consensus')::boolean AS actual_bool
                                FROM prediction_markets_log
                                WHERE evaluated_at IS NOT NULL
                                  AND consensus_markets ? %(mk)s
                                  AND grades ? %(mk)s
                                  AND evaluated_at >= NOW() - INTERVAL %(win)s
                            """, {"mk": mkt, "win": f"{WINDOW_DAYS} days"})
                        else:
                            cur.execute("""
                                SELECT
                                    (engine_markets->%(eng)s->>%(mk)s)::float AS pred_prob,
                                    (grades->%(mk)s->>%(eng)s)::boolean        AS actual_bool
                                FROM prediction_markets_log
                                WHERE evaluated_at IS NOT NULL
                                  AND engine_markets ? %(eng)s
                                  AND grades ? %(mk)s
                                  AND evaluated_at >= NOW() - INTERVAL %(win)s
                            """, {"mk": mkt, "eng": eng, "win": f"{WINDOW_DAYS} days"})

                        rows = cur.fetchall()
                        valid = [
                            (float(r["pred_prob"]), bool(r["actual_bool"]))
                            for r in rows
                            if r["pred_prob"] is not None and r["actual_bool"] is not None
                        ]
                        if len(valid) < MIN_SAMPLES:
                            continue

                        X = np.array([v[0] for v in valid])
                        y = np.array([float(v[1]) for v in valid])
                        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
                        iso.fit(X, y)
                        new_cals[(mkt, eng)] = iso
                        fitted_count += 1

                    except Exception as exc:
                        errors.append(f"{mkt}/{eng}: {exc}")

            # ── xG bias per league ────────────────────────────────────────────
            try:
                cur.execute("""
                    SELECT
                        league_id,
                        AVG(actual_home_goals - (consensus_markets->>'home_xg')::float) AS home_bias,
                        AVG(actual_away_goals - (consensus_markets->>'away_xg')::float) AS away_bias,
                        COUNT(*)                                                         AS n
                    FROM prediction_markets_log
                    WHERE evaluated_at IS NOT NULL
                      AND actual_home_goals IS NOT NULL
                      AND consensus_markets ? 'home_xg'
                      AND evaluated_at >= NOW() - INTERVAL '60 days'
                    GROUP BY league_id
                    HAVING COUNT(*) >= 8
                """)
                for r in cur.fetchall():
                    if r["league_id"] is not None:
                        new_bias[int(r["league_id"])] = {
                            "home": round(float(r["home_bias"] or 0) * XG_DAMPING, 3),
                            "away": round(float(r["away_bias"] or 0) * XG_DAMPING, 3),
                        }
            except Exception as exc:
                errors.append(f"xG bias: {exc}")

            # ── Atomic swap ───────────────────────────────────────────────────
            with self._lock:
                self._calibrators = new_cals
                self._xg_bias     = new_bias
                self.is_fitted    = fitted_count > 0

            result = {
                "fitted":       self.is_fitted,
                "calibrators":  fitted_count,
                "xg_leagues":   len(new_bias),
                "errors":       errors[:10],
            }
            self.last_fit_result = result
            log.info(
                "MarketRecalibrator: fitted %d calibrators, %d league xG bias entries",
                fitted_count, len(new_bias),
            )
            return result

        except Exception as exc:
            log.error("MarketRecalibrator.fit failed: %s", exc)
            return {"fitted": False, "error": str(exc)}
        finally:
            if conn is None:
                _conn.close()

    # ── xG Bias Corrector ────────────────────────────────────────────────────

    def calibrate_xg(
        self, home_xg: float, away_xg: float, league_id: Optional[int] = None,
    ) -> tuple:
        """Apply xG bias correction for a specific league. Returns (h_xg, a_xg)."""
        if not self.is_fitted or league_id is None:
            return home_xg, away_xg
        with self._lock:
            bias = self._xg_bias.get(int(league_id), {})
        h_adj = round(max(0.20, home_xg + bias.get("home", 0.0)), 2)
        a_adj = round(max(0.20, away_xg + bias.get("away", 0.0)), 2)
        return h_adj, a_adj

    # ── Market Calibration ───────────────────────────────────────────────────

    def calibrate(
        self, markets: dict, engine: str = "consensus", league_id: Optional[int] = None,
    ) -> dict:
        """
        Apply fitted calibrators to a market dict.
        Renormalises O/U pairs (over_X + under_X = 1.0) after calibration.
        """
        if not self.is_fitted:
            return markets

        result = dict(markets)
        with self._lock:
            cals = dict(self._calibrators)

        import numpy as np
        for mkt in TRACKED_MARKETS:
            key = (mkt, engine)
            if key not in cals or mkt not in result:
                continue
            raw = result[mkt]
            if not isinstance(raw, (int, float)):
                continue
            try:
                cal_p = float(cals[key].predict([float(raw)])[0])
                result[mkt] = round(max(0.01, min(0.99, cal_p)), 4)
            except Exception:
                pass

        # Renormalise complementary O/U pairs so they sum to 1.0
        for over_k, under_k in _OU_PAIRS:
            if under_k is None:
                continue
            ov = result.get(over_k)
            un = result.get(under_k)
            if ov is not None and un is not None:
                t = ov + un
                if t > 0:
                    result[over_k]  = round(ov / t, 4)
                    result[under_k] = round(un / t, 4)

        # Renormalise 1X2 triplet
        hw = result.get("home_win", 0.0)
        dr = result.get("draw",     0.0)
        aw = result.get("away_win", 0.0)
        t  = hw + dr + aw
        if t > 0:
            result["home_win"] = round(hw / t, 4)
            result["draw"]     = round(dr / t, 4)
            result["away_win"] = round(aw / t, 4)
            result["dc_1x"]    = round((hw + dr) / t, 4)
            result["dc_x2"]    = round((dr + aw) / t, 4)
            result["dc_12"]    = round((hw + aw) / t, 4)

        return result


# ── Singleton ───────────────────────────────────────────────────────────────────

_instance = MarketRecalibrator()


def get_market_recalibrator() -> MarketRecalibrator:
    return _instance


def recalibrate_markets_from_log() -> dict:
    """Entry point for _auto_recalibrate_bg() in routes/sync.py."""
    return _instance.fit()
