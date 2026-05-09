"""
Feedback Calibrator
====================
Learns from prediction_log evaluated outcomes to correct the probability
outputs of the ML engine.

Problem it solves:
  The XGBoost + RandomForest ensemble is trained on historical match features
  but never sees whether its live predictions were right or wrong. Over time
  it develops systematic biases — e.g. overconfident on Home Win, underconfident
  on Draw. This module fits an isotonic regression calibrator on top of the
  model's raw probability outputs using real evaluated predictions from
  prediction_log, then applies it to every future prediction.

How it works:
  1. Load prediction_log rows where actual IS NOT NULL (graded predictions).
  2. Build (N, 3) probability matrix [p_aw, p_d, p_hw] and (N,) outcome labels.
  3. Fit sklearn's CalibratedClassifierCV (isotonic) as a pass-through calibrator.
     Because we don't have access to raw features at calibration time, we fit
     a direct probability-to-probability mapping using VennAbersCalibrator
     (per-class isotonic regression).
  4. Save the fitted calibrators to the DB (ml_models table, key='calibrator').
  5. At prediction time, load the calibrator and transform the raw probabilities.

Minimum data requirement: 15 evaluated predictions.
Below this threshold the calibrator is not applied (raw probs used instead).

Usage:
  from ml.feedback_calibrator import FeedbackCalibrator
  cal = FeedbackCalibrator()
  cal.fit_from_db()           # train from prediction_log
  cal.save()                  # persist to DB
  adjusted = cal.apply(probs) # dict with home_win, draw, away_win
"""

import logging
import pickle
import psycopg2
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)

MIN_SAMPLES = 50  # minimum evaluated predictions needed before calibrating
                  # isotonic regression overfits badly on fewer than ~50 samples;
                  # the holdout accuracy guard below prevents bad fits anyway

OUTCOME_MAP = {
    "Home Win": 2, "Away Win": 0, "Draw": 1,
}


class _IsotonicCalibrator:
    """
    Per-class isotonic regression calibrator.
    Fits one isotonic regressor per outcome class on the predicted probability
    vs actual outcome (0/1 binary) pairs.
    """
    def __init__(self):
        self.calibrators = {}  # {class_idx: IsotonicRegression}
        self.is_fitted = False

    def fit(self, probs: np.ndarray, outcomes: np.ndarray):
        from sklearn.isotonic import IsotonicRegression
        n_classes = probs.shape[1]
        for k in range(n_classes):
            binary_labels = (outcomes == k).astype(float)
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs[:, k], binary_labels)
            self.calibrators[k] = ir
        self.is_fitted = True

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return probs
        n_classes = probs.shape[1]
        calibrated = np.zeros_like(probs)
        for k in range(n_classes):
            if k in self.calibrators:
                calibrated[:, k] = self.calibrators[k].predict(probs[:, k])
            else:
                calibrated[:, k] = probs[:, k]
        # Apply a 2% probability floor so isotonic regression can never collapse
        # an outcome to exactly 0% (which breaks UI display and log-loss)
        calibrated = np.maximum(calibrated, 0.02)
        # Re-normalise so probabilities sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
        return calibrated / row_sums


class FeedbackCalibrator:
    """
    High-level calibrator that loads data from prediction_log,
    fits per-class isotonic regression, and persists to the DB.
    """
    DB_KEY = "feedback_calibrator_v1"

    def __init__(self):
        self._cal: Optional[_IsotonicCalibrator] = None
        self.n_samples = 0
        self.pre_accuracy  = None
        self.post_accuracy = None   # holdout accuracy (out-of-sample)
        self.is_improvement = False  # True only when holdout acc >= pre_accuracy

    @property
    def is_fitted(self) -> bool:
        return self._cal is not None and self._cal.is_fitted and self.is_improvement

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit_from_db(self) -> dict:
        """
        Load evaluated predictions from prediction_log and fit the calibrator.
        Returns a summary dict.
        """
        from database import get_connection
        conn = get_connection()
        cur  = conn.cursor()
        try:
            cur.execute("""
                SELECT home_win_prob, draw_prob, away_win_prob, actual
                FROM prediction_log
                WHERE actual IS NOT NULL
                  AND home_win_prob IS NOT NULL
                  AND draw_prob     IS NOT NULL
                  AND away_win_prob IS NOT NULL
                ORDER BY created_at ASC
            """)
            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return {"success": False, "reason": "No evaluated predictions in prediction_log."}

        # Build arrays — probs in [p_aw, p_d, p_hw] order to match MetricsEngine
        probs_list, outcomes_list = [], []
        for r in rows:
            outcome_int = OUTCOME_MAP.get(r["actual"])
            if outcome_int is None:
                continue
            p_hw = float(r["home_win_prob"] or 0)
            p_d  = float(r["draw_prob"]     or 0)
            p_aw = float(r["away_win_prob"] or 0)
            total = p_hw + p_d + p_aw
            if total < 0.01:
                continue
            probs_list.append([p_aw / total, p_d / total, p_hw / total])
            outcomes_list.append(outcome_int)

        if len(probs_list) < MIN_SAMPLES:
            return {
                "success": False,
                "reason":  f"Need at least {MIN_SAMPLES} evaluated predictions. Have {len(probs_list)}.",
                "n_available": len(probs_list),
            }

        probs    = np.array(probs_list)
        outcomes = np.array(outcomes_list)

        # ── Chronological 80/20 holdout split ────────────────────────────────
        # Data is already ORDER BY created_at ASC so [:split] = older predictions,
        # [split:] = most recent predictions.  This is the correct temporal split:
        # calibrate on past, evaluate on future — mirrors real deployment.
        split = max(10, int(len(probs) * 0.8))
        probs_train,    probs_holdout    = probs[:split],    probs[split:]
        outcomes_train, outcomes_holdout = outcomes[:split], outcomes[split:]

        # Pre-calibration accuracy measured on the SAME holdout slice so the
        # pre vs post comparison is apples-to-apples.  Measuring pre_accuracy
        # on the full dataset while post_accuracy is on holdout only would
        # systematically flatter the raw model and under-state calibration gain.
        pre_preds_holdout  = np.argmax(probs_holdout, axis=1)
        self.pre_accuracy  = float((pre_preds_holdout == outcomes_holdout).mean())

        cal_eval = _IsotonicCalibrator()
        cal_eval.fit(probs_train, outcomes_train)
        calibrated_holdout = cal_eval.predict(probs_holdout)
        post_preds_holdout = np.argmax(calibrated_holdout, axis=1)
        self.post_accuracy  = float((post_preds_holdout == outcomes_holdout).mean())
        self.is_improvement = self.post_accuracy >= self.pre_accuracy

        # ── Refit on ALL data for production use ──────────────────────────────
        # Now that we've validated it helps on holdout, fit the final calibrator
        # on the full dataset for maximum coverage.
        cal = _IsotonicCalibrator()
        cal.fit(probs, outcomes)

        self._cal = cal
        self.n_samples = len(probs)

        log.info(
            "FeedbackCalibrator fitted on %d samples — pre %.1f%% → holdout %.1f%% (%s)",
            self.n_samples,
            self.pre_accuracy  * 100,
            self.post_accuracy * 100,
            "IMPROVEMENT" if self.is_improvement else "NO IMPROVEMENT — will not apply",
        )

        return {
            "success":           True,
            "n_samples":         self.n_samples,
            "pre_accuracy_pct":  round(self.pre_accuracy  * 100, 2),
            "post_accuracy_pct": round(self.post_accuracy * 100, 2),
            "improvement_pct":   round((self.post_accuracy - self.pre_accuracy) * 100, 2),
            "is_improvement":    self.is_improvement,
            "note": ("Calibrator will be applied to future predictions."
                     if self.is_improvement else
                     "Calibrator fitted but will NOT be applied — holdout accuracy "
                     "did not improve over raw model. Recalibrate when more data is available."),
        }

    # ── Apply ─────────────────────────────────────────────────────────────────

    def apply(self, probs_dict: dict) -> dict:
        """
        Given a probs dict {home_win, draw, away_win} from the ML model,
        return a calibrated version.  Falls through unchanged if not fitted.
        """
        if not self.is_fitted:
            return probs_dict

        p_hw = float(probs_dict.get("home_win", 0))
        p_d  = float(probs_dict.get("draw",     0))
        p_aw = float(probs_dict.get("away_win", 0))

        arr = np.array([[p_aw, p_d, p_hw]])   # [p_aw, p_d, p_hw]
        cal = self._cal.predict(arr)[0]        # [p_aw_cal, p_d_cal, p_hw_cal]

        return {
            "home_win": round(float(cal[2]), 4),
            "draw":     round(float(cal[1]), 4),
            "away_win": round(float(cal[0]), 4),
        }

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self):
        """Persist calibrator to ml_models table in the DB."""
        if not self._cal or not self._cal.is_fitted:
            return
        try:
            from database import get_connection
            byte_data = pickle.dumps(self._cal)
            conn = get_connection()
            cur  = conn.cursor()
            # Store post_accuracy as positive when improvement, negative when not.
            # This lets load() reconstruct is_improvement without a schema change.
            stored_acc = self.post_accuracy if self.is_improvement else -(self.post_accuracy or 0)
            cur.execute("""
                INSERT INTO ml_models (name, model_bytes, n_samples, cv_accuracy, created_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (name) DO UPDATE
                  SET model_bytes = EXCLUDED.model_bytes,
                      n_samples   = EXCLUDED.n_samples,
                      cv_accuracy = EXCLUDED.cv_accuracy,
                      created_at  = NOW()
            """, (self.DB_KEY, byte_data, self.n_samples, stored_acc))
            conn.commit()
            conn.close()
            log.info("FeedbackCalibrator saved to DB (improvement=%s).", self.is_improvement)
        except Exception as exc:
            log.warning("FeedbackCalibrator.save() failed: %s", exc)

    def load(self) -> bool:
        """Load calibrator from DB. Returns True if successful."""
        try:
            from database import get_connection
            conn = get_connection()
            cur  = conn.cursor()
            cur.execute(
                "SELECT model_bytes, n_samples, cv_accuracy FROM ml_models WHERE name = %s",
                (self.DB_KEY,)
            )
            row = cur.fetchone()
            conn.close()
            if not row or not row["model_bytes"]:
                return False
            self._cal          = pickle.loads(row["model_bytes"])
            self.n_samples     = int(row["n_samples"] or 0)
            stored_acc         = float(row["cv_accuracy"] or 0)
            # Negative stored value means calibrator was saved but did not improve accuracy
            self.is_improvement = stored_acc >= 0
            self.post_accuracy  = abs(stored_acc)
            log.info("FeedbackCalibrator loaded from DB (n=%d, acc=%.1f%%, improvement=%s)",
                     self.n_samples, self.post_accuracy * 100, self.is_improvement)
            return True
        except Exception as exc:
            # Silently skip on fresh deployments where ml_models table doesn't exist yet
            err = str(exc)
            if "ml_models" not in err and "does not exist" not in err:
                log.warning("FeedbackCalibrator.load() failed: %s", exc)
            return False


# ── Module-level singleton ────────────────────────────────────────────────────

_calibrator: Optional[FeedbackCalibrator] = None


def get_calibrator() -> FeedbackCalibrator:
    global _calibrator
    local_cal = _calibrator
    if local_cal is None:
        local_cal = FeedbackCalibrator()
        local_cal.load()   # try loading from DB on first access
        _calibrator = local_cal
    return local_cal


def reset_calibrator():
    """Force reload on next get_calibrator() call (called after recalibration)."""
    global _calibrator
    _calibrator = None


def recalibrate_with_feedback():
    """
    Convenience function imported by prediction_engine.predict_upcoming_fast().
    Fits the feedback calibrator from prediction_log and reloads the singleton.
    Silently skips if there aren't enough evaluated predictions yet.
    """
    cal = FeedbackCalibrator()
    result = cal.fit_from_db()
    if result.get("success"):
        cal.save()
        reset_calibrator()  # force singleton reload on next prediction
        log.info(
            "recalibrate_with_feedback: %d samples, %.1f%% → %.1f%%",
            cal.n_samples,
            result["pre_accuracy_pct"],
            result["post_accuracy_pct"],
        )
    else:
        log.debug("recalibrate_with_feedback skipped: %s", result.get("reason", "unknown"))
