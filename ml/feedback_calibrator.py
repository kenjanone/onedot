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

MIN_SAMPLES  = 50   # minimum evaluated predictions needed before calibrating.
                    # Lowered from 100: with 50 rows the calibrator can meaningfully
                    # correct systematic overconfidence at medium-high probabilities.
                    # Activates within ~5 weeks of a new season instead of ~10 weeks.
MIN_HOLDOUT  = 20   # minimum holdout rows regardless of dataset size.
                    # Lowered from 30 in tandem with MIN_SAMPLES so the split
                    # remains valid (≥20 holdout rows → ±11% standard error).
IMPROVEMENT_TOLERANCE = 0.015  # allow up to 1.5% degradation on holdout before
                                # refusing to apply the calibrator. Absorbs noise
                                # on smallish holdout sets.

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
        self.n_samples   = 0
        self.n_holdout   = 0        # actual holdout size used in last fit
        self.pre_accuracy  = None
        self.post_accuracy = None   # holdout accuracy (out-of-sample)
        self.is_improvement = False  # True when holdout acc >= pre_accuracy - TOLERANCE
        self._last_reason: str = ""  # human-readable reason calibrator is/isn't active

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
            reason = (
                f"Need at least {MIN_SAMPLES} evaluated predictions with probability scores. "
                f"Have {len(probs_list)}. "
                f"Keep syncing match results — calibration activates automatically."
            )
            self._last_reason = reason
            return {
                "success":     False,
                "reason":      reason,
                "n_available": len(probs_list),
                "n_needed":    MIN_SAMPLES,
            }

        probs    = np.array(probs_list)
        outcomes = np.array(outcomes_list)

        # ── Chronological holdout split ───────────────────────────────────────
        # Data is already ORDER BY created_at ASC so [:split] = older predictions,
        # [split:] = most recent predictions.  This is the correct temporal split:
        # calibrate on past, evaluate on future — mirrors real deployment.
        #
        # BUG FIX: old code used int(N * 0.8) which with N=91 gives only 19 holdout
        # rows — one wrong prediction = 5.3% accuracy swing (pure noise).
        # We now guarantee MIN_HOLDOUT rows in the holdout regardless of N.
        holdout_size = max(MIN_HOLDOUT, int(len(probs) * 0.2))
        split        = len(probs) - holdout_size

        probs_train,    probs_holdout    = probs[:split],    probs[split:]
        outcomes_train, outcomes_holdout = outcomes[:split], outcomes[split:]
        self.n_holdout = len(probs_holdout)

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

        # BUG FIX: strict >= fails when post == pre - epsilon due to float rounding
        # or tiny holdouts. Allow IMPROVEMENT_TOLERANCE degradation before rejecting.
        self.is_improvement = self.post_accuracy >= (self.pre_accuracy - IMPROVEMENT_TOLERANCE)

        # ── Refit on ALL data for production use ──────────────────────────────
        # Now that we've validated it helps on holdout, fit the final calibrator
        # on the full dataset for maximum coverage.
        cal = _IsotonicCalibrator()
        cal.fit(probs, outcomes)

        self._cal      = cal
        self.n_samples = len(probs)

        if self.is_improvement:
            self._last_reason = (
                f"Calibrator active — holdout {self.post_accuracy*100:.1f}% vs raw "
                f"{self.pre_accuracy*100:.1f}% (n_holdout={self.n_holdout}, "
                f"tolerance={IMPROVEMENT_TOLERANCE*100:.1f}%)"
            )
        else:
            self._last_reason = (
                f"Calibrator fitted but NOT applied — holdout {self.post_accuracy*100:.1f}% "
                f"vs raw {self.pre_accuracy*100:.1f}% (n_holdout={self.n_holdout}, "
                f"tolerance={IMPROVEMENT_TOLERANCE*100:.1f}%, "
                f"gap={((self.post_accuracy-self.pre_accuracy)*100):.1f}%). "
                f"Collect more graded predictions and recalibrate."
            )

        log.info(
            "FeedbackCalibrator fitted on %d samples (holdout=%d) — "
            "raw %.1f%% → calibrated %.1f%% (%s)",
            self.n_samples, self.n_holdout,
            self.pre_accuracy  * 100,
            self.post_accuracy * 100,
            "ACTIVE" if self.is_improvement else "NOT APPLIED",
        )

        return {
            "success":            True,
            "n_samples":          self.n_samples,
            "n_holdout":          self.n_holdout,
            "n_train":            split,
            "pre_accuracy_pct":   round(self.pre_accuracy  * 100, 2),
            "post_accuracy_pct":  round(self.post_accuracy * 100, 2),
            "improvement_pct":    round((self.post_accuracy - self.pre_accuracy) * 100, 2),
            "tolerance_pct":      round(IMPROVEMENT_TOLERANCE * 100, 2),
            "is_improvement":     self.is_improvement,
            "will_apply":         self.is_improvement,
            "note":               self._last_reason,
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

    def diagnose(self) -> dict:
        """
        Return a human-readable diagnostic dict explaining the calibrator's
        current state — useful for debugging via the API without needing DB access.
        """
        return {
            "is_fitted":           self.is_fitted,
            "is_improvement":      self.is_improvement,
            "n_samples":           self.n_samples,
            "n_holdout":           self.n_holdout,
            "pre_accuracy_pct":    round(self.pre_accuracy  * 100, 2) if self.pre_accuracy  is not None else None,
            "post_accuracy_pct":   round(self.post_accuracy * 100, 2) if self.post_accuracy is not None else None,
            "improvement_pct":     round((self.post_accuracy - self.pre_accuracy) * 100, 2)
                                   if self.pre_accuracy is not None and self.post_accuracy is not None else None,
            "tolerance_pct":       round(IMPROVEMENT_TOLERANCE * 100, 2),
            "min_samples_needed":  MIN_SAMPLES,
            "min_holdout_size":    MIN_HOLDOUT,
            "reason":              self._last_reason or "Calibrator not yet loaded or fitted.",
        }

    def save(self):
        """Persist calibrator to ml_models table in the DB."""
        if not self._cal or not self._cal.is_fitted:
            return
        try:
            from database import get_connection
            # Pickle a full state dict so that load() can restore n_holdout and
            # pre_accuracy — previously only the calibrator object was pickled,
            # causing those fields to always come back as 0/None after a restart.
            state = {
                "cal":           self._cal,
                "n_holdout":     self.n_holdout,
                "pre_accuracy":  self.pre_accuracy,
            }
            byte_data = pickle.dumps(state)
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
            log.info("FeedbackCalibrator saved to DB (improvement=%s, n_holdout=%d).",
                     self.is_improvement, self.n_holdout)
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

            unpickled = pickle.loads(row["model_bytes"])

            # Backwards-compatible unpickling: new format is a state dict
            # {cal, n_holdout, pre_accuracy}; old format is the raw calibrator object.
            if isinstance(unpickled, dict):
                self._cal         = unpickled.get("cal")
                self.n_holdout    = int(unpickled.get("n_holdout") or 0)
                self.pre_accuracy = unpickled.get("pre_accuracy")  # may be None for old rows
            else:
                # Old pickle format — just the _IsotonicCalibrator object
                self._cal         = unpickled
                self.n_holdout    = 0     # unknown from old format
                self.pre_accuracy = None  # unknown from old format

            self.n_samples      = int(row["n_samples"] or 0)
            stored_acc          = float(row["cv_accuracy"] or 0)
            # Negative stored value means calibrator was saved but did not improve accuracy
            self.is_improvement = stored_acc >= 0
            self.post_accuracy  = abs(stored_acc)
            self._last_reason   = (
                f"Loaded from DB — n_samples={self.n_samples}, "
                f"holdout_acc={self.post_accuracy*100:.1f}%, "
                f"active={self.is_improvement}"
            )
            log.info("FeedbackCalibrator loaded from DB (n=%d, acc=%.1f%%, holdout=%d, improvement=%s)",
                     self.n_samples, self.post_accuracy * 100, self.n_holdout, self.is_improvement)
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
    global _calibrator
    cal = FeedbackCalibrator()
    result = cal.fit_from_db()
    if result.get("success"):
        cal.save()
        # BUG FIX: previously called reset_calibrator() here, which cleared the
        # singleton and forced a DB reload on the next get_calibrator() call.
        # The DB reload lost n_holdout and pre_accuracy (save() didn't persist
        # them in the old format), so calibrator-debug showed nulls immediately
        # after a successful recalibrate.
        #
        # Fix: assign the freshly fitted in-memory calibrator directly as the
        # singleton.  It already has all fields populated correctly.  The DB
        # state (now including n_holdout + pre_accuracy in the state dict) will
        # be correct on the NEXT process restart too.
        _calibrator = cal
        log.info(
            "recalibrate_with_feedback: %d samples, holdout=%d, %.1f%% → %.1f%%",
            cal.n_samples,
            cal.n_holdout,
            result["pre_accuracy_pct"],
            result["post_accuracy_pct"],
        )
    else:
        log.debug("recalibrate_with_feedback skipped: %s", result.get("reason", "unknown"))
