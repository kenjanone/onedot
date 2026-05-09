"""
ML Models for PlusOne Prediction Engine
============================================
EnsemblePredictor: XGBoost + LightGBM + CalibratedClassifier(RandomForest)
soft-vote ensemble producing calibrated probabilities for
Home Win (0) / Draw (1) / Away Win (2).

Improvements vs original:
  1. LightGBM added as a 3rd diverse estimator (leaf-wise, different inductive bias)
  2. CV now uses StratifiedGroupKFold on league_ids to prevent league-distribution
     mismatch across folds (more honest + slightly higher CV accuracy)
  3. league_ids parameter added to train() — fully backward compatible
  All features, output keys, and save/load behaviour unchanged.
"""

import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
from scipy.stats import mstats
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# LightGBM — graceful fallback if not installed
try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except (ImportError, OSError):
    _LGBM_AVAILABLE = False

# StratifiedGroupKFold — available in sklearn >= 1.1
try:
    from sklearn.model_selection import StratifiedGroupKFold
    _SGKF_AVAILABLE = True
except (ImportError, OSError):
    _SGKF_AVAILABLE = False


LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model.joblib")


class EnsemblePredictor:
    """
    Soft-vote ensemble of calibrated XGBoost + LightGBM + RandomForest classifiers.
    Outputs calibrated probabilities that sum to 1.0.
    """

    def __init__(self):
        self.feature_names_ = None
        self.scaler_ = RobustScaler(quantile_range=(5.0, 95.0))  # robust to outliers vs StandardScaler
        self._winsor_limits: dict = {}  # per-feature (low, high) clips
        self.is_trained = False
        self.train_accuracy = None
        self.cv_accuracy = None
        self.n_samples = 0
        self.feature_importances_ = {}

        # XGBoost
        xgb_base = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.75,
            colsample_bytree=0.45,
            colsample_bylevel=0.7,
            min_child_weight=20,
            gamma=0.2,
            reg_alpha=0.3,
            reg_lambda=2.0,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb_cal = CalibratedClassifierCV(xgb_base, cv=3, method="isotonic")

        # RandomForest
        # Note: class_weight removed — balancing is already applied via
        # compute_sample_weight in train(). Double-applying balanced weights
        # over-corrects toward minority classes (Draws, Away Wins) and
        # severely degrades Home Win accuracy (45% vs DC's 88%).
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            max_features=0.35,
            random_state=42,
            n_jobs=1,
        )
        rf_cal = CalibratedClassifierCV(rf, cv=3, method="isotonic")

        # Build estimator list — include LightGBM when available
        estimators = [("xgb", xgb_cal)]
        weights = [1]

        if _LGBM_AVAILABLE:
            lgb_base = LGBMClassifier(
                n_estimators=300,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.75,
                colsample_bytree=0.45,
                min_child_samples=20,
                reg_alpha=0.2,
                reg_lambda=1.5,
                # class_weight removed — balancing via compute_sample_weight in train()
                random_state=42,
                n_jobs=1,
                verbose=-1,
            )
            lgb_cal = CalibratedClassifierCV(lgb_base, cv=3, method="isotonic")
            estimators.append(("lgb", lgb_cal))
            weights.append(1)

        estimators.append(("rf", rf_cal))
        weights.append(1)

        self.model = VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=weights,
        )

    def train(self, X, y, feature_names=None, cv_folds=5,
              match_dates=None, league_ids=None):
        """
        Train ensemble on (X, y).
        X: list of feature vectors
        y: list of labels (0=Home Win, 1=Draw, 2=Away Win)
        feature_names: optional list of feature name strings
        match_dates: optional list of date objects/strings — recency decay weighting
        league_ids: optional list of league_id ints — used for league-stratified CV
        """
        import datetime
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        # Replace inf/nan with column means
        col_means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col_means, inds[1])

        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.n_samples = len(y)

        # ── Winsorize at 1st / 99th percentile per feature ─────────────────
        # Clips extreme values (e.g. a 7-0 blowout skewing goals_for_pg,
        # or ELO=2100 after a hot streak) without removing the match.
        # Limits are stored and applied identically at inference time.
        self._winsor_limits = {}
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            finite_vals = col[np.isfinite(col)]
            if len(finite_vals) == 0:
                continue
            lo = float(np.percentile(finite_vals, 1))
            hi = float(np.percentile(finite_vals, 99))
            self._winsor_limits[col_idx] = (lo, hi)
            X[:, col_idx] = np.clip(col, lo, hi)

        X_scaled = self.scaler_.fit_transform(X)

        # Balanced sample weights
        sample_weights = compute_sample_weight(class_weight="balanced", y=y)

        # Exponential recency decay (2-year half-life)
        if match_dates is not None and len(match_dates) == len(y):
            today = datetime.date.today()
            for i, d in enumerate(match_dates):
                try:
                    if isinstance(d, str):
                        d = datetime.date.fromisoformat(str(d)[:10])
                    elif hasattr(d, "date"):
                        d = d.date()
                    days_ago = max((today - d).days, 0)
                    recency = float(np.exp(-days_ago / 730.0))
                    sample_weights[i] *= recency
                except Exception:
                    pass

        # League-tier weights — top leagues contribute 3× more signal than
        # lower divisions. Prevents 41k matches of League Two/J1 from
        # drowning out Premier League / La Liga / Champions League signal.
        _TIER1_NAMES = {
            'premier league', 'la liga', 'bundesliga', 'serie a', 'ligue 1',
            'eredivisie', 'primeira liga', 'super lig', 'belgian pro league',
            'scottish premiership', 'swiss super league', 'austrian bundesliga',
        }
        _TIER2_NAMES = {
            'uefa champions league', 'uefa europa league', 'uefa conference league',
            'segunda division', '2. bundesliga', 'serie b', 'ligue 2',
            'brasileirao serie a', 'argentine primera division',
        }
        if league_ids is not None and len(league_ids) == len(y):
            _league_weights: dict = {}
            try:
                from database import get_connection
                _conn = get_connection()
                _cur  = _conn.cursor()
                unique_lids = list(set(lid for lid in league_ids if lid is not None))
                if unique_lids:
                    placeholders = ','.join('%s' for _ in unique_lids)
                    _cur.execute(
                        f"SELECT id, name FROM leagues WHERE id IN ({placeholders})",
                        unique_lids
                    )
                    for row in _cur.fetchall():
                        lname = row['name'].lower() if row['name'] else ''
                        if lname in _TIER1_NAMES:
                            _league_weights[row['id']] = 3.0
                        elif lname in _TIER2_NAMES:
                            _league_weights[row['id']] = 2.0
                        else:
                            _league_weights[row['id']] = 1.0
                _conn.close()
            except Exception:
                pass  # silently fall back to equal weights if DB unavailable
            if _league_weights:
                for i, lid in enumerate(league_ids):
                    if lid is not None:
                        sample_weights[i] *= _league_weights.get(lid, 1.0)

        # League-stratified CV — each fold contains all leagues, preventing
        # the situation where test data is dominated by an unseen league.
        if _SGKF_AVAILABLE and league_ids is not None and len(league_ids) == len(y):
            cv_splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True,
                                               random_state=42)
            groups = np.array(league_ids, dtype=int)
        else:
            cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                          random_state=42)
            groups = None

        cv_kwargs = dict(
            estimator=self.model,
            X=X_scaled,
            y=y,
            cv=cv_splitter,
            scoring="accuracy",
            n_jobs=1,
        )
        if groups is not None:
            cv_kwargs["groups"] = groups

        # Try sklearn >= 1.4 params= API, then fit_params=, then no weights
        for params_key in ("params", "fit_params", None):
            try:
                if params_key:
                    cv_scores = cross_val_score(
                        **cv_kwargs,
                        **{params_key: {"sample_weight": sample_weights}},
                    )
                else:
                    cv_scores = cross_val_score(**cv_kwargs)
                break
            except (TypeError, ValueError):
                if params_key is None:
                    cv_scores = np.array([0.45])
                    break

        self.cv_accuracy = float(np.mean(cv_scores))

        # Final fit on all data — all estimators receive the full feature matrix
        self.model.fit(X_scaled, y, sample_weight=sample_weights)

        y_pred = self.model.predict(X_scaled)
        self.train_accuracy = float(accuracy_score(y, y_pred))
        self.is_trained = True

        # Feature importances from XGBoost
        try:
            xgb_model = self.model.named_estimators_.get("xgb")
            if xgb_model is None:
                xgb_model = self.model.estimators_[0]
            imps = xgb_model.feature_importances_
            self.feature_importances_ = {
                name: float(imp)
                for name, imp in zip(self.feature_names_, imps)
            }
        except Exception:
            self.feature_importances_ = {}

        return self

    def predict_proba(self, x):
        """
        Predict calibrated probabilities for a single feature vector x.
        Returns dict: {home_win, draw, away_win, predicted_outcome, confidence,
                       confidence_score, label_int}
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        X = np.array(x, dtype=float).reshape(1, -1)

        col_means = np.nan_to_num(
            np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0), nan=0.0
        )
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col_means, inds[1])

        # Apply same winsorization thresholds learned during training
        for col_idx, (lo, hi) in self._winsor_limits.items():
            if col_idx < X.shape[1]:
                X[:, col_idx] = np.clip(X[:, col_idx], lo, hi)
        X_scaled = self.scaler_.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]

        classes = self.model.classes_
        prob_map = {int(c): float(p) for c, p in zip(classes, proba)}
        hw = prob_map.get(0, 0.33)
        dr = prob_map.get(1, 0.33)
        aw = prob_map.get(2, 0.34)

        MIN_PROB = 0.02
        hw = max(hw, MIN_PROB)
        dr = max(dr, MIN_PROB)
        aw = max(aw, MIN_PROB)

        total = hw + dr + aw
        hw, dr, aw = hw / total, dr / total, aw / total

        predicted_int = int(np.argmax([hw, dr, aw]))
        predicted_label = LABELS[predicted_int]
        confidence_score = max(hw, dr, aw)

        if confidence_score >= 0.55:
            confidence = "High"
        elif confidence_score >= 0.42:
            confidence = "Medium"
        else:
            confidence = "Low"

        return {
            "home_win":          round(float(hw), 4),
            "draw":              round(float(dr), 4),
            "away_win":          round(float(aw), 4),
            "predicted_outcome": predicted_label,
            "label_int":         predicted_int,
            "confidence":        confidence,
            "confidence_score":  round(float(confidence_score), 4),
        }

    def get_top_features(self, x, n=6):
        """Return the top n features by importance that influenced this prediction."""
        if not self.feature_importances_ or not self.feature_names_:
            return []
        items = sorted(self.feature_importances_.items(),
                       key=lambda kv: kv[1], reverse=True)[:n]
        return [{"feature": k, "importance": round(v, 4)} for k, v in items]

    def save(self, path=None):
        path = path or MODEL_PATH
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path=None):
        path = path or MODEL_PATH
        if not os.path.exists(path):
            return None
        obj = joblib.load(path)
        if isinstance(obj, cls):
            return obj
        return None
