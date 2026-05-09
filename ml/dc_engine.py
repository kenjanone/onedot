"""
Dixon-Coles Professional Prediction Engine
==========================================
Adapts the Professional Football Prediction Engine v2.0 to use real
database data from the PlusOne backend instead of synthetic fixtures.

Components:
  1. Dixon-Coles Poisson Model  (MLE + time decay)
  2. Dynamic Elo Rating System  (margin-of-victory weighted)
  3. xG Proxy Model             (rolling xG averages per team)
  4. Monte Carlo Simulation     (50,000 match simulations)
  5. Ensemble Blending          (DC 45% + Elo 30% + xG 25%)
  6. Probability Calibration    (Isotonic regression)

Usage:
  from ml.dc_engine import get_dc_predictor, train_dc_model, predict_dc_match
  train_dc_model(cur)                                     # fit on DB data
  result = predict_dc_match(cur, home_team_id, away_team_id)
"""

import logging
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression

from database import get_connection
from ml.model_store import save_to_db, load_from_db

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

CFG = {
    # Time-decay xi for Dixon-Coles weighting: w = exp(-xi * days_ago).
    # Half-life = ln(2) / xi.  0.003 → half-life ≈ 231 days (~7.5 months).
    # Previously 0.0018 (~385-day half-life, ~13 months) which was too slow
    # to reflect post-transfer-window squad changes.
    "dc_time_decay_xi":     0.003,
    "dc_max_goals":         10,
    "elo_k_base":           40,
    "elo_home_advantage":   60,
    "elo_start":            1500,
    "elo_mov_factor":       0.25,
    "mc_simulations":       50_000,
    "ensemble_weights":     [0.45, 0.30, 0.25],
    "min_kelly_edge":       0.03,
    # How many months of match history to use for DC training.
    # 9 months ≈ 1 full football season.  Override via app_settings key
    # 'dc_lookback_months' (integer, 1–120).  Increasing this gives more
    # data but may dilute the time-decay weighting.
    "dc_lookback_months":   9,
}

HOME_ADVANTAGE = 1.22
LEAGUE_AVG_GOALS = 1.35

# ─── Singleton state ──────────────────────────────────────────────────────────

_dc_predictor = None
_dc_meta: dict = {}

# Shared training-state dict — written by both _auto_train (here) and
# _run_dc_training (routes/markets.py) so the /dc/status endpoint always
# reflects the true state regardless of which code path kicked off training.
dc_training_state: dict = {}


def _get_lookback_months() -> int:
    """
    Read dc_lookback_months from app_settings (allows live changes without
    restarting the server). Falls back to CFG default (9 months).
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(
            "SELECT value FROM app_settings WHERE key = 'dc_lookback_months' LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()
        if row and row["value"]:
            val = int(row["value"])
            return max(1, min(val, 120))   # clamp to 1–120 months
    except Exception:
        pass
    return CFG["dc_lookback_months"]


# ══════════════════════════════════════════════════════════════════════════════
# DIXON-COLES MODEL
# ══════════════════════════════════════════════════════════════════════════════

class DixonColesModel:
    """Dixon & Coles (1997) bivariate Poisson model with time decay."""

    def __init__(self, xi: float = CFG["dc_time_decay_xi"]):
        self.xi = xi
        self.params: dict = {}
        self.teams: list = []
        self.fitted: bool = False

    def _time_weight(self, dates: pd.Series, ref_date: pd.Timestamp) -> np.ndarray:
        days_ago = (ref_date - dates).dt.days.values.astype(float)
        return np.exp(-self.xi * days_ago)

    @staticmethod
    def _tau(hg: int, ag: int, lam_h: float, lam_a: float, rho: float) -> float:
        if   hg == 0 and ag == 0: return 1 - lam_h * lam_a * rho
        elif hg == 0 and ag == 1: return 1 + lam_h * rho
        elif hg == 1 and ag == 0: return 1 + lam_a * rho
        elif hg == 1 and ag == 1: return 1 - rho
        return 1.0

    def _neg_log_likelihood(self, params_vec: np.ndarray,
                             fixtures: pd.DataFrame,
                             weights: np.ndarray) -> float:
        n = len(self.teams)
        alpha = params_vec[:n]
        beta  = params_vec[n:2*n]
        gamma = params_vec[2*n]
        rho   = params_vec[2*n + 1]
        team_idx = {t: i for i, t in enumerate(self.teams)}
        ll = 0.0
        for i, row in fixtures.iterrows():
            hi = team_idx.get(row["home_team"], -1)
            ai = team_idx.get(row["away_team"], -1)
            if hi < 0 or ai < 0:
                continue
            lam_h = np.exp(alpha[hi] - beta[ai] + gamma)
            lam_a = np.exp(alpha[ai] - beta[hi])
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            tau = self._tau(hg, ag, lam_h, lam_a, rho)
            if tau <= 0:
                continue
            ll_m = (poisson.logpmf(hg, lam_h) + poisson.logpmf(ag, lam_a) +
                    math.log(max(tau, 1e-10)))
            ll += weights[i] * ll_m
        return -ll

    def fit(self, fixtures: pd.DataFrame):
        if len(fixtures) < 20:
            log.warning("DC: fewer than 20 fixtures — skipping fit.")
            return self
        ref_date = fixtures["date"].max()
        self.teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
        n = len(self.teams)
        weights = self._time_weight(fixtures["date"], ref_date)
        x0 = np.zeros(2 * n + 2)
        x0[2*n]     =  0.30
        x0[2*n + 1] = -0.10
        bounds = [(-3, 3)] * (2 * n) + [(0, 1)] + [(-0.5, 0.5)]
        try:
            result = minimize(
                self._neg_log_likelihood, x0,
                args=(fixtures, weights),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-7},
            )
            alpha = result.x[:n]
            beta  = result.x[n:2*n]
            gamma = result.x[2*n]
            rho   = result.x[2*n + 1]
        except Exception as e:
            log.error("DC fit failed: %s", e)
            return self
        self.params = {t: {"attack": alpha[i], "defence": beta[i]}
                       for i, t in enumerate(self.teams)}
        self.params["_home_adv"] = gamma
        self.params["_rho"]      = rho

        # ── Per-league home advantage (data-driven, no hardcoding) ──────────
        # After fitting the global gamma, compute a league-level offset by
        # measuring the actual home win rate in each league vs the global rate.
        # offset = log(league_hw_rate / global_hw_rate)
        # This is purely learned from the fixtures data — no fixed constants.
        if "league_id" in fixtures.columns:
            global_hw_rate = (fixtures["home_goals"] > fixtures["away_goals"]).mean()
            global_hw_rate = max(global_hw_rate, 0.30)  # safety floor
            league_adv = {}
            for lid, grp in fixtures.groupby("league_id"):
                if len(grp) < 20:
                    continue
                league_hw = (grp["home_goals"] > grp["away_goals"]).mean()
                league_hw = max(league_hw, 0.15)  # avoid log(0)
                # offset: positive = stronger home advantage than global avg
                league_adv[int(lid)] = float(np.log(league_hw / global_hw_rate))
            self.params["_league_home_adv"] = league_adv
            log.info(
                "DC per-league home adv computed for %d leagues (global_hw=%.1f%%)",
                len(league_adv), global_hw_rate * 100
            )
        else:
            self.params["_league_home_adv"] = {}

        self.fitted = True
        return self

    def score_matrix(self, home: str, away: str,
                     max_goals: int = CFG["dc_max_goals"], **kwargs):
        if not self.fitted or home not in self.params or away not in self.params:
            return None, LEAGUE_AVG_GOALS * HOME_ADVANTAGE, LEAGUE_AVG_GOALS
        hp = self.params[home]
        ap = self.params[away]
        gamma = self.params["_home_adv"]
        rho   = self.params["_rho"]

        # Apply per-league home advantage offset if available.
        # league_id is passed through kwargs; falls back to global gamma if not.
        league_id = kwargs.get("league_id")
        league_adv = self.params.get("_league_home_adv", {})
        if league_id is not None and int(league_id) in league_adv:
            gamma_eff = gamma + league_adv[int(league_id)]
        else:
            gamma_eff = gamma

        lam_h = np.exp(hp["attack"] - ap["defence"] + gamma_eff)
        lam_a = np.exp(ap["attack"] - hp["defence"])
        mat = np.zeros((max_goals + 1, max_goals + 1))
        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                tau = self._tau(hg, ag, lam_h, lam_a, rho)
                mat[hg, ag] = poisson.pmf(hg, lam_h) * poisson.pmf(ag, lam_a) * max(tau, 0)
        s = mat.sum()
        if s > 0:
            mat /= s
        return mat, lam_h, lam_a

    def predict(self, home: str, away: str, league_id=None) -> dict:
        mat, lam_h, lam_a = self.score_matrix(home, away, league_id=league_id)
        if mat is None:
            lam_h = LEAGUE_AVG_GOALS * HOME_ADVANTAGE
            lam_a = LEAGUE_AVG_GOALS
            p_hw = p_d = p_aw = 1/3
        else:
            p_hw = float(np.sum(np.tril(mat, -1)))
            p_d  = float(np.sum(np.diag(mat)))
            p_aw = float(np.sum(np.triu(mat, 1)))
        return {"model": "dixon_coles",
                "home_win": p_hw, "draw": p_d, "away_win": p_aw,
                "exp_home_goals": lam_h, "exp_away_goals": lam_a}


# ══════════════════════════════════════════════════════════════════════════════
# ELO RATING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class EloSystem:
    def __init__(self):
        self.ratings: dict = {}

    def _expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1 + 10 ** ((rb - ra) / 400))

    def _mov_multiplier(self, goal_diff: int, elo_diff: float) -> float:
        gd = abs(goal_diff)
        if gd == 0:
            return 1.0
        return math.log(gd + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))

    def update(self, home: str, away: str, hg: int, ag: int):
        K  = CFG["elo_k_base"]
        ha = CFG["elo_home_advantage"]
        r_h = self.ratings.get(home, CFG["elo_start"])
        r_a = self.ratings.get(away, CFG["elo_start"])
        e_h = self._expected(r_h + ha, r_a)
        gd = hg - ag
        s_h = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        mov = self._mov_multiplier(gd, (r_h + ha) - r_a)
        delta = K * mov * (s_h - e_h)
        self.ratings[home] = r_h + delta
        self.ratings[away] = r_a - delta

    def fit(self, fixtures: pd.DataFrame):
        for _, r in fixtures.sort_values("date").iterrows():
            self.update(r.home_team, r.away_team, r.home_goals, r.away_goals)
        return self

    def predict(self, home: str, away: str) -> dict:
        ha  = CFG["elo_home_advantage"]
        r_h = self.ratings.get(home, CFG["elo_start"])
        r_a = self.ratings.get(away, CFG["elo_start"])
        e_h = self._expected(r_h + ha, r_a)
        e_a = self._expected(r_a, r_h + ha)
        raw_diff = abs(e_h - e_a)
        p_draw = max(0.18, 0.32 - 0.55 * raw_diff)
        p_hw   = e_h * (1 - p_draw)
        p_aw   = e_a * (1 - p_draw)
        total  = p_hw + p_draw + p_aw
        return {"model": "elo",
                "home_win": p_hw / total, "draw": p_draw / total, "away_win": p_aw / total,
                "home_elo": round(r_h, 1), "away_elo": round(r_a, 1),
                "elo_diff": round((r_h + ha) - r_a, 1)}


# ══════════════════════════════════════════════════════════════════════════════
# xG MODEL
# ══════════════════════════════════════════════════════════════════════════════

class XGModel:
    """Rolling xG averages per team from DB xg columns (with goal fallback)."""

    def __init__(self, window: int = 10):
        self.window = window
        self.team_xg_att: dict = {}
        self.team_xg_def: dict = {}
        self.league_xg_avg: float = LEAGUE_AVG_GOALS

    def fit(self, fixtures: pd.DataFrame):
        teams = set(fixtures.home_team) | set(fixtures.away_team)
        for team in teams:
            hm = fixtures[fixtures.home_team == team].head(self.window)
            am = fixtures[fixtures.away_team == team].head(self.window)
            # Use home_xg/away_xg if available, else fall back to goals
            xg_for = (list(hm.get("home_xg", hm.home_goals)) +
                      list(am.get("away_xg", am.away_goals)))
            xg_ag  = (list(hm.get("away_xg", hm.away_goals)) +
                      list(am.get("home_xg", am.home_goals)))
            self.team_xg_att[team] = np.mean(xg_for) if xg_for else self.league_xg_avg
            self.team_xg_def[team] = np.mean(xg_ag)  if xg_ag  else self.league_xg_avg

        # Store per-league xG ratings so cross-league predictions
        # can use competition-specific attack/defence when available.
        # e.g. Aston Villa's Europa League xG differs from their PL xG.
        self._league_xg_att: dict = {}   # (team, league_id) → xg_att
        self._league_xg_def: dict = {}   # (team, league_id) → xg_def
        if "league_id" in fixtures.columns:
            for lid, grp in fixtures.groupby("league_id"):
                for team in set(grp.home_team) | set(grp.away_team):
                    hm = grp[grp.home_team == team].head(self.window)
                    am = grp[grp.away_team == team].head(self.window)
                    if len(hm) + len(am) < 2:
                        continue  # not enough data in this competition
                    xf = list(hm.get("home_xg", hm.home_goals)) +                          list(am.get("away_xg", am.away_goals))
                    xa = list(hm.get("away_xg", hm.away_goals)) +                          list(am.get("home_xg", am.home_goals))
                    if xf: self._league_xg_att[(team, int(lid))] = float(np.mean(xf))
                    if xa: self._league_xg_def[(team, int(lid))] = float(np.mean(xa))

        avg_att = np.mean(list(self.team_xg_att.values())) if self.team_xg_att else self.league_xg_avg
        avg_def = np.mean(list(self.team_xg_def.values())) if self.team_xg_def else self.league_xg_avg
        self.league_xg_avg = (avg_att + avg_def) / 2
        return self

    def predict(self, home: str, away: str, league_id: int = None) -> dict:
        av = self.league_xg_avg or LEAGUE_AVG_GOALS
        # Use competition-specific xG when available (2+ games in that competition)
        # Falls back to all-competition xG automatically.
        lid = int(league_id) if league_id is not None else None
        h_att = (self._league_xg_att.get((home, lid)) if lid else None)                 or self.team_xg_att.get(home, av)
        h_def = (self._league_xg_def.get((home, lid)) if lid else None)                 or self.team_xg_def.get(home, av)
        a_att = (self._league_xg_att.get((away, lid)) if lid else None)                 or self.team_xg_att.get(away, av)
        a_def = (self._league_xg_def.get((away, lid)) if lid else None)                 or self.team_xg_def.get(away, av)
        exp_h = (h_att / av) * (a_def / av) * av * HOME_ADVANTAGE
        exp_a = (a_att / av) * (h_def / av) * av
        max_g = CFG["dc_max_goals"]
        p_hw = p_d = p_aw = 0.0
        for hg in range(max_g + 1):
            for ag in range(max_g + 1):
                p = poisson.pmf(hg, exp_h) * poisson.pmf(ag, exp_a)
                if   hg > ag: p_hw += p
                elif hg == ag: p_d  += p
                else:          p_aw += p
        total = p_hw + p_d + p_aw or 1
        return {"model": "xg",
                "home_win": p_hw / total, "draw": p_d / total, "away_win": p_aw / total,
                "exp_home_xg": round(exp_h, 3), "exp_away_xg": round(exp_a, 3)}


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class MonteCarloEngine:
    def __init__(self, n: int = CFG["mc_simulations"]):
        self.n = n

    def simulate(self, exp_h: float, exp_a: float) -> dict:
        home_g  = np.random.poisson(exp_h, self.n)
        away_g  = np.random.poisson(exp_a, self.n)
        total_g = home_g + away_g
        outcomes = np.where(home_g > away_g, 2, np.where(home_g == away_g, 1, 0))
        from collections import Counter
        score_counts = Counter(zip(home_g.tolist(), away_g.tolist()))
        top_scores = [
            {"score": f"{hg}-{ag}", "probability": round(cnt / self.n, 4)}
            for (hg, ag), cnt in score_counts.most_common(5)
        ]
        return {
            "home_win": float((outcomes == 2).mean()),
            "draw":     float((outcomes == 1).mean()),
            "away_win": float((outcomes == 0).mean()),
            "over_1_5":  float((total_g > 1).mean()),
            "over_2_5":  float((total_g > 2).mean()),
            "over_3_5":  float((total_g > 3).mean()),
            "under_2_5": float((total_g <= 2).mean()),
            "btts_yes":  float(((home_g > 0) & (away_g > 0)).mean()),
            "btts_no":   float(((home_g == 0) | (away_g == 0)).mean()),
            "home_cs":   float((away_g == 0).mean()),
            "away_cs":   float((home_g == 0).mean()),
            "1x":        float((outcomes >= 1).mean()),
            "x2":        float((outcomes <= 1).mean()),
            "12":        float((outcomes != 1).mean()),
            "exp_home_goals": float(home_g.mean()),
            "exp_away_goals": float(away_g.mean()),
            "top_scorelines": top_scores,
        }


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

class ProbabilityCalibrator:
    def __init__(self):
        self.calibrators = {}
        self.fitted = False

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray):
        for cls in range(3):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_probs[:, cls], (labels == cls).astype(float))
            self.calibrators[cls] = iso
        self.fitted = True
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return raw_probs
        cal = np.zeros_like(raw_probs)
        for cls in range(3):
            cal[:, cls] = self.calibrators[cls].predict(raw_probs[:, cls])
        row_sums = np.where(cal.sum(axis=1, keepdims=True) == 0, 1,
                            cal.sum(axis=1, keepdims=True))
        return cal / row_sums


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE BLENDER
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleBlender:
    def __init__(self, weights=None):
        w = weights or CFG["ensemble_weights"]
        self.weights = np.array(w) / sum(w)

    def blend(self, dc: dict, elo: dict, xg: dict) -> dict:
        p_hw = self.weights[0]*dc["home_win"] + self.weights[1]*elo["home_win"] + self.weights[2]*xg["home_win"]
        p_d  = self.weights[0]*dc["draw"]     + self.weights[1]*elo["draw"]     + self.weights[2]*xg["draw"]
        p_aw = self.weights[0]*dc["away_win"] + self.weights[1]*elo["away_win"] + self.weights[2]*xg["away_win"]
        total = p_hw + p_d + p_aw or 1
        return {"home_win": float(p_hw/total), "draw": float(p_d/total), "away_win": float(p_aw/total)}


# ══════════════════════════════════════════════════════════════════════════════
# FULL PREDICTOR (uses team names internally, mapped from IDs externally)
# ══════════════════════════════════════════════════════════════════════════════

class DCPredictor:
    """
    Complete ensemble predictor. Fitted from DB data.
    All internal operations use team name strings.
    External callers pass team_ids; we resolve via self.team_names dict.
    """

    def __init__(self):
        self.dc          = DixonColesModel()
        self.elo         = EloSystem()
        self.xg          = XGModel()
        self.mc          = MonteCarloEngine()
        self.blender     = EnsembleBlender()
        self.calibrator  = ProbabilityCalibrator()
        self.team_names:        dict = {}   # team_id → name
        self.team_match_counts: dict = {}   # team_id → n completed matches seen in DC training
        self.fitted: bool = False
        self.n_matches: int = 0

    # ── Load fixtures from DB ─────────────────────────────────────────────────
    @staticmethod
    def _load_fixtures(cur, months: int = None) -> pd.DataFrame:
        """Load completed matches within the configured lookback window."""
        if months is None:
            months = _get_lookback_months()
        log.info("DCPredictor: loading fixtures with %d-month lookback", months)
        cur.execute("""
            SELECT
                m.id,
                m.match_date                AS date,
                m.home_score                AS home_goals,
                m.away_score                AS away_goals,
                m.home_team_id,
                m.away_team_id,
                m.season_id,
                m.league_id,
                ht.name                     AS home_team,
                at.name                     AS away_team
            FROM matches m
            JOIN teams ht ON ht.id = m.home_team_id
            JOIN teams at ON at.id = m.away_team_id
            WHERE m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
              AND m.match_date >= CURRENT_DATE - (%(months)s || ' months')::INTERVAL
            ORDER BY m.match_date DESC
        """, {"months": months})
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])
        df["date"]       = pd.to_datetime(df["date"])
        df["home_goals"] = df["home_goals"].astype(float)
        df["away_goals"] = df["away_goals"].astype(float)
        return df

    # ── Fit all models ────────────────────────────────────────────────────────
    def fit(self, cur, months: int = None):
        months   = months or _get_lookback_months()
        log.info("DCPredictor: loading fixtures from DB…")
        fixtures = self._load_fixtures(cur, months=months)
        if fixtures.empty or len(fixtures) < 20:
            log.warning("DCPredictor: not enough fixtures (%d). Skipping.", len(fixtures))
            return self

        # Build team_id → name map AND per-team match counts
        from collections import Counter
        _counts: Counter = Counter()
        for _, r in fixtures.iterrows():
            self.team_names[r["home_team_id"]] = r["home_team"]
            self.team_names[r["away_team_id"]] = r["away_team"]
            _counts[r["home_team_id"]] += 1
            _counts[r["away_team_id"]] += 1
        self.team_match_counts = dict(_counts)

        self.n_matches = len(fixtures)
        log.info("DCPredictor: fitting on %d matches…", self.n_matches)

        self.dc.fit(fixtures)
        self.elo.fit(fixtures)
        self.xg.fit(fixtures)

        # Calibration pass
        raw, labels = [], []
        for _, r in fixtures.iterrows():
            try:
                dc_p  = self.dc.predict(r.home_team, r.away_team)
                elo_p = self.elo.predict(r.home_team, r.away_team)
                xg_p  = self.xg.predict(r.home_team, r.away_team)
                bl    = self.blender.blend(dc_p, elo_p, xg_p)
                raw.append([bl["away_win"], bl["draw"], bl["home_win"]])
                outcome = (2 if r.home_goals > r.away_goals else
                           1 if r.home_goals == r.away_goals else 0)
                labels.append(outcome)
            except Exception:
                pass
        if len(raw) > 30:
            self.calibrator.fit(np.array(raw), np.array(labels))

        self.fitted = True
        log.info("DCPredictor: fitted successfully.")
        return self

    # ── Single match prediction ───────────────────────────────────────────────
    def predict(self, home_team_id: int, away_team_id: int, league_id: int = None) -> dict:
        home = self.team_names.get(home_team_id)
        away = self.team_names.get(away_team_id)

        # ── Graceful fallback for unknown team IDs ──────────────────────────────
        # Teams that play in multiple leagues (e.g. "Chelsea eng" in UCL) may
        # have a different ID than their domestic entry. We use their name if we
        # can look it up from the DB, otherwise fall back to league-average params.
        missing = []
        if not home: missing.append(("home", home_team_id))
        if not away: missing.append(("away", away_team_id))

        if missing:
            try:
                conn = get_connection()
                cur  = conn.cursor()
                ids = [tid for _, tid in missing]
                placeholders = ", ".join(["%s"] * len(ids))
                cur.execute(f"SELECT id, name FROM teams WHERE id IN ({placeholders})", ids)
                rows = {r["id"]: r["name"] for r in cur.fetchall()}
                conn.close()
                for side, tid in missing:
                    resolved = rows.get(tid)
                    if side == "home":
                        home = resolved
                    else:
                        away = resolved
            except Exception:
                pass

        # If we still don't have names, inject league-average model params
        if not home:
            home = f"__avg_{home_team_id}__"
            if home not in self.dc.params and self.dc.fitted:
                avg_atk = float(np.mean([v["attack"]  for v in self.dc.params.values() if isinstance(v, dict) and "attack" in v] or [0.0]))
                avg_def = float(np.mean([v["defence"] for v in self.dc.params.values() if isinstance(v, dict) and "defence" in v] or [0.0]))
                self.dc.params[home] = {"attack": avg_atk, "defence": avg_def}
            self.elo.ratings.setdefault(home, CFG["elo_start"])
            self.xg.team_xg_att.setdefault(home, self.xg.league_xg_avg)
            self.xg.team_xg_def.setdefault(home, self.xg.league_xg_avg)
        if not away:
            away = f"__avg_{away_team_id}__"
            if away not in self.dc.params and self.dc.fitted:
                avg_atk = float(np.mean([v["attack"]  for v in self.dc.params.values() if isinstance(v, dict) and "attack" in v] or [0.0]))
                avg_def = float(np.mean([v["defence"] for v in self.dc.params.values() if isinstance(v, dict) and "defence" in v] or [0.0]))
                self.dc.params[away] = {"attack": avg_atk, "defence": avg_def}
            self.elo.ratings.setdefault(away, CFG["elo_start"])
            self.xg.team_xg_att.setdefault(away, self.xg.league_xg_avg)
            self.xg.team_xg_def.setdefault(away, self.xg.league_xg_avg)

        # Store resolved names for future calls
        self.team_names[home_team_id] = home
        self.team_names[away_team_id] = away
        # ────────────────────────────────────────────────────────────────────────

        dc_pred  = self.dc.predict(home, away, league_id=league_id)
        elo_pred = self.elo.predict(home, away)
        xg_pred  = self.xg.predict(home, away, league_id=league_id)
        blended  = self.blender.blend(dc_pred, elo_pred, xg_pred)

        exp_h = (dc_pred["exp_home_goals"] * 0.5 + xg_pred["exp_home_xg"] * 0.5)
        exp_a = (dc_pred["exp_away_goals"] * 0.5 + xg_pred["exp_away_xg"] * 0.5)

        # ── ClubElo confidence-weighted blend ────────────────────────────────────────
        clubelo_probs = _fetch_clubelo_probs(home_team_id, away_team_id)
        clubelo_w_dc  = 1.0          # default: no ClubElo → full DC
        if clubelo_probs:
            # Use the MINIMUM of each team's actual DC training match count.
            # The team with less data is the bottleneck — if either team is unknown
            # to DC (0 matches), DC gets no weight and ClubElo dominates fully.
            # Threshold: 30 matches → DC fully trusted (w_dc = 1.0)
            n_home = self.team_match_counts.get(home_team_id, 0)
            n_away = self.team_match_counts.get(away_team_id, 0)
            n_pair = min(n_home, n_away)   # bottleneck team drives the weight
            clubelo_w_dc = float(min(1.0, n_pair / 30.0))
            w_elo = 1.0 - clubelo_w_dc
            log.debug(
                "DC+ClubElo weights: home_n=%d away_n=%d n_pair=%d w_dc=%.2f",
                n_home, n_away, n_pair, clubelo_w_dc
            )

            # Blend: DC weight towards DC probs, remainder towards ClubElo
            blended["home_win"] = clubelo_w_dc * blended["home_win"] + w_elo * clubelo_probs["home_win"]
            blended["draw"]     = clubelo_w_dc * blended["draw"]     + w_elo * clubelo_probs["draw"]
            blended["away_win"] = clubelo_w_dc * blended["away_win"] + w_elo * clubelo_probs["away_win"]
            # Re-normalise
            _s = blended["home_win"] + blended["draw"] + blended["away_win"] or 1.0
            blended = {k: v / _s for k, v in blended.items()}

            # Also blend expected goals → better MC simulation
            exp_h = clubelo_w_dc * exp_h + w_elo * clubelo_probs["lambda_home"]
            exp_a = clubelo_w_dc * exp_a + w_elo * clubelo_probs["lambda_away"]

            log.debug(
                "DC+ClubElo blend: w_dc=%.2f home_win=%.3f draw=%.3f away_win=%.3f",
                clubelo_w_dc, blended["home_win"], blended["draw"], blended["away_win"]
            )

        raw_arr = np.array([[blended["away_win"], blended["draw"], blended["home_win"]]])
        cal_arr = self.calibrator.calibrate(raw_arr)[0]
        calibrated = {"away_win": float(cal_arr[0]), "draw": float(cal_arr[1]), "home_win": float(cal_arr[2])}

        mc_results = self.mc.simulate(exp_h, exp_a)

        probs = np.array([calibrated["home_win"], calibrated["draw"], calibrated["away_win"]])
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = float(-np.log(1 / 3))
        confidence_score = round((1 - entropy / max_entropy) * 100, 1)

        outcome_label = max(calibrated, key=calibrated.get).replace("_", " ").title()

        return {
            "home_team":  home,
            "away_team":  away,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "models": {
                "dixon_coles": {k: round(v, 4) for k, v in dc_pred.items() if isinstance(v, float)},
                "elo":         {k: round(v, 4) if isinstance(v, float) else v for k, v in elo_pred.items()},
                "xg":          {k: round(v, 4) for k, v in xg_pred.items() if isinstance(v, float)},
                "clubelo":     clubelo_probs or {},
            },
            "blended":    {k: round(v, 4) for k, v in blended.items()},
            "calibrated": {k: round(v, 4) for k, v in calibrated.items()},
            "markets":    mc_results,
            "prediction": outcome_label,
            "confidence": confidence_score,
            "exp_home_goals": round(exp_h, 2),
            "exp_away_goals": round(exp_a, 2),
            "model_info": {
                "n_trained_on":  _dc_meta.get("n_matches", 0),
                "trained_at":    _dc_meta.get("trained_at"),
                "clubelo_w_dc":  round(clubelo_w_dc, 3),
                "clubelo_found": clubelo_probs is not None,
            }
        }

    # ── Elo leaderboard ───────────────────────────────────────────────────────
    def elo_leaderboard(self) -> list:
        return sorted(
            [{"team": t, "elo": round(r, 1), "delta": round(r - CFG["elo_start"], 1)}
             for t, r in self.elo.ratings.items()],
            key=lambda x: -x["elo"],
        )

    # ── Live feedback calibration from prediction_log ──────────────────────────
    def fit_calibrator_from_log(self) -> int:
        """
        Refit the DC ProbabilityCalibrator using real graded outcomes from
        prediction_log (dc_correct column). This is how DC learns from mistakes.
        Returns the number of samples used, or 0 if skipped.
        """
        try:
            conn = get_connection()
            cur  = conn.cursor()
            cur.execute("""
                SELECT home_win_prob, draw_prob, away_win_prob, actual
                FROM prediction_log
                WHERE dc_correct IS NOT NULL
                  AND home_win_prob IS NOT NULL
                  AND draw_prob     IS NOT NULL
                  AND away_win_prob IS NOT NULL
                  AND actual IS NOT NULL
                ORDER BY evaluated_at DESC
                LIMIT 500
            """)
            rows = cur.fetchall()
            conn.close()
        except Exception as exc:
            log.warning("DC fit_calibrator_from_log: DB query failed: %s", exc)
            return 0

        OUTCOME_MAP = {"Home Win": 2, "Draw": 1, "Away Win": 0}
        probs_list, labels_list = [], []
        for r in rows:
            outcome = OUTCOME_MAP.get(str(r["actual"]).strip())
            if outcome is None:
                continue
            hw = float(r["home_win_prob"] or 0)
            dr = float(r["draw_prob"]     or 0)
            aw = float(r["away_win_prob"] or 0)
            total = hw + dr + aw
            if total < 0.01:
                continue
            # calibrator expects [p_aw, p_d, p_hw] order
            probs_list.append([aw / total, dr / total, hw / total])
            labels_list.append(outcome)

        if len(probs_list) < 15:
            log.debug("DC fit_calibrator_from_log: only %d samples (need 15), skipping.",
                      len(probs_list))
            return 0

        probs = np.array(probs_list)
        labels = np.array(labels_list)

        # Sanity check: reject obviously corrupted data where one class dominates
        # >85% of outcomes. Real football data is never this skewed — it means
        # the prediction_log was bulk-imported with fake outcomes (e.g. all
        # "Home Win"). Training a calibrator on such data inverts probabilities.
        for cls in range(3):
            cls_rate = float((labels == cls).mean())
            if cls_rate > 0.85:
                log.warning(
                    "DC fit_calibrator_from_log: class %d = %.1f%% of samples — "
                    "data looks corrupted (bulk import?). Skipping calibration.",
                    cls, cls_rate * 100,
                )
                return 0

        self.calibrator.fit(probs, labels)
        log.info("DC calibrator refitted from %d live prediction outcomes.", len(probs_list))
        return len(probs_list)


# ══════════════════════════════════════════════════════════════════════════════
# CLUBELO INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_clubelo_probs(home_team_id: int, away_team_id: int) -> dict | None:
    """
    Fetch ClubElo score-probability distribution for a specific fixture.

    The team_clubelo table stores TWO rows per fixture (one per team) with the
    same full match JSON in raw_data. Each row's raw_data is:
        {"raw": {"Home": "...", "Away": "...", "Date": "...",
                 "GD<-5": "0.001", "GD=-5": "0.002", ..., "GD=0": "0.29", ..., "GD>5": "0.001",
                 "R:0-0": "0.09", "R:1-0": "0.13", ...}}

    We query by home_team_id (one of the two rows always has team_id = home team)
    and verify the Away team name matches. Falls back to querying by away_team_id
    and checking the Home name if the first query misses.

    Returns:
        {
          "home_win": float, "draw": float, "away_win": float,
          "lambda_home": float, "lambda_away": float,
          "source": "clubelo"
        }
    or None if no ClubElo data found / parse error.
    """
    try:
        conn = get_connection()
        cur  = conn.cursor()

        # --- Resolve team names so we can do name-based matching inside JSONB ---
        ids = [home_team_id, away_team_id]
        cur.execute("SELECT id, name FROM teams WHERE id IN (%s, %s)", ids)
        name_map = {r["id"]: r["name"] for r in cur.fetchall()}
        home_name = name_map.get(home_team_id, "")
        away_name = name_map.get(away_team_id, "")

        raw_row = None

        # Strategy A: query by home team_id, strict match on Home and Away names
        if home_name and away_name:
            cur.execute("""
                SELECT raw_data
                FROM team_clubelo
                WHERE team_id = %s
                  AND elo_date >= CURRENT_DATE - INTERVAL '7 days'
                  AND (
                      (raw_data->'raw'->>'Home' ILIKE %s AND raw_data->'raw'->>'Away' ILIKE %s)
                      OR (raw_data->>'Home' ILIKE %s AND raw_data->>'Away' ILIKE %s)
                  )
                ORDER BY elo_date DESC
                LIMIT 1
            """, (home_team_id, f"%{home_name}%", f"%{away_name}%", f"%{home_name}%", f"%{away_name}%"))
            raw_row = cur.fetchone()

        # Strategy B: query by away team_id, strict match on Home and Away names
        if not raw_row and home_name and away_name:
            cur.execute("""
                SELECT raw_data
                FROM team_clubelo
                WHERE team_id = %s
                  AND elo_date >= CURRENT_DATE - INTERVAL '7 days'
                  AND (
                      (raw_data->'raw'->>'Home' ILIKE %s AND raw_data->'raw'->>'Away' ILIKE %s)
                      OR (raw_data->>'Home' ILIKE %s AND raw_data->>'Away' ILIKE %s)
                  )
                ORDER BY elo_date DESC
                LIMIT 1
            """, (away_team_id, f"%{home_name}%", f"%{away_name}%", f"%{home_name}%", f"%{away_name}%"))
            raw_row = cur.fetchone()

        conn.close()

        if not raw_row:
            return None

        # --- Parse raw_data JSONB --- 
        rd = raw_row["raw_data"]
        if isinstance(rd, str):
            import json as _json
            rd = _json.loads(rd)
        # Handle {"raw": {...}} wrapper
        inner = rd.get("raw", rd)

        def _f(key):
            """Safe float parse — values are stored as strings."""
            v = inner.get(key)
            try:
                return float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        # --- Derive 1X2 from GD fields ---
        p_hw = (_f("GD=1") + _f("GD=2") + _f("GD=3") + _f("GD=4") +
                _f("GD=5") + _f("GD>5"))
        p_d  = _f("GD=0")
        p_aw = (_f("GD=-1") + _f("GD=-2") + _f("GD=-3") + _f("GD=-4") +
                _f("GD=-5") + _f("GD<-5"))
        total = p_hw + p_d + p_aw or 1.0
        p_hw /= total
        p_d  /= total
        p_aw /= total

        # --- Derive implied λ (expected goals) from R:i-j score matrix ---
        lambda_h = 0.0
        lambda_a = 0.0
        for i in range(7):
            for j in range(7):
                p_score = _f(f"R:{i}-{j}")
                lambda_h += i * p_score
                lambda_a += j * p_score
        # Floor at plausible xG values
        lambda_h = max(lambda_h, 0.5)
        lambda_a = max(lambda_a, 0.5)

        if p_hw + p_d + p_aw < 0.5:          # sanity: all near zero → bad data
            return None

        return {
            "home_win":    round(p_hw, 4),
            "draw":        round(p_d,  4),
            "away_win":    round(p_aw, 4),
            "lambda_home": round(lambda_h, 3),
            "lambda_away": round(lambda_a, 3),
            "source":      "clubelo",
        }
    except Exception as exc:
        log.debug("_fetch_clubelo_probs: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON API
# ══════════════════════════════════════════════════════════════════════════════

def get_dc_predictor() -> DCPredictor | None:
    """Return the in-memory singleton. None if not yet trained."""
    return _dc_predictor


def train_dc_model() -> dict:
    """
    Train (or retrain) the DC ensemble from DB data.
    After training, automatically saves the model to Supabase so it
    survives Railway/Render redeploys.
    """
    global _dc_predictor, _dc_meta
    import time
    t0 = time.time()
    conn = get_connection()
    cur  = conn.cursor()
    try:
        predictor = DCPredictor()
        predictor.fit(cur)
        if not predictor.fitted:
            return {"trained": False, "error": "Not enough data in DB (need ≥20 completed matches)."}
        elapsed = round(time.time() - t0, 1)
        _dc_predictor = predictor
        _dc_meta = {
            "n_matches":  predictor.n_matches,
            "trained_at": __import__("datetime").datetime.utcnow().isoformat(),
            "elapsed_s":  elapsed,
        }
        log.info("DC model trained: %d matches, %.1fs", predictor.n_matches, elapsed)
        # Apply live feedback calibration from prediction_log (dc_correct data)
        # This overwrites the in-sample calibrator with real prediction outcomes.
        try:
            n_cal = predictor.fit_calibrator_from_log()
            if n_cal:
                _dc_meta["calibrator_samples"] = n_cal
                log.info("DC live calibration applied on %d real outcomes.", n_cal)
        except Exception as exc:
            log.warning("DC live calibration after train failed: %s", exc)
        # — Persist to Supabase so the model survives redeploys —
        saved = save_to_db(
        predictor,
        name="dc_model",
        n_samples=predictor.n_matches,
        n_features=len(predictor.team_names) * 2,
        train_accuracy=float(_dc_meta.get("log_likelihood", 0) or 0),
        cv_accuracy=float(_dc_meta.get("calibrator_samples", 0) or 0),
    )
        if saved:
            log.info("DC model saved to Supabase ml_models table.")
        else:
            log.warning("DC model could not be saved to Supabase (will retrain on next cold start).")
        return {"trained": True, **_dc_meta}
    except Exception as e:
        log.error("DC training error: %s", e)
        return {"trained": False, "error": str(e)}
    finally:
        conn.close()


def predict_dc_match(home_team_id: int, away_team_id: int, league_id: int = None) -> dict:
    """Run DC prediction for a single match. Returns error dict if untrained."""
    eng = get_dc_predictor()
    if eng is None or not eng.fitted:
        return {"error": "DC model not trained yet. POST /api/dc/train first."}
    return eng.predict(home_team_id, away_team_id, league_id=league_id)


def dc_status() -> dict:
    eng = get_dc_predictor()
    return {
        "dc_model_trained": eng is not None and eng.fitted,
        "n_matches":        _dc_meta.get("n_matches", 0),
        "trained_at":       _dc_meta.get("trained_at"),
        "n_teams":          len(eng.team_names) if eng else 0,
    }


# Auto-load or auto-train on startup (non-blocking daemon thread)
def _auto_train():
    global _dc_predictor, _dc_meta
    import time as _time
    # 1️⃣  Try loading the saved model from Supabase first (fast cold-start)
    try:
        loaded = load_from_db(name="dc_model")
        if loaded is not None and getattr(loaded, "fitted", False):
            _dc_predictor = loaded
            # Reconstruct minimal meta from the loaded model
            _dc_meta = {
                "n_matches":  getattr(loaded, "n_matches", 0),
                "trained_at": "(loaded from Supabase)",
                "elapsed_s":  0,
            }
            dc_training_state.clear()
            dc_training_state.update({"status": "done", "result": {"trained": True, "source": "supabase"}})
            log.info("DC model loaded from Supabase (%d teams).", len(loaded.team_names))
            return
    except Exception as e:
        log.warning("DC Supabase load failed, will train from scratch: %s", e)

    # 2️⃣  No saved model found — train from DB data
    dc_training_state.clear()
    dc_training_state.update({"status": "running", "started_at": _time.time()})
    try:
        result = train_dc_model()
        if result.get("trained"):
            dc_training_state.clear()
            dc_training_state.update({"status": "done", "result": result})
            log.info("DC auto-train complete: %d matches", result.get("n_matches", 0))
        else:
            dc_training_state.clear()
            dc_training_state.update({"status": "error", "error": result.get("error", "unknown")})
            log.warning("DC auto-train failed: %s", result.get("error"))
    except Exception as e:
        dc_training_state.clear()
        dc_training_state.update({"status": "error", "error": str(e)})
        log.warning("DC auto-train exception (will retry on next /train call): %s", e)

import threading
threading.Thread(target=_auto_train, daemon=True).start()
