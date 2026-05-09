"""
Shared Market Calculator
========================
Computes a full market sheet from expected goals using analytical Poisson.
Used by all three engines (DC, ML, Legacy) to produce a uniform output.

Markets covered:
  - 1X2 Direct (home_win, draw, away_win)
  - Double Chance (dc_1x, dc_x2, dc_12)
  - BTTS Yes / No
  - Total Goals O/U 0.5 – 4.5
  - Combined BTTS + O/U 2.5
  - Home Team Goals O/U 0.5 – 2.5
  - Away Team Goals O/U 0.5 – 2.5
  - Asian Handicap lines -2 to +2

All functions are pure (no DB, no side-effects).
"""

import math
import logging
from typing import Optional

log = logging.getLogger(__name__)

MAX_GOALS = 8

# Asian handicap lines to compute (home perspective)
AH_LINES = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]


# ─── Poisson score matrix ──────────────────────────────────────────────────────

def _poisson_pmf(lam: float, k: int) -> float:
    """P(X=k) for Poisson(λ) — numerically stable via log-gamma."""
    if lam <= 0 or k < 0:
        return 0.0
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def _score_matrix(home_xg: float, away_xg: float) -> dict:
    """Return {(hg, ag): probability} for all goal pairs 0..MAX_GOALS, normalised."""
    home_xg = max(0.10, float(home_xg))
    away_xg = max(0.10, float(away_xg))
    h_pmf = [_poisson_pmf(home_xg, g) for g in range(MAX_GOALS + 1)]
    a_pmf = [_poisson_pmf(away_xg, g) for g in range(MAX_GOALS + 1)]

    mat: dict = {}
    total = 0.0
    for hg in range(MAX_GOALS + 1):
        for ag in range(MAX_GOALS + 1):
            p = h_pmf[hg] * a_pmf[ag]
            mat[(hg, ag)] = p
            total += p
    if total > 0:
        mat = {k: v / total for k, v in mat.items()}
    return mat


# ─── Main calculator ───────────────────────────────────────────────────────────

def compute_all_markets(home_xg: float, away_xg: float) -> dict:
    """
    Compute a full market sheet from expected goals using independent Poisson.
    Returns a dict with all market keys listed in the module docstring.
    """
    mat = _score_matrix(home_xg, away_xg)

    # ── 1X2 ──
    home_win = sum(p for (hg, ag), p in mat.items() if hg > ag)
    draw     = sum(p for (hg, ag), p in mat.items() if hg == ag)
    away_win = sum(p for (hg, ag), p in mat.items() if hg < ag)

    # ── Double Chance ──
    dc_1x = home_win + draw
    dc_x2 = draw + away_win
    dc_12 = home_win + away_win

    # ── BTTS ──
    btts_yes = sum(p for (hg, ag), p in mat.items() if hg > 0 and ag > 0)
    btts_no  = 1.0 - btts_yes

    # ── Total Goals O/U ──
    def _over(n: float) -> float:
        return sum(p for (hg, ag), p in mat.items() if hg + ag > n)

    over_0_5 = _over(0.5)
    over_1_5 = _over(1.5)
    over_2_5 = _over(2.5)
    over_3_5 = _over(3.5)
    over_4_5 = _over(4.5)

    # ── Combined BTTS + O/U 2.5 ──
    btts_yes_over_2_5  = sum(p for (hg, ag), p in mat.items() if hg > 0 and ag > 0 and hg + ag > 2)
    btts_yes_under_2_5 = sum(p for (hg, ag), p in mat.items() if hg > 0 and ag > 0 and hg + ag <= 2)
    btts_no_over_2_5   = sum(p for (hg, ag), p in mat.items() if (hg == 0 or ag == 0) and hg + ag > 2)
    btts_no_under_2_5  = sum(p for (hg, ag), p in mat.items() if (hg == 0 or ag == 0) and hg + ag <= 2)

    # ── Home Team Goals O/U ──
    home_over_0_5 = sum(p for (hg, _), p in mat.items() if hg > 0)
    home_over_1_5 = sum(p for (hg, _), p in mat.items() if hg > 1)
    home_over_2_5 = sum(p for (hg, _), p in mat.items() if hg > 2)

    # ── Away Team Goals O/U ──
    away_over_0_5 = sum(p for (_, ag), p in mat.items() if ag > 0)
    away_over_1_5 = sum(p for (_, ag), p in mat.items() if ag > 1)
    away_over_2_5 = sum(p for (_, ag), p in mat.items() if ag > 2)

    # ── Asian Handicap ──
    # Convention: line L applied to home team.
    # effective_gd = (hg - ag) + L
    # effective > 0  → home AH wins
    # effective == 0 → push (whole lines only, money refunded)
    # effective < 0  → away AH wins
    def _ah(line: float):
        home_ah, push, away_ah = 0.0, 0.0, 0.0
        for (hg, ag), p in mat.items():
            eff = (hg - ag) + line
            if abs(eff) < 1e-9:
                push += p
            elif eff > 0:
                home_ah += p
            else:
                away_ah += p
        return round(home_ah, 4), round(push, 4), round(away_ah, 4)

    ah: dict = {}
    for line in AH_LINES:
        h, psh, a = _ah(line)
        label = f"home{'+' if line >= 0 else ''}{line:g}"
        ah[label] = {"home": h, "push": psh, "away": a}

    return {
        # 1X2
        "home_win": round(home_win, 4),
        "draw":     round(draw,     4),
        "away_win": round(away_win, 4),
        # Double Chance
        "dc_1x": round(dc_1x, 4),
        "dc_x2": round(dc_x2, 4),
        "dc_12": round(dc_12, 4),
        # BTTS
        "btts_yes": round(btts_yes, 4),
        "btts_no":  round(btts_no,  4),
        # Total Goals O/U
        "over_0_5":  round(over_0_5, 4), "under_0_5": round(1 - over_0_5, 4),
        "over_1_5":  round(over_1_5, 4), "under_1_5": round(1 - over_1_5, 4),
        "over_2_5":  round(over_2_5, 4), "under_2_5": round(1 - over_2_5, 4),
        "over_3_5":  round(over_3_5, 4), "under_3_5": round(1 - over_3_5, 4),
        "over_4_5":  round(over_4_5, 4), "under_4_5": round(1 - over_4_5, 4),
        # Combined
        "btts_yes_over_2_5":  round(btts_yes_over_2_5,  4),
        "btts_yes_under_2_5": round(btts_yes_under_2_5, 4),
        "btts_no_over_2_5":   round(btts_no_over_2_5,   4),
        "btts_no_under_2_5":  round(btts_no_under_2_5,  4),
        # Home Goals O/U
        "home_over_0_5": round(home_over_0_5, 4), "home_under_0_5": round(1 - home_over_0_5, 4),
        "home_over_1_5": round(home_over_1_5, 4), "home_under_1_5": round(1 - home_over_1_5, 4),
        "home_over_2_5": round(home_over_2_5, 4), "home_under_2_5": round(1 - home_over_2_5, 4),
        # Away Goals O/U
        "away_over_0_5": round(away_over_0_5, 4), "away_under_0_5": round(1 - away_over_0_5, 4),
        "away_over_1_5": round(away_over_1_5, 4), "away_under_1_5": round(1 - away_over_1_5, 4),
        "away_over_2_5": round(away_over_2_5, 4), "away_under_2_5": round(1 - away_over_2_5, 4),
        # Asian Handicap
        "asian_handicap": ah,
        # Expected goals (source)
        "home_xg": round(float(home_xg), 2),
        "away_xg": round(float(away_xg), 2),
    }


def _override_1x2(markets: dict, probs: dict) -> dict:
    """
    Override Poisson-implied 1X2 and double-chance with the engine's
    calibrated probabilities, which are more accurate than raw Poisson.
    """
    hw = probs.get("home_win", markets["home_win"])
    dr = probs.get("draw",     markets["draw"])
    aw = probs.get("away_win", markets["away_win"])
    # Normalise
    total = hw + dr + aw or 1.0
    hw, dr, aw = hw / total, dr / total, aw / total
    markets = dict(markets)
    markets["home_win"] = round(hw, 4)
    markets["draw"]     = round(dr, 4)
    markets["away_win"] = round(aw, 4)
    markets["dc_1x"]    = round(hw + dr, 4)
    markets["dc_x2"]    = round(dr + aw, 4)
    markets["dc_12"]    = round(hw + aw, 4)
    return markets


# ─── Market blender ────────────────────────────────────────────────────────────

def blend_markets(
    engine_markets: dict,
    weights: dict,
    per_market_weights: Optional[dict] = None,
) -> dict:
    """
    Blend named-engine market dicts using global weights, with optional
    per-market weight overrides.

    Args:
        engine_markets:     {"dc": {...}, "ml": {...}, "legacy": {...}}
        weights:            {"dc": 0.45, "ml": 0.30, "legacy": 0.25}
        per_market_weights: {"btts_yes": {"dc": 0.50, "ml": 0.30, "legacy": 0.20}, ...}
    """
    per_market_weights = per_market_weights or {}
    engines = [e for e in ["dc", "ml", "legacy"] if e in engine_markets]
    if not engines:
        return {}

    # Normalise global weights
    global_w_raw = [weights.get(e, 0.0) for e in engines]
    total_gw = sum(global_w_raw)
    norm_global = [w / total_gw for w in global_w_raw] if total_gw > 0 else [1 / len(engines)] * len(engines)

    result: dict = {}

    # Collect all scalar keys from any engine dict
    all_scalar_keys: set = set()
    for ed in engine_markets.values():
        if isinstance(ed, dict):
            all_scalar_keys.update(k for k, v in ed.items() if isinstance(v, (int, float)) and k not in ("home_xg", "away_xg"))

    for key in all_scalar_keys:
        # Use per-market weights when available
        mkt_w_dict = per_market_weights.get(key)
        if mkt_w_dict and isinstance(mkt_w_dict, dict):
            raw = [mkt_w_dict.get(e, 0.0) for e in engines]
            total_mw = sum(raw)
            norm_mkt = [w / total_mw for w in raw] if total_mw > 0 else norm_global
        else:
            norm_mkt = norm_global

        blended = sum(
            norm_mkt[i] * engine_markets[e].get(key, 0.0)
            for i, e in enumerate(engines)
        )
        result[key] = round(blended, 4)

    # xG — plain weighted average via global weights
    result["home_xg"] = round(sum(norm_global[i] * engine_markets[e].get("home_xg", 1.35) for i, e in enumerate(engines)), 2)
    result["away_xg"] = round(sum(norm_global[i] * engine_markets[e].get("away_xg", 1.10) for i, e in enumerate(engines)), 2)

    # Asian Handicap — blend nested subdicts using global weights
    ah_keys: set = set()
    for ed in engine_markets.values():
        if isinstance(ed, dict):
            ah_keys.update(ed.get("asian_handicap", {}).keys())

    blended_ah: dict = {}
    for ah_label in sorted(ah_keys):
        blended_ah[ah_label] = {}
        for sub in ("home", "push", "away"):
            blended_ah[ah_label][sub] = round(
                sum(
                    norm_global[i] * engine_markets[e].get("asian_handicap", {}).get(ah_label, {}).get(sub, 0.0)
                    for i, e in enumerate(engines)
                ), 4
            )
    result["asian_handicap"] = blended_ah
    return result


# ─── Best bet selector ─────────────────────────────────────────────────────────

# (market_key, min_probability, market_category)
_THRESHOLDS = [
    ("home_win",            0.55, "1X2"),
    ("away_win",            0.48, "1X2"),
    ("draw",                0.35, "1X2"),
    ("dc_1x",               0.65, "Double Chance"),
    ("dc_x2",               0.65, "Double Chance"),
    ("dc_12",               0.62, "Double Chance"),
    ("btts_yes",            0.53, "BTTS"),
    ("btts_no",             0.58, "BTTS"),
    ("over_1_5",            0.75, "Total Goals"),
    ("over_2_5",            0.53, "Total Goals"),
    ("over_3_5",            0.38, "Total Goals"),
    ("under_2_5",           0.48, "Total Goals"),
    ("under_3_5",           0.60, "Total Goals"),
    ("btts_yes_over_2_5",   0.38, "Combined"),
    ("btts_yes_under_2_5",  0.22, "Combined"),
    ("btts_no_over_2_5",    0.18, "Combined"),
    ("btts_no_under_2_5",   0.32, "Combined"),
    ("home_over_0_5",       0.72, "Home Goals"),
    ("home_over_1_5",       0.48, "Home Goals"),
    ("home_over_2_5",       0.28, "Home Goals"),
    ("away_over_0_5",       0.68, "Away Goals"),
    ("away_over_1_5",       0.42, "Away Goals"),
    ("away_over_2_5",       0.22, "Away Goals"),
]

_FRIENDLY = {
    "home_win":            "Home Win",
    "draw":                "Draw",
    "away_win":            "Away Win",
    "dc_1x":               "Double Chance 1X (Home or Draw)",
    "dc_x2":               "Double Chance X2 (Draw or Away)",
    "dc_12":               "Double Chance 12 (Home or Away)",
    "btts_yes":            "Both Teams to Score — Yes",
    "btts_no":             "Both Teams to Score — No",
    "over_1_5":            "Over 1.5 Goals",
    "over_2_5":            "Over 2.5 Goals",
    "over_3_5":            "Over 3.5 Goals",
    "under_2_5":           "Under 2.5 Goals",
    "under_3_5":           "Under 3.5 Goals",
    "btts_yes_over_2_5":   "BTTS Yes + Over 2.5",
    "btts_yes_under_2_5":  "BTTS Yes + Under 2.5",
    "btts_no_over_2_5":    "BTTS No + Over 2.5",
    "btts_no_under_2_5":   "BTTS No + Under 2.5",
    "home_over_0_5":       "{home} to Score",
    "home_over_1_5":       "{home} Over 1.5 Goals",
    "home_over_2_5":       "{home} Over 2.5 Goals",
    "away_over_0_5":       "{away} to Score",
    "away_over_1_5":       "{away} Over 1.5 Goals",
    "away_over_2_5":       "{away} Over 2.5 Goals",
}

_AH_THRESHOLD = 0.58


def select_best_bets(
    markets: dict,
    home_name: str = "Home",
    away_name: str = "Away",
) -> list:
    """
    Return ranked bet recommendations from a consensus market dict.
    Max 12 bets, sorted strong → value → speculative.
    """
    bets = []
    for key, threshold, category in _THRESHOLDS:
        prob = markets.get(key)
        if prob is None or prob < threshold:
            continue
        name = _FRIENDLY.get(key, key).replace("{home}", home_name).replace("{away}", away_name)
        tier = "strong" if prob >= 0.70 else ("value" if prob >= 0.55 else "speculative")
        bets.append({
            "market_key":      key,
            "market":          category,
            "pick":            name,
            "probability":     round(prob, 4),
            "probability_pct": round(prob * 100, 1),
            "fair_odds":       round(1.0 / prob, 2) if prob > 0 else None,
            "tier":            tier,
        })

    # Asian Handicap picks
    for label, vals in markets.get("asian_handicap", {}).items():
        for side in ("home", "away"):
            p = vals.get(side, 0.0)
            if p < _AH_THRESHOLD:
                continue
            line_str = label.replace("home", "")
            team = home_name if side == "home" else away_name
            tier = "strong" if p >= 0.70 else "value"
            bets.append({
                "market_key":      f"ah_{side}_{label}",
                "market":          "Asian Handicap",
                "pick":            f"{team} AH {line_str}",
                "probability":     round(p, 4),
                "probability_pct": round(p * 100, 1),
                "fair_odds":       round(1.0 / p, 2) if p > 0 else None,
                "tier":            tier,
            })

    _tier_rank = {"strong": 0, "value": 1, "speculative": 2}
    bets.sort(key=lambda b: (_tier_rank.get(b["tier"], 9), -b["probability"]))
    return bets[:12]
