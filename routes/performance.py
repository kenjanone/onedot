"""
Prediction Performance API Routes
===================================
GET /api/performance             — Overall metrics (Brier, RPS, accuracy, ROI)
GET /api/performance/drift       — Rolling 20-match Brier score over time
GET /api/performance/calibration — Calibration bin table
GET /api/performance/per-league  — Per-league accuracy and Brier breakdown
GET /api/performance/confusion   — Confusion matrix
GET /api/performance/markets     — Market accuracy (BTTS, Over 2.5)
"""

import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from database import get_connection
from ml.metrics import MetricsEngine

router = APIRouter()
log    = logging.getLogger(__name__)


def _load_completed(cur, league: str = None) -> pd.DataFrame:
    query = """
        SELECT
            id, home_team, away_team, league, match_date,
            home_win_prob, draw_prob, away_win_prob,
            predicted, actual, correct,
            confidence, confidence_score,
            btts_yes, over_2_5, home_xg, away_xg,
            btts_correct, over_2_5_correct
        FROM prediction_log
        WHERE actual IS NOT NULL
          AND home_win_prob IS NOT NULL
          AND draw_prob     IS NOT NULL
          AND away_win_prob IS NOT NULL
    """
    params = []
    if league:
        query += " AND league = %s"
        params.append(league)
    query += " ORDER BY match_date ASC NULLS LAST, created_at ASC"

    try:
        cur.execute(query, params)
        rows = cur.fetchall()
    except Exception as e:
        log.warning("prediction_log query failed: %s", e)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df = df.rename(columns={
        "home_win_prob": "prob_home_win",
        "draw_prob":     "prob_draw",
        "away_win_prob": "prob_away_win",
        "actual":        "actual_outcome",
        "predicted":     "predicted_outcome",
    })

    outcome_map = {
        "Home Win": 2, "home_win": 2, "2": 2,
        "Draw":     1, "draw":     1, "1": 1,
        "Away Win": 0, "away_win": 0, "0": 0,
    }
    if "actual_outcome" in df.columns:
        df["actual_int"] = df["actual_outcome"].astype(str).map(outcome_map)
    else:
        return pd.DataFrame()

    if "predicted_outcome" in df.columns:
        df["predicted_int"] = df["predicted_outcome"].astype(str).map(outcome_map)

    df = df.dropna(subset=["actual_int", "prob_home_win", "prob_draw", "prob_away_win"])
    return df


def _load_all_predictions(cur) -> pd.DataFrame:
    try:
        cur.execute("""
            SELECT league,
                   COUNT(*) AS total_predictions,
                   SUM(CASE WHEN actual IS NOT NULL THEN 1 ELSE 0 END) AS evaluated
            FROM prediction_log
            WHERE league IS NOT NULL
            GROUP BY league
            ORDER BY total_predictions DESC
        """)
        rows = cur.fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    except Exception as e:
        log.warning("_load_all_predictions failed: %s", e)
        return pd.DataFrame()


def _build_matrices(df: pd.DataFrame):
    p_aw = df["prob_away_win"].astype(float).values
    p_d  = df["prob_draw"].astype(float).values
    p_hw = df["prob_home_win"].astype(float).values
    probs    = np.stack([p_aw, p_d, p_hw], axis=1)
    outcomes = df["actual_int"].astype(int).values
    return probs, outcomes


# ─── GET /api/performance ─────────────────────────────────────────────────────

@router.get("")
def get_overall_performance(league: str = None):
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df = _load_completed(cur, league)
    finally:
        conn.close()

    if df.empty:
        return {"message": "No completed predictions in prediction_log yet.", "n": 0}

    probs, outcomes = _build_matrices(df)
    summary = MetricsEngine.full_summary(probs, outcomes)

    roi_records = []
    for _, r in df.iterrows():
        pred = r.get("predicted_int")
        if pd.isna(pred):
            continue
        pred = int(pred)
        odds_cols = {2: "market_home_odds", 1: "market_draw_odds", 0: "market_away_odds"}
        col = odds_cols.get(pred)
        if col and col in r and pd.notna(r[col]):
            roi_records.append({
                "predicted_outcome": pred,
                "actual_outcome":    int(r["actual_int"]),
                "odds_taken":        float(r[col]),
            })

    roi = MetricsEngine.roi(roi_records) if roi_records else {}
    return {**summary, "roi": roi, "league": league or "All Leagues"}


# ─── GET /api/performance/drift ───────────────────────────────────────────────

@router.get("/drift")
def get_rolling_drift(window: int = 20):
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df = _load_completed(cur)
    finally:
        conn.close()

    if df.empty or len(df) < window:
        return {"message": f"Need at least {window} completed predictions. Have {len(df)}.", "n": len(df)}

    df = df.reset_index(drop=True)
    rows = []
    for i in range(window, len(df) + 1):
        chunk = df.iloc[i - window: i]
        probs, outcomes = _build_matrices(chunk)
        rows.append({
            "match_number":  i,
            "rolling_brier": round(MetricsEngine.brier_score(probs, outcomes), 4),
            "rolling_rps":   round(MetricsEngine.rps(probs, outcomes), 4),
            "rolling_acc":   round(MetricsEngine.accuracy(probs, outcomes) * 100, 2),
        })

    return {"window": window, "n_predictions": len(df), "latest": rows[-1] if rows else {}, "drift": rows}


# ─── GET /api/performance/calibration ─────────────────────────────────────────

@router.get("/calibration")
def get_calibration():
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df = _load_completed(cur)
    finally:
        conn.close()

    if df.empty:
        return {"message": "No completed predictions yet.", "bins": []}

    probs, outcomes = _build_matrices(df)
    cal_df = MetricsEngine.calibration(probs, outcomes)
    well_calibrated = int((cal_df["well_calibrated"]).sum())
    return {
        "n_predictions":        len(df),
        "n_bins":               len(cal_df),
        "well_calibrated_bins": well_calibrated,
        "bins":                 cal_df.to_dict(orient="records"),
    }


# ─── GET /api/performance/per-league ─────────────────────────────────────────

def _league_tier(accuracy, evaluated) -> dict:
    """
    Return a confidence tier for a league based on accuracy + sample size.
    Predictions are NEVER disabled — the tier is a UI signal only, letting
    the models keep accumulating feedback data to graduate to higher tiers.

    reliable   (green)  — accuracy >= 45% AND >= 15 evaluated
    learning   (yellow) — accuracy 30-44% OR < 15 evaluated
    unreliable (red)    — accuracy < 30% AND >= 10 evaluated
    new        (grey)   — fewer than 10 evaluated (insufficient data)
    """
    if accuracy is None or evaluated is None or evaluated < 10:
        return {"tier": "new",        "label": "Insufficient Data",    "color": "grey"}
    if accuracy < 30:
        return {"tier": "unreliable", "label": "Low Confidence",       "color": "red"}
    if accuracy < 45 or evaluated < 15:
        return {"tier": "learning",   "label": "Model Still Learning", "color": "yellow"}
    return     {"tier": "reliable",   "label": "Reliable",             "color": "green"}


@router.get("/per-league")
def get_per_league():
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df_completed = _load_completed(cur)
        df_all       = _load_all_predictions(cur)
    finally:
        conn.close()

    if df_all.empty:
        return {"message": "No predictions yet.", "leagues": []}

    evaluated_metrics = {}
    if not df_completed.empty and "league" in df_completed.columns:
        for league, grp in df_completed.groupby("league"):
            if len(grp) < 1:
                continue
            probs, outcomes = _build_matrices(grp)
            evaluated_metrics[league] = {
                "evaluated":  len(grp),
                "accuracy":   round(MetricsEngine.accuracy(probs, outcomes) * 100, 2),
                "brier":      round(MetricsEngine.brier_score(probs, outcomes), 4),
                "rps":        round(MetricsEngine.rps(probs, outcomes), 4),
            }

    rows = []
    for _, row in df_all.iterrows():
        league    = row["league"]
        total     = int(row["total_predictions"])
        evaluated = int(row["evaluated"])
        metrics   = evaluated_metrics.get(league, {})
        accuracy  = metrics.get("accuracy")
        tier      = _league_tier(accuracy, evaluated)
        rows.append({
            "league":          league,
            "matches":         total,
            "evaluated":       evaluated,
            "accuracy":        accuracy,
            "brier":           metrics.get("brier"),
            "rps":             metrics.get("rps"),
            "has_results":     evaluated > 0,
            "confidence_tier": tier["tier"],
            "tier_label":      tier["label"],
            "tier_color":      tier["color"],
        })

    rows.sort(key=lambda x: (not x["has_results"], x.get("brier") or 999))
    return {"leagues": rows}


# ─── GET /api/performance/confusion ──────────────────────────────────────────

@router.get("/confusion")
def get_confusion_matrix():
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df = _load_completed(cur)
    finally:
        conn.close()

    if df.empty:
        return {"message": "No completed predictions yet."}

    probs, outcomes = _build_matrices(df)
    cm = MetricsEngine.confusion_matrix(probs, outcomes)
    # orient='index' gives {row → {col: value}} which matches the frontend's
    # matrix["Actual Away Win"]["Pred Home Win"] access pattern.
    return {"n_predictions": len(df), "matrix": cm.to_dict(orient='index'), "labels": ["Away Win", "Draw", "Home Win"]}


# ─── GET /api/performance/markets ────────────────────────────────────────────

@router.get("/markets")
def get_market_accuracy():
    """
    Full per-market, per-engine accuracy breakdown.
    Uses prediction_markets_log when graded rows exist;
    falls back to legacy prediction_log BTTS/O2.5 columns.
    """
    _MARKET_META = {
        "home_win":          "Home Win",
        "draw":              "Draw",
        "away_win":          "Away Win",
        "dc_1x":             "Double Chance 1X",
        "dc_x2":             "Double Chance X2",
        "dc_12":             "Double Chance 12",
        "btts_yes":          "BTTS Yes",
        "btts_no":           "BTTS No",
        "over_1_5":          "Over 1.5",
        "over_2_5":          "Over 2.5",
        "over_3_5":          "Over 3.5",
        "under_2_5":         "Under 2.5",
        "under_3_5":         "Under 3.5",
        "btts_yes_over_2_5": "BTTS Yes + Over 2.5",
        "home_over_1_5":     "Home Over 1.5",
        "away_over_1_5":     "Away Over 1.5",
    }

    def _acc(correct, total):
        c, t = float(correct or 0), int(total or 0)
        return round(c / t * 100, 1) if t > 0 else None

    def _tier(acc_pct, n):
        if n < 10:         return "learning"
        if acc_pct >= 60:  return "reliable"
        if acc_pct >= 45:  return "learning"
        return "unreliable"

    def _pct(correct, total):
        return round(int(correct or 0) / int(total) * 100, 1) if total and int(total) > 0 else None

    conn = get_connection()
    cur  = conn.cursor()

    try:
        # Check if new table has graded data
        total_graded = 0
        try:
            cur.execute("SELECT COUNT(*) AS n FROM prediction_markets_log WHERE evaluated_at IS NOT NULL")
            total_graded = int((cur.fetchone() or {}).get("n", 0))
        except Exception:
            pass

        if total_graded == 0:
            # Legacy fallback
            try:
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE btts_correct IS NOT NULL)     AS bt,
                        SUM(CASE WHEN btts_correct THEN 1 ELSE 0 END)        AS bc,
                        COUNT(*) FILTER (WHERE over_2_5_correct IS NOT NULL) AS ot,
                        SUM(CASE WHEN over_2_5_correct THEN 1 ELSE 0 END)    AS oc
                    FROM prediction_log
                """)
                t = cur.fetchone()
                bt = int(t["bt"] or 0) if t else 0
                ot = int(t["ot"] or 0) if t else 0
                return {
                    "total_graded": 0, "legacy_data": True,
                    "markets": {
                        "btts_yes": {"label": "BTTS Yes", "total": bt, "consensus": {"accuracy_pct": _pct(t["bc"], bt) if t else None}},
                        "over_2_5": {"label": "Over 2.5", "total": ot, "consensus": {"accuracy_pct": _pct(t["oc"], ot) if t else None}},
                    },
                    "league_rankings": {}, "summary": {},
                }
            except Exception:
                return {"total_graded": 0, "markets": {}, "league_rankings": {}, "summary": {}}

        # Full market accuracy from prediction_markets_log
        markets_out: dict = {}
        for mkt_key, mkt_label in _MARKET_META.items():
            try:
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE grades ? %(mk)s)                              AS n,
                        SUM((grades->%(mk)s->>'consensus')::boolean::int)::float             AS cons_c,
                        AVG((brier_scores->%(mk)s->>'consensus')::float)                     AS cons_b,
                        COUNT(*) FILTER (WHERE grades->%(mk)s->>'dc' IS NOT NULL)            AS dc_n,
                        SUM((grades->%(mk)s->>'dc')::boolean::int)::float                    AS dc_c,
                        AVG((brier_scores->%(mk)s->>'dc')::float)                            AS dc_b,
                        COUNT(*) FILTER (WHERE grades->%(mk)s->>'ml' IS NOT NULL)            AS ml_n,
                        SUM((grades->%(mk)s->>'ml')::boolean::int)::float                    AS ml_c,
                        AVG((brier_scores->%(mk)s->>'ml')::float)                            AS ml_b,
                        COUNT(*) FILTER (WHERE grades->%(mk)s->>'legacy' IS NOT NULL)        AS leg_n,
                        SUM((grades->%(mk)s->>'legacy')::boolean::int)::float                AS leg_c,
                        AVG((brier_scores->%(mk)s->>'legacy')::float)                        AS leg_b
                    FROM prediction_markets_log
                    WHERE evaluated_at IS NOT NULL AND grades ? %(mk)s
                """, {"mk": mkt_key})
                r = cur.fetchone()
                if not r or not r["n"]:
                    continue
                n = int(r["n"] or 0)
                best_eng, best_acc = "consensus", _acc(r["cons_c"], n) or 0
                eng_stats = {}
                for eng, cor, bri, dn in [
                    ("consensus", "cons_c", "cons_b", "n"),
                    ("dc",        "dc_c",   "dc_b",   "dc_n"),
                    ("ml",        "ml_c",   "ml_b",   "ml_n"),
                    ("legacy",    "leg_c",  "leg_b",  "leg_n"),
                ]:
                    en  = int(r[dn] or 0)
                    acc = _acc(r[cor], en)
                    eng_stats[eng] = {"n": en, "accuracy_pct": acc,
                                      "avg_brier": round(float(r[bri] or 0), 4) if r[bri] else None}
                    if acc is not None and acc > best_acc:
                        best_acc, best_eng = acc, eng
                markets_out[mkt_key] = {"label": mkt_label, "total": n,
                                        "best_engine": best_eng, "best_accuracy": round(best_acc, 1),
                                        **eng_stats}
            except Exception as exc:
                log.debug("market acc %s: %s", mkt_key, exc)

        # Per-league rankings for each market
        league_rankings: dict = {}
        for mkt_key in list(markets_out.keys())[:9]:
            try:
                cur.execute("""
                    SELECT league,
                        COUNT(*) FILTER (WHERE grades ? %(mk)s)                          AS n,
                        SUM((grades->%(mk)s->>'consensus')::boolean::int)::float         AS correct
                    FROM prediction_markets_log
                    WHERE evaluated_at IS NOT NULL AND grades ? %(mk)s AND league IS NOT NULL
                    GROUP BY league HAVING COUNT(*) FILTER (WHERE grades ? %(mk)s) >= 5
                    ORDER BY (SUM((grades->%(mk)s->>'consensus')::boolean::int)::float /
                              NULLIF(COUNT(*) FILTER (WHERE grades ? %(mk)s), 0)) DESC NULLS LAST
                    LIMIT 20
                """, {"mk": mkt_key})
                ranked = []
                for rank, row in enumerate(cur.fetchall(), start=1):
                    nr  = int(row["n"] or 0)
                    acc = round(float(row["correct"] or 0) / nr * 100, 1) if nr > 0 else None
                    ranked.append({"rank": rank, "league": row["league"], "n": nr,
                                   "accuracy_pct": acc, "tier": _tier(acc or 0, nr)})
                league_rankings[mkt_key] = ranked
            except Exception as exc:
                log.debug("league ranking %s: %s", mkt_key, exc)

        # Summary: best and worst markets
        ranked_mkts = sorted(
            [(k, v.get("consensus", {}).get("accuracy_pct") or 0) for k, v in markets_out.items()],
            key=lambda x: -x[1],
        )
        best_markets  = [{"market": k, "label": _MARKET_META.get(k, k), "accuracy_pct": a}
                         for k, a in ranked_mkts[:3] if a > 0]
        worst_markets = [{"market": k, "label": _MARKET_META.get(k, k), "accuracy_pct": a}
                         for k, a in reversed(ranked_mkts[-3:]) if a > 0]

        return {
            "total_graded":    total_graded,
            "markets":         markets_out,
            "league_rankings": league_rankings,
            "summary":         {"best_markets": best_markets, "worst_markets": worst_markets},
        }

    except Exception as exc:
        log.warning("get_market_accuracy failed: %s", exc)
        return {"total_graded": 0, "markets": {}, "league_rankings": {}, "summary": {}}
    finally:
        conn.close()



# ─── GET /api/performance/engines ────────────────────────────────────────────

@router.get("/engines")
def get_engine_performance():
    """
    Per-engine accuracy breakdown from prediction_log.
    Uses dc_correct, ml_correct, legacy_correct, and correct (consensus)
    columns that are populated by do_evaluate_predictions().
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        # Overall per-engine counts
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE dc_correct     IS NOT NULL) AS dc_evaluated,
                SUM(CASE WHEN dc_correct     THEN 1 ELSE 0 END)    AS dc_correct,
                COUNT(*) FILTER (WHERE ml_correct     IS NOT NULL) AS ml_evaluated,
                SUM(CASE WHEN ml_correct     THEN 1 ELSE 0 END)    AS ml_correct,
                COUNT(*) FILTER (WHERE legacy_correct IS NOT NULL) AS leg_evaluated,
                SUM(CASE WHEN legacy_correct THEN 1 ELSE 0 END)    AS leg_correct,
                COUNT(*) FILTER (WHERE correct        IS NOT NULL) AS con_evaluated,
                SUM(CASE WHEN correct        THEN 1 ELSE 0 END)    AS con_correct
            FROM prediction_log
        """)
        r = cur.fetchone()

        def _acc(correct, total):
            c, t = int(correct or 0), int(total or 0)
            return round(c / t * 100, 1) if t > 0 else None

        overall = {
            "dc":        {"evaluated": int(r["dc_evaluated"]  or 0), "correct": int(r["dc_correct"]  or 0), "accuracy_pct": _acc(r["dc_correct"],  r["dc_evaluated"])},
            "ml":        {"evaluated": int(r["ml_evaluated"]  or 0), "correct": int(r["ml_correct"]  or 0), "accuracy_pct": _acc(r["ml_correct"],  r["ml_evaluated"])},
            "legacy":    {"evaluated": int(r["leg_evaluated"] or 0), "correct": int(r["leg_correct"] or 0), "accuracy_pct": _acc(r["leg_correct"], r["leg_evaluated"])},
            "consensus": {"evaluated": int(r["con_evaluated"] or 0), "correct": int(r["con_correct"] or 0), "accuracy_pct": _acc(r["con_correct"], r["con_evaluated"])},
        }

        # Per-outcome breakdown (Home Win / Draw / Away Win) for each engine
        cur.execute("""
            SELECT
                actual,
                SUM(CASE WHEN dc_correct     THEN 1 ELSE 0 END) AS dc_correct,
                COUNT(*) FILTER (WHERE dc_correct IS NOT NULL)   AS dc_total,
                SUM(CASE WHEN ml_correct     THEN 1 ELSE 0 END) AS ml_correct,
                COUNT(*) FILTER (WHERE ml_correct IS NOT NULL)   AS ml_total,
                SUM(CASE WHEN legacy_correct THEN 1 ELSE 0 END) AS leg_correct,
                COUNT(*) FILTER (WHERE legacy_correct IS NOT NULL) AS leg_total,
                SUM(CASE WHEN correct        THEN 1 ELSE 0 END) AS con_correct,
                COUNT(*) FILTER (WHERE correct IS NOT NULL)      AS con_total
            FROM prediction_log
            WHERE actual IS NOT NULL
            GROUP BY actual
            ORDER BY actual
        """)
        by_outcome = []
        for row in cur.fetchall():
            act = row["actual"]
            by_outcome.append({
                "outcome":   act,
                "dc":     {"correct": int(row["dc_correct"]  or 0), "total": int(row["dc_total"]  or 0), "accuracy_pct": _acc(row["dc_correct"],  row["dc_total"])},
                "ml":     {"correct": int(row["ml_correct"]  or 0), "total": int(row["ml_total"]  or 0), "accuracy_pct": _acc(row["ml_correct"],  row["ml_total"])},
                "legacy": {"correct": int(row["leg_correct"] or 0), "total": int(row["leg_total"] or 0), "accuracy_pct": _acc(row["leg_correct"], row["leg_total"])},
                "consensus": {"correct": int(row["con_correct"] or 0), "total": int(row["con_total"] or 0), "accuracy_pct": _acc(row["con_correct"], row["con_total"])},
            })

        # Recent 30-day trend per engine
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE dc_correct IS NOT NULL)     AS dc_evaluated,
                SUM(CASE WHEN dc_correct     THEN 1 ELSE 0 END)    AS dc_correct,
                COUNT(*) FILTER (WHERE ml_correct IS NOT NULL)     AS ml_evaluated,
                SUM(CASE WHEN ml_correct     THEN 1 ELSE 0 END)    AS ml_correct,
                COUNT(*) FILTER (WHERE legacy_correct IS NOT NULL) AS leg_evaluated,
                SUM(CASE WHEN legacy_correct THEN 1 ELSE 0 END)    AS leg_correct,
                COUNT(*) FILTER (WHERE correct IS NOT NULL)        AS con_evaluated,
                SUM(CASE WHEN correct        THEN 1 ELSE 0 END)    AS con_correct
            FROM prediction_log
            WHERE evaluated_at >= NOW() - INTERVAL '30 days'
        """)
        t = cur.fetchone()
        last_30 = {
            "dc":        {"evaluated": int(t["dc_evaluated"]  or 0), "correct": int(t["dc_correct"]  or 0), "accuracy_pct": _acc(t["dc_correct"],  t["dc_evaluated"])},
            "ml":        {"evaluated": int(t["ml_evaluated"]  or 0), "correct": int(t["ml_correct"]  or 0), "accuracy_pct": _acc(t["ml_correct"],  t["ml_evaluated"])},
            "legacy":    {"evaluated": int(t["leg_evaluated"] or 0), "correct": int(t["leg_correct"] or 0), "accuracy_pct": _acc(t["leg_correct"], t["leg_evaluated"])},
            "consensus": {"evaluated": int(t["con_evaluated"] or 0), "correct": int(t["con_correct"] or 0), "accuracy_pct": _acc(t["con_correct"], t["con_evaluated"])},
        }

        return {
            "overall":    overall,
            "last_30":    last_30,
            "by_outcome": by_outcome,
        }
    except Exception as e:
        log.warning("get_engine_performance failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

