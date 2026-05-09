-- ============================================================
-- Migration: prediction_markets_log
-- Run once against your Postgres database.
-- After applying, the market_logger, market_grader,
-- market_weights, and market_recalibrator modules will
-- automatically populate and use this table.
-- ============================================================

CREATE TABLE IF NOT EXISTS prediction_markets_log (
    id                  SERIAL PRIMARY KEY,

    -- ── Match identity ────────────────────────────────────────
    match_id            INTEGER,
    fixture_id          INTEGER,          -- API fixture id (may differ from internal match_id)
    home_team           TEXT NOT NULL,
    away_team           TEXT NOT NULL,
    league              TEXT,
    match_date          DATE,

    -- ── Full market sheet per engine (JSONB) ──────────────────
    -- Shape: { "dc": {market_data}, "ml": {market_data}, "legacy": {market_data} }
    -- Each engine dict contains keys: result_1x2, double_chance, btts, over_2_5,
    --   combined, home_goals_ou, away_goals_ou, asian_handicap (all as {prob, pick, confidence})
    engine_predictions  JSONB,

    -- ── Consensus blended output ──────────────────────────────
    consensus_markets   JSONB,            -- Same shape as engine_predictions but blended
    best_bets           JSONB,            -- Array of {market, pick, confidence, value_edge}
    weights_used        JSONB,            -- {"dc": 0.37, "ml": 0.35, "legacy": 0.28} per market

    -- ── Prediction metadata ───────────────────────────────────
    predicted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- ── Grading (filled after match completes) ────────────────
    evaluated_at        TIMESTAMPTZ,
    actual_home_goals   INTEGER,
    actual_away_goals   INTEGER,

    -- ── Per-market, per-engine correctness ───────────────────
    -- Shape: { "result_1x2": {"dc": true, "ml": false, "legacy": true, "consensus": true}, ... }
    grades              JSONB,

    -- ── Brier scores per market per engine ───────────────────
    -- Shape: { "result_1x2": {"dc": 0.12, "ml": 0.18, "legacy": 0.22, "consensus": 0.11}, ... }
    brier_scores        JSONB,

    -- ── Uniqueness ───────────────────────────────────────────
    CONSTRAINT uq_pml_match UNIQUE (match_id)
);

-- Index for grading lookups (ungraded rows with a known match)
CREATE INDEX IF NOT EXISTS idx_pml_ungraded
    ON prediction_markets_log (match_id)
    WHERE evaluated_at IS NULL;

-- Index for performance queries (graded rows only)
CREATE INDEX IF NOT EXISTS idx_pml_evaluated
    ON prediction_markets_log (league, evaluated_at)
    WHERE evaluated_at IS NOT NULL;

-- Index on date for rolling-window queries
CREATE INDEX IF NOT EXISTS idx_pml_match_date
    ON prediction_markets_log (match_date DESC NULLS LAST);
