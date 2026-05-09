-- ============================================================
-- PlusOne — Performance Index Migration
-- Run ONCE in Supabase SQL Editor (or psql).
-- All statements use IF NOT EXISTS — safe to re-run.
-- ============================================================

-- ── Index strategy ──────────────────────────────────────────
--
-- PARTIAL B-TREE  — for upcoming fixtures: indexes only the
--   small subset of rows where home_score IS NULL.  Storage:
--   O(upcoming_count) vs O(all_matches) for a full index.
--
-- BRIN  — for match_date on the full matches table.
--   Matches are inserted roughly in date order, so BRIN is
--   100-1000x smaller than a B-tree index (pages-level
--   granularity) and still fast for range scans.
--
-- COMPOSITE B-TREE  — for form / H2H lookups; most selective
--   column listed first (team_id before match_date).
--
-- ── Hot path 1: upcoming fixture scan ───────────────────────
-- Used by consensus_engine.upcoming_consensus_fast() and
-- routes/markets.py /upcoming.
-- Query pattern: WHERE home_score IS NULL AND match_date >= CURRENT_DATE
-- Partial index: only indexes the small set of future matches.

CREATE INDEX IF NOT EXISTS idx_matches_upcoming
    ON matches (match_date ASC)
    WHERE home_score IS NULL;

-- ── Hot path 2: upcoming by league (filtered) ───────────────
-- Serves ?league_id= filter on /upcoming without full scan.

CREATE INDEX IF NOT EXISTS idx_matches_league_upcoming
    ON matches (league_id, match_date ASC)
    WHERE home_score IS NULL;

-- ── Hot path 3: team form — home matches ────────────────────
-- _compute_form() in batch_features.py: filter by home_team_id
-- then sort by match_date DESC. Already capped at 50 per team
-- by DataCache, but this avoids the full-table scan to get there.

CREATE INDEX IF NOT EXISTS idx_matches_home_form
    ON matches (home_team_id, match_date DESC)
    WHERE home_score IS NOT NULL;

-- ── Hot path 4: team form — away matches ────────────────────

CREATE INDEX IF NOT EXISTS idx_matches_away_form
    ON matches (away_team_id, match_date DESC)
    WHERE home_score IS NOT NULL;

-- ── Hot path 5: H2H lookups ─────────────────────────────────
-- NOTE: h2h_index in DataCache now handles in-memory O(1) H2H
-- for batch/training paths. This index covers the raw SQL path
-- in routes/markets.py match-preview (live queries, not batched).

CREATE INDEX IF NOT EXISTS idx_matches_h2h_home
    ON matches (home_team_id, away_team_id, match_date DESC)
    WHERE home_score IS NOT NULL;

-- ── Hot path 6: BRIN on match_date (full table) ─────────────
-- Supports DataCache._load() which fetches all matches in a
-- 4-year window: WHERE match_date >= CURRENT_DATE - INTERVAL '4 years'
-- BRIN is orders of magnitude smaller than B-tree for insert-ordered data.

CREATE INDEX IF NOT EXISTS idx_matches_date_brin
    ON matches USING BRIN (match_date)
    WITH (pages_per_range = 32);

-- ── Hot path 7: calibration queries ─────────────────────────
-- feedback_calibrator.py and dc_engine.fit_calibrator_from_log()
-- read: WHERE actual IS NOT NULL ORDER BY evaluated_at DESC LIMIT N

CREATE INDEX IF NOT EXISTS idx_pred_log_calibration
    ON prediction_log (evaluated_at DESC)
    WHERE actual IS NOT NULL;

-- ── Hot path 8: DataCache bulk load — league_standings ───────
-- DataCache loads: WHERE season_id IN (%s, %s)
-- Composite (season_id, team_id) covers both the IN filter and
-- the (team_id, league_id, season_id) key construction in Python.

CREATE INDEX IF NOT EXISTS idx_standings_season_team
    ON league_standings (season_id, team_id);

-- ── Hot path 9: DataCache bulk load — team_squad_stats ───────

CREATE INDEX IF NOT EXISTS idx_squad_stats_season_team
    ON team_squad_stats (season_id, team_id);

-- ── Hot path 10: DataCache bulk load — player_stats ──────────
-- Query: WHERE minutes IS NOT NULL AND season_id IN (...)
-- ORDER BY team_id, season_id, player_name, minutes DESC

CREATE INDEX IF NOT EXISTS idx_player_stats_season_team
    ON player_stats (season_id, team_id)
    WHERE minutes IS NOT NULL;

-- ── Hot path 11: ungraded predictions (auto-evaluate) ────────
-- do_evaluate_predictions() scans for rows where actual IS NULL
-- and a corresponding completed match exists.

CREATE INDEX IF NOT EXISTS idx_pred_log_ungraded
    ON prediction_log (match_date ASC)
    WHERE actual IS NULL;

-- ── Hot path 12: team venue stats ────────────────────────────
-- DataCache loads: WHERE games > 0
-- (team_id, season_id, venue) covers the triple-key lookup in Python.

CREATE INDEX IF NOT EXISTS idx_venue_stats_lookup
    ON team_venue_stats (team_id, season_id, venue)
    WHERE games > 0;

-- ── Verification ─────────────────────────────────────────────
-- After running, confirm indexes were created:
--
-- SELECT indexname, tablename, indexdef
-- FROM pg_indexes
-- WHERE schemaname = 'public'
--   AND indexname LIKE 'idx_%'
-- ORDER BY tablename, indexname;
