from fastapi import APIRouter, HTTPException
from typing import Optional
from database import get_connection

router = APIRouter()

@router.get("")
def list_teams(league_id: Optional[int] = None):
    """
    Return teams for the dropdown selector.

    When a league_id is provided we return ALL teams that have appeared in
    ANY match for that competition across ALL seasons — not just the latest.

    Rationale for removing the season filter:
      - UEFA competitions use season labels like "2024-25" while domestic
        leagues use "2024-2025". String-sorted LIMIT N windows are fragile
        and cause UEFA teams to vanish whenever a new domestic season starts.
      - The correct intent is: show every team that has ever played in this
        competition. Relegated/promoted filtering makes sense for domestic
        leagues but NOT for knock-out/group-stage UEFA competitions where
        the participant list changes entirely each season.
      - DISTINCT ON (normalised name) already collapses duplicate name
        variants (e.g. "Arsenal" vs "Arsenal FC") to one row.

    For European competitions like the Conference League this was the primary
    cause of seeing only a subset of the ~36 participants: if a match was
    stored under a season name not in the LIMIT 2 window it was invisible.
    """
    conn = get_connection()
    cur  = conn.cursor()
    if league_id:
        # Expanded suffix list covers common UEFA club name prefixes/suffixes
        # that FBref inconsistently includes: BSC, GNK, NK, AS, AC, RC, RB,
        # VfL, VfB, TSG, RCD, UD, SD.  Without stripping these, DISTINCT ON
        # would treat "Young Boys" and "BSC Young Boys" as two different teams
        # and keep both (or worse, drop the one the user expects to see).
        cur.execute("""
            SELECT DISTINCT ON (LOWER(REGEXP_REPLACE(t.name,
                    '^\\s*(BSC|GNK|NK|AS|SS|AC|RC|RB|VfL|VfB|TSG|RCD|UD|SD)\\s+|'
                    '\\s*(FC|AFC|SC|FK|SK|CF|SV|AC|RC|AS|NK|GNK|RB|SS)\\s*$',
                    '', 'i')))
                t.id, t.name, t.logo_url,
                t.league_id AS domestic_league_id,
                dl.name     AS league
            FROM teams t
            JOIN leagues dl ON dl.id = t.league_id
            JOIN matches m  ON (m.home_team_id = t.id OR m.away_team_id = t.id)
            WHERE m.league_id = %s
            ORDER BY
                LOWER(REGEXP_REPLACE(t.name,
                    '^\\s*(BSC|GNK|NK|AS|SS|AC|RC|RB|VfL|VfB|TSG|RCD|UD|SD)\\s+|'
                    '\\s*(FC|AFC|SC|FK|SK|CF|SV|AC|RC|AS|NK|GNK|RB|SS)\\s*$',
                    '', 'i')),
                LENGTH(t.name) ASC,
                t.id ASC
        """, (league_id,))
    else:
        cur.execute("""
            SELECT t.*, l.name AS league
            FROM teams t
            JOIN leagues l ON l.id = t.league_id
            ORDER BY l.name, t.name
        """)
    rows = cur.fetchall()
    conn.close()
    return rows

@router.get("/{team_id}")
def get_team(team_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT t.*, l.name AS league FROM teams t JOIN leagues l ON l.id=t.league_id WHERE t.id=%s", (team_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Team not found")
    return row

@router.get("/{team_id}/head-to-head/{opponent_id}")
def head_to_head(team_id: int, opponent_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT m.match_date, m.gameweek, s.name AS season, l.name AS league,
               ht.name AS home_team, ht.logo_url AS home_logo, m.home_score, 
               m.away_score, at.name AS away_team, at.logo_url AS away_logo,
               m.venue, m.score_raw
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        JOIN leagues l ON l.id = m.league_id
        JOIN seasons s ON s.id = m.season_id
        WHERE (m.home_team_id=%s AND m.away_team_id=%s)
           OR (m.home_team_id=%s AND m.away_team_id=%s)
        ORDER BY m.match_date DESC
    """, (team_id, opponent_id, opponent_id, team_id))
    rows = cur.fetchall()
    conn.close()
    return rows

@router.delete("/{team_id}")
def delete_team(team_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM teams WHERE id=%s RETURNING id", (team_id,))
    row = cur.fetchone()
    conn.commit()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Team not found")
    return {"deleted": team_id}
