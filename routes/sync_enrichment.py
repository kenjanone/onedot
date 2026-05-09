from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from database import get_connection
from routes.deps import require_admin
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

class SyncEnrichmentPayload(BaseModel):
    league: str
    odds: Optional[List[dict]] = []
    injuries: Optional[List[dict]] = []
    clubelo: Optional[List[dict]] = []

def safe_num(val):
    if val is None or val == "": return None
    try: return float(val)
    except: return None


# ClubElo uses 2-4 char country/league codes; map each to the canonical full name
# used by FBref so we don't create duplicate league rows.
CLUBELO_LEAGUE_MAP = {
    # ── Top divisions ──────────────────────────────────────────────────────────
    "Aut": "Austrian Bundesliga",
    "Bel": "Belgian Pro League",
    "Bul": "Bulgarian First League",
    "Cro": "Croatian Football League",
    "Cze": "Czech First League",
    "Den": "Danish Superliga",
    "Eng": "Premier League",
    "Fra": "Ligue 1",
    "Ger": "Bundesliga",
    "Esp": "La Liga",
    "Gre": "Greek Super League",
    "Hun": "Hungarian OTP Bank Liga",
    "Isr": "Israeli Premier League",      # added — ClubElo code confirmed
    "Ita": "Serie A",
    "Ned": "Eredivisie",
    "Nor": "Norwegian Eliteserien",        # added — was missing, caused ghost rows
    "Pol": "Polish Ekstraklasa",           # added — was missing, caused ghost rows
    "Por": "Primeira Liga",
    "Rom": "Romanian Liga I",              # added — was missing, caused ghost rows
    "Rus": "Russian Premier League",
    "Sco": "Scottish Premiership",
    "Srb": "Serbian SuperLiga",
    "Sui": "Swiss Super League",
    "Svn": "Slovenian PrvaLiga",           # added — was missing, caused ghost rows
    "Swe": "Swedish Allsvenskan",          # added — was missing, caused ghost rows
    "Tur": "Super Lig",
    "Ukr": "Ukrainian Premier League",
    # ── Second / lower divisions ───────────────────────────────────────────────
    "Eng2": "Championship",
    "Eng3": "League One",                  # added — was missing, caused ghost rows
    "Eng4": "League Two",                  # added — was missing, caused ghost rows
    "Ger2": "2. Bundesliga",
    "Esp2": "Segunda Division",            # fixed — was "La Liga 2", canonical is "Segunda Division"
    "Ita2": "Serie B",
    "Fra2": "Ligue 2",
    "Ned2": "Eerste Divisie",
    "Por2": "Liga Portugal 2",
    "Sco2": "Scottish Championship",
    # ── European / continental ─────────────────────────────────────────────────
    "UCL":  "UEFA Champions League",
    "UEL":  "UEFA Europa League",
    "UECL": "UEFA Conference League",      # fixed — canonical name matches soccerdata_sync.py
}

# Football-Data.co.uk uses Div codes in their CSVs (E0, D1, SP1, etc.)
# Map these to canonical league names to prevent ghost league rows.
FOOTBALL_DATA_DIV_MAP = {
    # England
    "E0":  "Premier League",
    "E1":  "Championship",
    "E2":  "League One",
    "E3":  "League Two",
    "EC":  "National League",          # Non-League / Conference
    # Germany
    "D1":  "Bundesliga",
    "D2":  "2. Bundesliga",
    # Spain
    "SP1": "La Liga",
    "SP2": "Segunda Division",
    # Italy
    "I1":  "Serie A",
    "I2":  "Serie B",
    # France
    "F1":  "Ligue 1",
    "F2":  "Ligue 2",
    # Netherlands
    "N1":  "Eredivisie",
    # Belgium
    "B1":  "Belgian Pro League",
    # Turkey
    "T1":  "Super Lig",
    # Scotland
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    # Greece
    "G1":  "Greek Super League",   # aligned with FBref canonical name
    # Portugal
    "P1":  "Primeira Liga",
    "P2":  "Liga Portugal 2",
    # Austria
    "A1":  "Austrian Bundesliga",
    # USA
    "USA": "MLS",
    # Brazil
    "BSA": "Brasileirao Serie A",
    # Argentina
    "ARG": "Argentine Primera Division",
    # Japan
    "J1":  "J1 League",
    # Catch-all codes seen in fixtures.csv
    "CONF": "UEFA Conference League",
    "UCL":  "UEFA Champions League",
    "UEL":  "UEFA Europa League",
}

# ─── Team name aliases ────────────────────────────────────────────────────────
# Maps Football-Data.co.uk, ClubElo, and Transfermarkt names → FBref canonical
# names.  Add entries whenever a new source uses a different spelling.
TEAM_NAME_ALIASES: dict[str, str] = {
    # ── England ───────────────────────────────────────────────────────────────
    "man city":                   "Manchester City",
    "manchester city":            "Manchester City",
    "man utd":                    "Manchester United",
    "man united":                 "Manchester United",
    "manchester united":          "Manchester United",
    "nott'm forest":              "Nottingham Forest",
    "nottm forest":               "Nottingham Forest",
    "nottingham forest":          "Nottingham Forest",
    "wolverhampton wanderers":    "Wolverhampton Wanderers",
    "wolverhampton":              "Wolverhampton Wanderers",
    "sheffield utd":              "Sheffield United",
    "sheffield united":           "Sheffield United",
    "luton town":                 "Luton",
    "west bromwich albion":       "West Brom",
    "west brom":                  "West Brom",
    "queens park rangers":        "QPR",
    "brighton & hove albion":     "Brighton",
    "brighton and hove albion":   "Brighton",
    "huddersfield town":          "Huddersfield",
    "stoke city":                 "Stoke",
    "swansea city":               "Swansea",
    "cardiff city":               "Cardiff",
    "wigan athletic":             "Wigan",
    "blackburn rovers":           "Blackburn",
    "bolton wanderers":           "Bolton",
    "ipswich town":               "Ipswich",
    "oxford united":              "Oxford Utd",
    "bristol city":               "Bristol City",
    "birmingham city":            "Birmingham City",
    "leicester city":             "Leicester City",
    "newcastle united":           "Newcastle United",
    "norwich city":               "Norwich",
    "watford fc":                 "Watford",
    "crystal palace":             "Crystal Palace",
    "west ham united":            "West Ham United",
    "west ham":                   "West Ham United",
    "tottenham hotspur":          "Tottenham Hotspur",
    "tottenham":                  "Tottenham Hotspur",
    "aston villa":                "Aston Villa",
    # ── Germany ───────────────────────────────────────────────────────────────
    "eintr frankfurt":            "Eintracht Frankfurt",
    "eintracht frankfurt":        "Eintracht Frankfurt",
    "hertha berlin":              "Hertha BSC",
    "hertha bsc":                 "Hertha BSC",
    "m'gladbach":                 "Mönchengladbach",
    "b. monchengladbach":         "Mönchengladbach",
    "borussia m.gladbach":        "Mönchengladbach",
    "bayer leverkusen":           "Leverkusen",
    "rb leipzig":                 "RB Leipzig",
    "vfb stuttgart":               "Stuttgart",
    "sc freiburg":                 "Freiburg",
    "fc augsburg":                 "Augsburg",
    "vfl bochum":                  "Bochum",
    "fc koln":                     "Köln",
    "1. fc koln":                  "Köln",
    "fc heidenheim":               "Heidenheim",
    "sv darmstadt 98":             "Darmstadt 98",
    "vfl wolfsburg":               "Wolfsburg",
    "borussia dortmund":           "Dortmund",
    "fc bayern munich":            "Bayern Munich",
    "bayern munich":               "Bayern Munich",
    "tsg hoffenheim":              "Hoffenheim",
    "1899 hoffenheim":             "Hoffenheim",
    "mainz 05":                    "Mainz 05",
    "1. fsv mainz 05":             "Mainz 05",
    "vfb Stuttgart":               "Stuttgart",
    # ── Spain ─────────────────────────────────────────────────────────────────
    "athletic bilbao":            "Athletic Club",
    "atletico madrid":            "Atlético Madrid",
    "real madrid":                "Real Madrid",
    "barcelona":                  "Barcelona",
    "espanol":                    "Espanyol",
    "deportivo alaves":           "Alavés",
    "alaves":                     "Alavés",
    "real sociedad":              "Real Sociedad",
    "real betis":                 "Betis",
    "betis":                      "Betis",
    "rayo vallecano":             "Rayo Vallecano",
    "villarreal":                 "Villarreal",
    "valencia cf":                "Valencia",
    "girona fc":                  "Girona",
    "celta vigo":                 "Celta Vigo",
    "cadiz cf":                   "Cádiz",
    "cadiz":                      "Cádiz",
    "getafe cf":                  "Getafe",
    "las palmas":                 "Las Palmas",
    # ── France ────────────────────────────────────────────────────────────────
    "psg":                        "Paris Saint-Germain",
    "paris sg":                   "Paris Saint-Germain",
    "paris saint-germain":        "Paris Saint-Germain",
    "paris saint germain":        "Paris Saint-Germain",
    "st etienne":                 "Saint-Étienne",
    "saint-etienne":              "Saint-Étienne",
    "olympique marseille":        "Marseille",
    "olympique lyonnais":         "Lyon",
    "as monaco":                  "Monaco",
    "stade rennes":               "Rennes",
    "stade brest":                "Brest",
    "rc lens":                    "Lens",
    "losc lille":                 "Lille",
    "ogc nice":                   "Nice",
    "rc strasbourg":              "Strasbourg",
    "toulouse fc":                "Toulouse",
    "montpellier hsc":            "Montpellier",
    "nantes":                     "Nantes",
    # ── Italy ─────────────────────────────────────────────────────────────────
    "milan":                      "Milan",
    "ac milan":                   "Milan",
    "inter milan":                "Inter",
    "internazionale":             "Inter",
    "fc internazionale":          "Inter",
    "hellas verona":              "Verona",
    "as roma":                    "Roma",
    "ss lazio":                   "Lazio",
    "ssc napoli":                 "Napoli",
    "atalanta bc":                "Atalanta",
    "juventus fc":                "Juventus",
    "acf fiorentina":             "Fiorentina",
    "torino fc":                  "Torino",
    "genoa cfc":                  "Genoa",
    "udinese calcio":             "Udinese",
    "cagliari calcio":            "Cagliari",
    "bologna fc":                 "Bologna",
    "monza":                      "Monza",
    "ac monza":                   "Monza",
    "empoli fc":                  "Empoli",
    "us lecce":                   "Lecce",
    "frosinone calcio":           "Frosinone",
    "salernitana":                "Salernitana",
    # ── Netherlands ───────────────────────────────────────────────────────────
    "psv eindhoven":              "PSV",
    "ajax amsterdam":             "Ajax",
    "afc ajax":                   "Ajax",
    "feyenoord rotterdam":        "Feyenoord",
    "az alkmaar":                 "AZ",
    "fc utrecht":                 "Utrecht",
    "fc twente":                  "Twente",
    "sc heerenveen":              "Heerenveen",
    # ── Turkey ────────────────────────────────────────────────────────────────
    "besiktas jk":                "Beşiktaş",
    "besiktas":                   "Beşiktaş",
    "galatasaray sk":             "Galatasaray",
    "fenerbahce sk":              "Fenerbahçe",
    "fenerbahce":                 "Fenerbahçe",
    "trabzonspor":                "Trabzonspor",
    # ── Belgium ───────────────────────────────────────────────────────────────
    "club brugge":                "Club Brugge",
    "kv mechelen":                "Mechelen",
    "rsc anderlecht":             "Anderlecht",
    # ── Portugal ──────────────────────────────────────────────────────────────
    "sl benfica":                 "Benfica",
    "fc porto":                   "Porto",
    "sporting cp":                "Sporting CP",
    "sc braga":                   "Braga",
    # ── Scotland ──────────────────────────────────────────────────────────────
    "celtic fc":                  "Celtic",
    "rangers fc":                 "Rangers",
    "heart of midlothian":        "Hearts",
    "hibernian fc":               "Hibernian",
    # ── FBref abbreviation reverse aliases ───────────────────────────────────
    # These handle FBref's own abbreviations arriving from fixtures/stats pages
    "nott'ham forest":            "Nottingham Forest",
    "newcastle utd":              "Newcastle United",
    "sheffield utd":              "Sheffield United",
    "sheffield wed":              "Sheffield Wednesday",
    "manchester utd":             "Manchester United",
    "leeds utd":                  "Leeds United",
    "wolves":                     "Wolverhampton Wanderers",
    "oxford utd":                 "Oxford United",
    "sunderland afc":             "Sunderland",
    "norwich city":               "Norwich City",
    "ipswich town":               "Ipswich Town",
    "luton town":                 "Luton Town",
    "swansea city":               "Swansea City",
    "cardiff city":               "Cardiff City",
    "stoke city":                 "Stoke City",
    "huddersfield town":          "Huddersfield Town",
}


import re

# Common suffixes/prefixes added by different data sources that should be
# stripped before matching. Longest/most specific first.
_TEAM_SUFFIXES = [
    r"\bfc$", r"\bafc$", r"\bsc$", r"\bfk$", r"\bsk$", r"\bjk$",
    r"\bcf$", r"\bac$", r"\bbc$", r"\bif$", r"\bbk$", r"\bsv$",
    r"\bvfl$", r"\bvfb$", r"\btsg$", r"\bfsv$",
]
_TEAM_PREFIXES = [
    r"^fc\b", r"^ac\b", r"^as\b", r"^ss\b", r"^us\b", r"^rc\b",
    r"^sc\b", r"^fk\b", r"^sk\b", r"^bk\b", r"^if\b", r"^rsc\b",
    r"^sl\b", r"^kv\b", r"^sv\b", r"^vfl\b", r"^vfb\b", r"^tsg\b",
    r"^ogc\b", r"^losc\b", r"^rb\b",
]

def _strip_affixes(name: str) -> str:
    """Remove common FC/AFC/SC prefixes and suffixes so that
    'Arsenal FC', 'FC Arsenal', and 'Arsenal' all reduce to 'arsenal'."""
    s = name.strip().lower()
    for pat in _TEAM_SUFFIXES:
        s = re.sub(pat, "", s).strip()
    for pat in _TEAM_PREFIXES:
        s = re.sub(pat, "", s).strip()
    return s.strip()


def _normalize_team_name(raw: str) -> str:
    """Map external data source names → FBref canonical names.

    Resolution order:
      1. Exact alias match on raw lowercased name
      2. Exact alias match on suffix/prefix-stripped name
      3. Return original (DB ILIKE lookup handles the rest)
    """
    stripped_raw = raw.strip()
    lower_raw    = stripped_raw.lower()

    # 1. Direct alias hit
    if lower_raw in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[lower_raw]

    # 2. Alias hit after stripping suffixes/prefixes
    stripped = _strip_affixes(stripped_raw)
    if stripped in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[stripped]

    # 3. No alias — return original so DB ILIKE can still find an exact match
    return stripped_raw


def _get_league(cur, name: str):
    """Resolve a league name (including ClubElo short codes and Football-Data Div
    codes) to a leagues.id.  Short codes are mapped to canonical full names to
    avoid duplicate league rows.
    """
    raw = name.strip()

    # Resolve ClubElo short code to its full canonical name (case-insensitive)
    canonical = (
        CLUBELO_LEAGUE_MAP.get(raw)
        or CLUBELO_LEAGUE_MAP.get(raw.capitalize())
        or CLUBELO_LEAGUE_MAP.get(raw.upper())
        # Resolve Football-Data.co.uk Div codes (E0, D1, SP1 …)
        or FOOTBALL_DATA_DIV_MAP.get(raw)
        or FOOTBALL_DATA_DIV_MAP.get(raw.upper())
        or raw
    )

    # 1. Exact match (case-insensitive)
    cur.execute("SELECT id FROM leagues WHERE name ILIKE %s LIMIT 1", (canonical,))
    res = cur.fetchone()
    if res:
        return res["id"]

    # 2. Partial keyword match: helps when FBref used a slightly different spelling
    #    e.g. canonical="Austrian Bundesliga", DB has "Austrian Football Bundesliga"
    keyword = canonical.split()[0]  # use the most distinctive first word
    if len(keyword) > 3:  # skip very short words like 'La', 'De'
        cur.execute("SELECT id FROM leagues WHERE name ILIKE %s LIMIT 1", (f"%{keyword}%",))
        res = cur.fetchone()
        if res:
            return res["id"]

    # 3. Insert with canonical full name (never insert a bare short code)
    cur.execute("INSERT INTO leagues (name) VALUES (%s) RETURNING id", (canonical,))
    return cur.fetchone()["id"]


def _get_team(cur, name: str, league_id: int, team_cache: dict, allow_create: bool = True):
    """Resolve a team name to a teams.id.

    Lookup order:
      1. Exact name match within the given league              (fastest, most correct)
      2. Exact name match across ALL leagues                   (handles UEFA cross-league teams
                                                                and teams that moved divisions)
      3. Short-name guard                                       (rejects garbage rows)
      4. Fuzzy match within the given league                   (handles minor spelling variants)
      5. Insert new team — ONLY when allow_create=True         (see below)

    allow_create=False is used for ClubElo data.
    ClubElo infers league from a 2-4 char country code (e.g. "ENG2" → Championship).
    If a team is not already in the DB, we must NOT create it under that inferred
    league — the team may not be in the Championship at all, or may later be scraped
    by the extension under a different (correct) league.  Creating it here would
    permanently anchor the team to the wrong league, corrupting all subsequent
    cross-league lookups (step 2 always returns the first match it finds).

    Callers that DO know the correct league (odds, injuries — which come from
    Football-Data Div codes that map 1-to-1 to canonical leagues) keep the
    default allow_create=True.
    """
    raw   = name.strip()
    clean = _normalize_team_name(raw)   # map aliases → FBref canonical names

    # In-memory deduplication during massive sync batches
    cache_key = (clean.lower(), league_id)
    if cache_key in team_cache:
        return team_cache[cache_key]

    # Also prepare a suffix-stripped version for broader matching
    # e.g. "Arsenal FC" → "arsenal", "FC Barcelona" → "barcelona"
    clean_stripped = _strip_affixes(clean)

    # 1. Exact match within league
    cur.execute("SELECT id FROM teams WHERE name ILIKE %s AND league_id = %s LIMIT 1", (clean, league_id))
    res = cur.fetchone()
    if res:
        team_cache[cache_key] = res["id"]
        return res["id"]

    # 1b. Suffix-stripped match within league
    #     Catches "Arsenal FC" vs "Arsenal", "FC Porto" vs "Porto" etc.
    if clean_stripped and clean_stripped != clean.lower():
        cur.execute("SELECT id FROM teams WHERE name ILIKE %s AND league_id = %s LIMIT 1", (f"%{clean_stripped}%", league_id))
        res = cur.fetchone()
        if res:
            team_cache[cache_key] = res["id"]
            return res["id"]

    # 2. Cross-league fallback: team may be registered under a domestic league
    #    but also appear in enrichment data for UCL / Europa / ClubElo etc.
    cur.execute("SELECT id FROM teams WHERE name ILIKE %s LIMIT 1", (clean,))
    res = cur.fetchone()
    if res:
        team_cache[cache_key] = res["id"]
        return res["id"]

    # 2b. Cross-league suffix-stripped fallback
    if clean_stripped and clean_stripped != clean.lower():
        cur.execute("SELECT id FROM teams WHERE name ILIKE %s LIMIT 1", (f"%{clean_stripped}%",))
        res = cur.fetchone()
        if res:
            team_cache[cache_key] = res["id"]
            return res["id"]

    # Team not found in DB at all from this point forward.
    # If the caller is ClubElo (allow_create=False), stop here — do not create a
    # ghost team under a ClubElo-inferred league that may be wrong.
    if not allow_create:
        logger.debug(
            "Team %r not in DB; skipping creation (allow_create=False, league_id=%s)",
            clean, league_id,
        )
        return None

    # 3. Genuinely new team — only create if name looks legitimate.
    #    Names shorter than 3 chars are almost certainly garbage from a
    #    malformed CSV row (e.g. empty HomeTeam field parsed as "  ").
    if len(clean) < 3:
        logger.warning("Skipping suspect team name (too short): %r", raw)
        return None

    # 4. Fuzzy match against existing teams in the league to prevent duplicates
    import difflib
    cur.execute("SELECT id, name FROM teams WHERE league_id = %s", (league_id,))
    league_teams = cur.fetchall()
    team_names = [t["name"] for t in league_teams]

    matches = difflib.get_close_matches(clean, team_names, n=1, cutoff=0.85)
    if matches:
        matched_name = matches[0]
        for t in league_teams:
            if t["name"] == matched_name:
                team_cache[cache_key] = t["id"]
                logger.debug("Fuzzy matched enrichment team %r to existing team %r", clean, matched_name)
                return t["id"]

    # 5. Insert brand-new team under the caller-supplied league
    cur.execute("INSERT INTO teams (name, league_id) VALUES (%s, %s) RETURNING id", (clean, league_id))
    new_id = cur.fetchone()["id"]
    team_cache[cache_key] = new_id
    logger.debug("Created new team: %r (league_id=%s)", clean, league_id)
    return new_id

def _find_match(cur, home_team_id, away_team_id, match_date: str):
    if not match_date:
        # Fallback if no date is found
        cur.execute("""
            SELECT id FROM matches 
            WHERE home_team_id = %s AND away_team_id = %s 
            ORDER BY match_date DESC LIMIT 1
        """, (home_team_id, away_team_id))
    else:
        # Odds files/FBref dates can differ by 1-2 days due to timezone disparities.
        # We search within a generous 3-day window of the declared match date to perfectly align historical matches.
        cur.execute("""
            SELECT id FROM matches 
            WHERE home_team_id = %s AND away_team_id = %s 
              AND match_date >= %s::date - INTERVAL '3 days' 
              AND match_date <= %s::date + INTERVAL '3 days'
            LIMIT 1
        """, (home_team_id, away_team_id, match_date, match_date))
        
    res = cur.fetchone()
    return res["id"] if res else None

def _parse_odds_date(d_str):
    if not d_str: return None
    try:
        if "/" in d_str:
            parts = d_str.split("/")
            if len(parts[2]) == 2: parts[2] = "20" + parts[2]
            return f"{parts[2]}-{parts[1]}-{parts[0]}"
        return d_str
    except:
        return None

@router.post("/enrichment")
def sync_enrichment(payload: SyncEnrichmentPayload, _admin: dict = Depends(require_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        default_league_id = _get_league(cur, payload.league)
        team_cache = {}
        
        # Helper to determine the correct league dynamically per row.
        # This prevents team-corruption when batch syncing multi-league CSV files (like fixtures.csv)
        def _resolve_league(row_obj):
            row_league = row_obj.get("League") or row_obj.get("league")
            if row_league and row_league != "Unknown League" and row_league != "Global Enrichment Sync":
                return _get_league(cur, row_league)
            return default_league_id
            
        odds_count = 0
        injuries_count = 0
        clubelo_count = 0
        
        # ── 1. Match Odds ───────────────────────────────────────────────────────────
        if payload.odds:
            for row in payload.odds:
                home_name = row.get("HomeTeam") or row.get("Home")
                away_name = row.get("AwayTeam") or row.get("Away")
                if not home_name or not away_name: continue
                
                l_id = _resolve_league(row)
                h_id = _get_team(cur, home_name, l_id, team_cache)
                a_id = _get_team(cur, away_name, l_id, team_cache)
                m_date = _parse_odds_date(row.get("Date"))
                m_id = _find_match(cur, h_id, a_id, m_date)
                
                if m_id:
                    h_odds = safe_num(row.get("B365H"))
                    d_odds = safe_num(row.get("B365D"))
                    a_odds = safe_num(row.get("B365A"))
                    cur.execute("""
                        INSERT INTO match_odds (match_id, b365_home_win, b365_draw, b365_away_win, raw_data)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (match_id) DO UPDATE SET
                            b365_home_win=EXCLUDED.b365_home_win,
                            b365_draw=EXCLUDED.b365_draw,
                            b365_away_win=EXCLUDED.b365_away_win,
                            raw_data=EXCLUDED.raw_data,
                            scraped_at=NOW()
                    """, (m_id, h_odds, d_odds, a_odds, json.dumps(row)))
                    odds_count += 1

        # ── 2. Player Injuries ──────────────────────────────────────────────────────
        if payload.injuries:
            for row in payload.injuries:
                club = row.get("Club")
                player = row.get("Player")
                if not club or not player: continue
                
                l_id = _resolve_league(row)
                t_id = _get_team(cur, club, l_id, team_cache)
                cur.execute("""
                    INSERT INTO player_injuries (team_id, player_name, injury_type, return_date, raw_data)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (team_id, player_name) DO UPDATE SET
                        injury_type=EXCLUDED.injury_type,
                        return_date=EXCLUDED.return_date,
                        raw_data=EXCLUDED.raw_data,
                        scraped_at=NOW()
                """, (t_id, player, row.get("Injury"), row.get("Return_Date"), json.dumps(row)))
                injuries_count += 1

        # ── 3. Team ClubElo ─────────────────────────────────────────────────────────
        if payload.clubelo:
            for row in payload.clubelo:
                # The extension wraps the original CSV row inside row["raw"].
                # All team/date fields live there; ELO probabilities are in row["odds"].
                raw_data = row.get("raw", row)
                home = row.get("home") or raw_data.get("Home") or raw_data.get("HomeTeam")
                away = row.get("away") or raw_data.get("Away") or raw_data.get("AwayTeam")
                target_date = row.get("date") or raw_data.get("Date")

                # ── ELO values ────────────────────────────────────────────────
                # The extension sends win-probability odds, not raw ELO ratings.
                # We store home-win probability as a proxy for relative strength.
                odds_block = row.get("odds", {})
                home_elo_val = safe_num(odds_block.get("home"))   # home win probability
                away_elo_val = safe_num(odds_block.get("away"))   # away win probability

                if not home or not away or not target_date:
                    continue

                # ── League resolution ─────────────────────────────────────────
                # ClubElo only sends "ENG" for ALL English divisions, so we CANNOT
                # trust the league field from the extension (it maps ENG → "Premier
                # League" for every English team regardless of actual division).
                # Instead we resolve the league by looking up each team's ACTUAL
                # league_id already stored in the teams table, which was populated
                # correctly by FBref scraping.  This prevents Championship / League
                # One / League Two teams from being misassigned to Premier League.
                for tm, elo_val in [(home, home_elo_val), (away, away_elo_val)]:
                    if not tm:
                        continue
                    clean_tm = _normalize_team_name(tm)

                    # Look up team by name across ALL leagues — do NOT filter by
                    # league_id here, because we don't know the correct league yet.
                    cur.execute(
                        "SELECT id, league_id FROM teams WHERE name ILIKE %s LIMIT 1",
                        (clean_tm,)
                    )
                    team_row = cur.fetchone()

                    if not team_row:
                        # Team not in DB at all — skip, never create ghost records
                        logger.debug("ClubElo: team %r not found in DB, skipping", clean_tm)
                        continue

                    t_id   = team_row["id"]
                    # Use the team's REAL league_id from the DB, not ClubElo's guess
                    real_league_id = team_row["league_id"]

                    cur.execute("""
                        INSERT INTO team_clubelo (team_id, elo_date, elo, raw_data)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (team_id, elo_date) DO UPDATE SET
                            elo=EXCLUDED.elo,
                            raw_data=EXCLUDED.raw_data,
                            scraped_at=NOW()
                    """, (t_id, target_date, elo_val, json.dumps(row)))
                    clubelo_count += 1
                
        conn.commit()
        return {
            "success": True, 
            "inserted": {
                "match_odds": odds_count,
                "player_injuries": injuries_count,
                "team_clubelo": clubelo_count
            }
        }
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Enrichment Sync Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
