"""Markets supported by both simulation samples and verified result settlement."""

PLAYER_COMPONENTS = {
    "player_points": ("points",),
    "player_rebounds": ("rebounds",),
    "player_assists": ("assists",),
    "player_threes": ("threes",),
    "player_points_rebounds": ("points", "rebounds"),
    "player_points_assists": ("points", "assists"),
    "player_rebounds_assists": ("rebounds", "assists"),
    "player_points_rebounds_assists": ("points", "rebounds", "assists"),
}
MARKETS = ("h2h", "spreads", "totals", "player_points", "player_rebounds", "player_assists",
           "totals_q1", "spreads_q1", "team_totals", *tuple(PLAYER_COMPONENTS)[3:])
EXTENDED_MARKETS = MARKETS[6:]
LABELS = {"h2h": "Match winner", "spreads": "Handicap", "totals": "Game total",
          "totals_q1": "1st quarter total", "spreads_q1": "1st quarter handicap",
          "team_totals": "Team total", "player_points": "Player points",
          "player_rebounds": "Player rebounds", "player_assists": "Player assists",
          "player_threes": "Player made threes", "player_points_rebounds": "Player points + rebounds",
          "player_points_assists": "Player points + assists", "player_rebounds_assists": "Player rebounds + assists",
          "player_points_rebounds_assists": "Player points + rebounds + assists"}
