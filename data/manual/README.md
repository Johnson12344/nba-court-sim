# Manual CSV input

Place `team_stats.csv` and `player_stats.csv` here. Run `python -m nba_sim validate-data` before simulation. These files are never silently generated.

Required `team_stats.csv` columns:

`team_id,team_abbreviation,team_name,season,games,minutes,points,fga,fgm,fg3a,fg3m,fta,ftm,oreb,dreb,assists,turnovers`

Required `player_stats.csv` columns:

`player_id,player_name,team_abbreviation,season,games,minutes,points,fga,fgm,fg3a,fg3m,fta,ftm,oreb,dreb,assists,turnovers,fouls`

Optional player columns include `starts`, `steals`, `blocks`, expected-minute overrides, and zone fields such as `rim_fga`, `rim_fgm`, through the five configured zones. Values must be observed aggregates, not projections disguised as observations.

Optional enrichment files:

- `shot_events.csv`: use NBA-style fields including `season,team_abbreviation,player_id,shot_zone_basic,shot_zone_area,shot_distance,loc_x,loc_y,shot_made_flag`.
- `play_by_play.csv`: use `season,game_id,team_tricode,person_id,is_field_goal,shot_result,description` for shooter and assisted-shot evidence.
- `rotations.csv`: use `season,game_id,team_tricode,person_id,minutes,position`; a non-empty position marks an observed starter.

The program labels these as manual CSV enrichment. Do not put projected or invented events in these files while describing them as observations.

