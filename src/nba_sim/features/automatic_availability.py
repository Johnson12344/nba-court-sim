from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone

from nba_sim.data.injury_reports import EASTERN, MAX_REPORT_AGE_MINUTES, report_time
from nba_sim.features.availability import _allocate
from nba_sim.market.edges import normalized_name
from nba_sim.market.events import age_minutes, utc_time
from nba_sim.market.odds_api import NBA_TEAM_NAMES
from nba_sim.models import TeamProfile


def player_key(name: str) -> str:
    # Official reports use 'Last [suffix], First'; keep suffixes to avoid
    # conflating different players. Accents/spacing/punctuation are normalized.
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first.strip()} {last.strip()}"
    return normalized_name(name)


def apply_official_report(teams: tuple[TeamProfile, TeamProfile], event: dict | None,
                          report: dict | None, *, now: datetime | None = None,
                          refresh_error: str | None = None) -> tuple[tuple[TeamProfile, TeamProfile], dict]:
    now = now or datetime.now(timezone.utc)
    review = {"mode": "automatic", "provider": "NBA official injury report", "reviewed": False,
              "reasons": [], "rows": [], "excluded": [], "uncertain": [],
              "note": "Not listed means no reported issue in a submitted team report, not confirmed participation. "
                      "Probable/questionable/doubtful players are modeled as playing. Minutes and starters remain estimates."}
    reasons = review["reasons"]
    if refresh_error:
        reasons.append(refresh_error)
    if not event or not utc_time(event.get("commence_time")):
        reasons.append("Automatic availability needs a dated upcoming fixture; refresh odds with analyze.")
        return teams, review
    start = utc_time(event["commence_time"])
    if (start <= now or event.get("home_team") != NBA_TEAM_NAMES.get(teams[0].abbreviation)
            or event.get("away_team") != NBA_TEAM_NAMES.get(teams[1].abbreviation)):
        reasons.append("Injury report cannot be applied: fixture is past or does not match these teams.")
        return teams, review
    if not report:
        reasons.append("No readable NBA injury report cached. Run analyze to download it automatically.")
        return teams, review
    review.update(source=report.get("source"), published_at=report.get("published_at"),
                  retrieved_at=report.get("retrieved_at"), event_id=event.get("id"))
    publication = report_time(report.get("source", ""))
    age = age_minutes(report.get("published_at"), now)
    checked_age = age_minutes(report.get("retrieved_at"), now)
    if (publication is None or publication != utc_time(report.get("published_at"))
            or age is None or checked_age is None or age > MAX_REPORT_AGE_MINUTES
            or checked_age > MAX_REPORT_AGE_MINUTES):
        reasons.append("NBA injury report is older than 12 hours or undated; availability was not applied. Rerun analyze.")
        return teams, review
    game_date = start.astimezone(EASTERN).date().isoformat()
    matchup = f"{teams[1].abbreviation}@{teams[0].abbreviation}"
    review["game_date_eastern"] = game_date
    updated = []
    for team in teams:
        sections = [s for s in report["teams"] if s["game_date"] == game_date
                    and s["matchup"] == matchup and s["team"] == team.abbreviation]
        if len(sections) != 1 or not sections[0]["submitted"]:
            reasons.append(f"{team.abbreviation}: NBA injury report not submitted/available for this fixture yet. Rerun analyze later.")
            review["rows"].extend({"team": team.abbreviation, "player_id": p.player_id,
                                   "player_name": p.name, "status": "unverified"} for p in team.players)
            updated.append(team)
            continue
        section = sections[0]
        model_keys = Counter(player_key(p.name) for p in team.players)
        report_keys = Counter(player_key(row["name"]) for row in section["players"])
        unmatched = [row["name"] for row in section["players"]
                     if model_keys[player_key(row["name"])] != 1 or report_keys[player_key(row["name"])] != 1]
        ambiguous = any(n != 1 for n in model_keys.values())
        if unmatched or ambiguous:
            detail = "; ".join(unmatched) if unmatched else "duplicate modeled names"
            reasons.append(f"{team.abbreviation}: injury report/rotation names could not be matched uniquely: {detail}.")
        indexed = {player_key(row["name"]): row for row in section["players"]
                   if model_keys[player_key(row["name"])] == 1 and report_keys[player_key(row["name"])] == 1}
        players, exclusions, uncertain = [], [], []
        team_rows = []
        for player in team.players:
            row = indexed.get(player_key(player.name))
            status = row["status"] if row else ("unverified" if unmatched or ambiguous else "not_listed")
            if status not in {"out", "available", "probable", "questionable", "doubtful", "not_listed", "unverified"}:
                status = "unverified"
                reasons.append(f"{team.abbreviation}: unrecognized injury status for {player.name}.")
            record = {"team": team.abbreviation, "player_id": player.player_id, "player_name": player.name,
                      "status": status, "source": report["source"], "published_at": report["published_at"],
                      "retrieved_at": report["retrieved_at"], "page": row.get("page") if row else None}
            team_rows.append(record)
            if status == "out":
                exclusions.append(f"{team.abbreviation}: {player.name}")
                continue
            if status in {"questionable", "doubtful", "probable"}:
                uncertain.append(f"{player.name} ({status})")
            players.append(player)
        if uncertain:
            reasons.append(f"{team.abbreviation}: uncertain availability, modeled as playing: {', '.join(uncertain)}.")
            review["uncertain"].extend(f"{team.abbreviation}: {name}" for name in uncertain)
        if len(players) < 5:
            raise ValueError(f"{team.abbreviation}: fewer than five modeled players remain after the official injury report. "
                             "Refresh current roster/statistics; a valid simulation cannot be formed.")
        targets = _allocate(players, {})
        adjusted = tuple(replace(p, expected_minutes=targets[p.player_id]) for p in players if targets[p.player_id] > 0)
        for row in team_rows:
            row["expected_minutes"] = targets.get(row["player_id"], 0.0)
        review["rows"].extend(team_rows)
        review["excluded"].extend(exclusions)
        updated.append(replace(team, players=adjusted, data_evidence={
            **team.data_evidence, "availability_source": report["source"], "fixed_minutes": {},
        }))
    review["reviewed"] = not reasons
    return tuple(updated), review
