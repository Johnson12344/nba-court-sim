"""Public NBA injury reports, with publication time and fixture-specific coverage.

An absent player is only 'not listed' when their team's report was submitted.
This feed does not confirm starting lineups or future playing time.
"""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import requests
from pypdf import PdfReader

from nba_sim.market.events import age_minutes
from nba_sim.market.odds_api import NBA_TEAM_NAMES

EASTERN = ZoneInfo("America/New_York")
SCHEMA_VERSION = 1
MAX_REPORT_AGE_MINUTES = 720
_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nba-court-sim/0.5)", "Accept": "*/*"}
_FILE = re.compile(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})(AM|PM)\.pdf$")
_HEADER = re.compile(r"Injury\s+Report:\s*(\d{2}/\d{2}/\d{2})\s+(\d{2}:\d{2})\s*(AM|PM)")
_STATUS = re.compile(r"^(.+?,.+?)\s+(Out|Questionable|Doubtful|Probable|Available)(?:\s+|$)(.*)$")
_TEAM_NAMES = {**{name: code for code, name in NBA_TEAM_NAMES.items()}, "LA Clippers": "LAC"}
_TEAMS = [(re.compile(r"^" + re.escape(name).replace(r"\ ", r"\s*") + r"(?:\s+|$)"), code)
          for name, code in _TEAM_NAMES.items()]


class InjuryReportError(RuntimeError):
    pass


def report_time(url: str) -> datetime | None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "ak-static.cms.nba.com":
        return None
    match = _FILE.fullmatch(parsed.path.rsplit("/", 1)[-1])
    if not match or not parsed.path.startswith("/referee/injury/"):
        return None
    try:
        return datetime.strptime(" ".join(match.groups()), "%Y-%m-%d %I %M %p").replace(tzinfo=EASTERN).astimezone(timezone.utc)
    except ValueError:
        return None


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href" and value)


def latest_report_url(html: str, page_url: str, now: datetime) -> str:
    parser = _Links()
    parser.feed(html)
    candidates = []
    for link in parser.links:
        url = urljoin(page_url, link)
        published = report_time(url)
        if published is not None and published <= now:
            candidates.append((published, url))
    if not candidates:
        raise InjuryReportError("The NBA page has no usable published injury report yet.")
    return max(candidates)[1]


def parse_report_pages(pages: list[str], source: str) -> dict:
    """Parse layout text; reject incomplete/changed layouts instead of clearing players."""
    published = report_time(source)
    if published is None:
        raise InjuryReportError("Unrecognized official injury-report URL.")
    date = matchup = team = None
    sections: dict[tuple, dict] = {}
    for page_number, page in enumerate(pages, 1):
        lines = [re.sub(r"\s+", " ", line).strip() for line in page.splitlines() if line.strip()]
        headers = [(i, _HEADER.search(line)) for i, line in enumerate(lines) if _HEADER.search(line)]
        if len(headers) != 1:
            raise InjuryReportError("Injury report page header could not be read.")
        index, header = headers[0]
        header_time = datetime.strptime(" ".join(header.groups()), "%m/%d/%y %I:%M %p").replace(tzinfo=EASTERN)
        if header_time.astimezone(timezone.utc) != published:
            raise InjuryReportError("Injury report publication time does not match its link.")
        # Some NBA PDFs use an inverted vertical transform; pypdf layout then
        # yields bottom-to-top lines. Header/footer order identifies that case.
        if index > len(lines) // 2:
            lines.reverse()
        for line in lines:
            if _HEADER.search(line) or re.match(r"^Page \d+ of \d+$", line) or line.startswith("Game Date"):
                continue
            found = re.match(r"^(\d{2}/\d{2}/\d{4})\s+", line)
            if found:
                date = datetime.strptime(found[1], "%m/%d/%Y").date().isoformat()
                matchup = team = None
                line = line[found.end():]
            line = re.sub(r"^\d{2}:\d{2}\s*\(ET\)\s*", "", line)
            found = re.match(r"^([A-Z]{3}@[A-Z]{3})\s+", line)
            if found:
                matchup, team = found[1], None
                line = line[found.end():]
            new_team = False
            for pattern, code in _TEAMS:
                found = pattern.match(line)
                if found:
                    team, line, new_team = code, line[found.end():].strip(), True
                    if not date or not matchup or team not in matchup.split("@"):
                        raise InjuryReportError("Injury report team could not be matched to a game.")
                    break
            if not date or not matchup or not team:
                raise InjuryReportError("Injury report game context could not be read.")
            key = (date, matchup, team)
            section = sections.setdefault(key, {"game_date": date, "matchup": matchup, "team": team,
                                                 "submitted": False, "players": []})
            if line == "NOT YET SUBMITTED":
                if section["submitted"]:
                    raise InjuryReportError("Conflicting injury report submission status.")
                section["pending"] = True
                continue
            row = _STATUS.match(line)
            if row:
                if section.get("pending"):
                    raise InjuryReportError("Conflicting injury report submission status.")
                name, status, reason = row.groups()
                if any(p["name"] == name for p in section["players"]):
                    raise InjuryReportError("Duplicate player in injury report.")
                section["players"].append({"name": name, "status": status.lower(), "reason": reason,
                                            "page": page_number})
                section["submitted"] = True
            elif line in {"None", "None.", "No injuries to report"}:
                if section["players"] or section.get("pending"):
                    raise InjuryReportError("Conflicting injury report submission status.")
                section["submitted"] = True
            elif new_team or "," in line or re.search(r"\b(Out|Questionable|Doubtful|Probable|Available)\b", line):
                raise InjuryReportError("An injury report player row could not be read safely.")
            # Remaining lines contain only wrapped reasons, not participation data.
    if not sections or not any(s["submitted"] or s.get("pending") for s in sections.values()):
        raise InjuryReportError("No team reports could be read.")
    return {"schema_version": SCHEMA_VERSION, "provider": "NBA official injury report", "source": source,
            "published_at": published.isoformat(), "teams": list(sections.values())}


def parse_report_pdf(content: bytes, source: str) -> dict:
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False)
                 for page in reader.pages]
        return parse_report_pages(pages, source)
    except InjuryReportError:
        raise
    except Exception as exc:
        raise InjuryReportError("The NBA injury-report PDF could not be read.") from exc


def load_injury_report(processed: Path, season: str) -> dict | None:
    if not re.fullmatch(r"\d{4}-\d{2}", season):
        raise ValueError("Season must use YYYY-YY format")
    try:
        report = json.loads((processed / f"injuries_{season}.json").read_text(encoding="utf-8"))
        if (report.get("schema_version") == SCHEMA_VERSION and report.get("season") == season
                and isinstance(report.get("source"), str) and isinstance(report.get("teams"), list)
                and all(isinstance(s, dict) and all(isinstance(s.get(k), str) for k in ("team", "game_date", "matchup"))
                        and isinstance(s.get("submitted"), bool) and isinstance(s.get("players"), list)
                        and all(isinstance(p, dict) and isinstance(p.get("name"), str)
                                and isinstance(p.get("status"), str) for p in s["players"])
                        for s in report["teams"])):
            return report
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _download_report(season: str, now: datetime) -> tuple[str, bytes]:
    page_url = f"https://official.nba.com/nba-injury-report-{season}-season/"
    try:
        page = requests.get(page_url, headers=_REQUEST_HEADERS, timeout=20)
        page.raise_for_status()
        source = latest_report_url(page.text, page_url, now)
        response = requests.get(source, headers=_REQUEST_HEADERS, timeout=20)
        response.raise_for_status()
        return source, response.content
    except (requests.RequestException, InjuryReportError):
        # The public index is sometimes unavailable while the public PDF host
        # still works. Probe a bounded set of the same published file names.
        eastern_now = now.astimezone(EASTERN)
        year = eastern_now.year - (eastern_now.month < 7)
        current_season = f"{year}-{str(year + 1)[-2:]}"
        if season == current_season:
            latest = eastern_now.replace(minute=(eastern_now.minute // 15) * 15, second=0, microsecond=0)
            for offset in range(8):
                stamp = latest - timedelta(minutes=15 * offset)
                source = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_" + stamp.strftime("%Y-%m-%d_%I_%M%p.pdf")
                try:
                    response = requests.get(source, headers=_REQUEST_HEADERS, timeout=15)
                    if response.status_code in {403, 404}:
                        continue  # A not-yet-published CDN object can return either.
                    response.raise_for_status()
                    return source, response.content
                except requests.RequestException:
                    break  # Do not repeatedly retry a timeout or rate limit.
        raise InjuryReportError("NBA injury reports are temporarily unavailable; rerun analyze to retry.") from None


def refresh_injury_report(raw: Path, processed: Path, season: str, *, force: bool = False,
                         now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    cached = load_injury_report(processed, season)
    checked_age = age_minutes(cached.get("retrieved_at"), now) if cached else None
    published_age = age_minutes(cached.get("published_at"), now) if cached else None
    if (not force and checked_age is not None and checked_age <= 5
            and published_age is not None and published_age <= MAX_REPORT_AGE_MINUTES):
        return cached
    source, content = _download_report(season, now)
    report = parse_report_pdf(content, source)
    report.update(retrieved_at=now.isoformat(), season=season)
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    (raw / urlparse(source).path.rsplit("/", 1)[-1]).write_bytes(content)
    # Replace only after downloading and validating the entire report.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=processed, delete=False) as handle:
        json.dump(report, handle, indent=2)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, processed / f"injuries_{season}.json")
    finally:
        temporary.unlink(missing_ok=True)
    return report
