#!/usr/bin/env python3
"""
ncaaf_pull.py — pull every FBS college-football game from ESPN (keyless).

Outputs (under DATA_DIR):
  schedule/games_<season>.json   one compact dict per event from the weekly scoreboards
  games/<event_id>.json.gz       compact per-game record: box score, drives+plays, odds archive

Usage:
  python3 ncaaf_pull.py schedule 2015 2026        # build schedules for seasons 2015..2026
  python3 ncaaf_pull.py games 2015 2026           # fetch summary+odds for every completed game
  python3 ncaaf_pull.py all 2015 2026

Safe to re-run: everything is cached on disk.
"""
import gzip
import json
import os
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DATA_DIR = os.environ.get("NCAAF_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
SCHED_DIR = os.path.join(DATA_DIR, "schedule")
GAME_DIR = os.path.join(DATA_DIR, "games")
os.makedirs(SCHED_DIR, exist_ok=True)
os.makedirs(GAME_DIR, exist_ok=True)

SITE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
_session = requests.Session()  # default python-requests UA; browser-style UAs get 403 through the proxy


def get_json(url, tries=6, timeout=40):
    last = None
    for i in range(tries):
        try:
            r = _session.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (404,):
                return None
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa
            last = repr(e)
        time.sleep(min(30, (1.5 ** i) + random.random()))
    raise RuntimeError(f"failed {url}: {last}")


# ----------------------------------------------------------------------------- schedule

def season_calendar(season):
    d = get_json(f"{SITE}/scoreboard?dates={season}&groups=80&limit=1")
    cal = d["leagues"][0].get("calendar", [])
    out = []
    for block in cal:
        stype = int(block.get("value"))
        for e in block.get("entries", []):
            out.append((stype, int(e["value"]), e.get("label"), e.get("startDate"), e.get("endDate")))
    return out


def compact_event(e, season, stype, week):
    c = e["competitions"][0]
    comps = {}
    for t in c["competitors"]:
        tm = t["team"]
        comps[t["homeAway"]] = {
            "id": tm["id"],
            "abbr": tm.get("abbreviation"),
            "name": tm.get("displayName"),
            "short": tm.get("shortDisplayName"),
            "conf": tm.get("conferenceId"),
            "score": (int(t["score"]) if str(t.get("score", "")).lstrip("-").isdigit() else None),
            "rank": t.get("curatedRank", {}).get("current"),
            "winner": t.get("winner"),
            "records": [r.get("summary") for r in t.get("records", [])],
            "linescores": [ls.get("value") for ls in t.get("linescores", [])],
        }
    odds = None
    if c.get("odds"):
        o = c["odds"][0]
        odds = {"provider": o.get("provider", {}).get("name"), "details": o.get("details"),
                "overUnder": o.get("overUnder"), "spread": o.get("spread")}
    st = c.get("status", {}).get("type", {})
    v = c.get("venue", {}) or {}
    return {
        "id": e["id"],
        "season": season,
        "stype": stype,
        "week": week,
        "date": c.get("date") or e.get("date"),
        "name": e.get("name"),
        "neutral": bool(c.get("neutralSite")),
        "conf_game": bool(c.get("conferenceCompetition")),
        "venue": {"id": v.get("id"), "name": v.get("fullName"),
                  "city": (v.get("address") or {}).get("city"), "state": (v.get("address") or {}).get("state"),
                  "indoor": v.get("indoor")},
        "attendance": c.get("attendance"),
        "completed": bool(st.get("completed")),
        "status": st.get("name"),
        "home": comps.get("home"),
        "away": comps.get("away"),
        "odds_live": odds,
    }


def build_schedule(season):
    path = os.path.join(SCHED_DIR, f"games_{season}.json")
    seen = {}
    cal = season_calendar(season)
    for stype, week, label, sd, ed in cal:
        if stype not in (2, 3):
            continue
        url = f"{SITE}/scoreboard?dates={season}&seasontype={stype}&week={week}&groups=80&limit=400"
        d = get_json(url)
        evs = d.get("events", []) if d else []
        for e in evs:
            try:
                rec = compact_event(e, season, stype, week)
            except Exception as ex:  # noqa
                print(f"  skip event {e.get('id')}: {ex}")
                continue
            seen[rec["id"]] = rec
        print(f"  {season} type{stype} week{week:>2} ({label}): {len(evs)} events, cum {len(seen)}")
    games = sorted(seen.values(), key=lambda g: (g["date"] or "", g["id"]))
    with open(path, "w") as f:
        json.dump(games, f)
    print(f"schedule {season}: {len(games)} games -> {path}")
    return games


# ----------------------------------------------------------------------------- per game

def _num(x):
    try:
        return float(x)
    except Exception:  # noqa
        return None


def compact_odds_item(it):
    def tot(block):
        try:
            return _num(block["total"]["american"])
        except Exception:  # noqa
            return None

    def spr(block):
        try:
            return _num(block["pointSpread"]["american"])
        except Exception:  # noqa
            return None

    h = it.get("homeTeamOdds") or {}
    a = it.get("awayTeamOdds") or {}
    return {
        "provider": (it.get("provider") or {}).get("name"),
        "priority": (it.get("provider") or {}).get("priority"),
        "details": it.get("details"),
        "spread": it.get("spread"),
        "overUnder": it.get("overUnder"),
        "overOdds": it.get("overOdds"),
        "underOdds": it.get("underOdds"),
        "home_fav": h.get("favorite"),
        "home_ml": h.get("moneyLine"),
        "away_ml": a.get("moneyLine"),
        "home_spread_odds": h.get("spreadOdds"),
        "away_spread_odds": a.get("spreadOdds"),
        "open_total": tot(it.get("open") or {}),
        "close_total": tot(it.get("close") or {}),
        "current_total": tot(it.get("current") or {}),
        "open_home_spread": spr(h.get("open") or {}),
        "close_home_spread": spr(h.get("close") or {}),
        "current_home_spread": spr(h.get("current") or {}),
    }


def compact_play(p):
    s = p.get("start") or {}
    return [
        s.get("down"), s.get("distance"), s.get("yardsToEndzone"),
        p.get("statYardage"),
        (p.get("type") or {}).get("text"),
        bool(p.get("scoringPlay")), bool(p.get("isTurnover")), bool(p.get("isPenalty")),
        (p.get("period") or {}).get("number"), (p.get("clock") or {}).get("displayValue"),
        p.get("homeScore"), p.get("awayScore"),
        (s.get("team") or {}).get("id"),
    ]


def compact_drive(d):
    st = d.get("start") or {}
    en = d.get("end") or {}
    te = (d.get("timeElapsed") or {}).get("displayValue")
    plays = [compact_play(p) for p in d.get("plays", [])]
    return {
        "team": (d.get("team") or {}).get("id"),
        "team_abbr": (d.get("team") or {}).get("abbreviation"),
        "q0": (st.get("period") or {}).get("number"), "c0": (st.get("clock") or {}).get("displayValue"),
        "yl0": st.get("yardLine"),
        "q1": (en.get("period") or {}).get("number"), "c1": (en.get("clock") or {}).get("displayValue"),
        "yl1": en.get("yardLine"),
        "yards": d.get("yards"), "n": d.get("offensivePlays"), "result": d.get("result"),
        "score": bool(d.get("isScore")), "elapsed": te,
        "plays": plays,
    }


def fetch_game(gid):
    out = os.path.join(GAME_DIR, f"{gid}.json.gz")
    if os.path.exists(out):
        return "cached"
    summ = get_json(f"{SITE}/summary?event={gid}")
    odds = get_json(f"{CORE}/events/{gid}/competitions/{gid}/odds")
    rec = {"id": gid, "box": {}, "drives": [], "odds": [], "meta": {}}
    if summ:
        bs = summ.get("boxscore", {})
        for t in bs.get("teams", []):
            rec["box"][t.get("homeAway")] = {
                "id": t["team"]["id"],
                "abbr": t["team"].get("abbreviation"),
                "stats": {s.get("name"): s.get("displayValue") for s in t.get("statistics", [])},
            }
        drives = (summ.get("drives") or {}).get("previous", []) or []
        rec["drives"] = [compact_drive(d) for d in drives]
        hdr = summ.get("header", {})
        comp = (hdr.get("competitions") or [{}])[0]
        gi = summ.get("gameInfo", {}) or {}
        rec["meta"] = {
            "season": (hdr.get("season") or {}).get("year"),
            "stype": (hdr.get("season") or {}).get("type"),
            "week": hdr.get("week"),
            "neutral": comp.get("neutralSite"),
            "conf_game": comp.get("conferenceCompetition"),
            "attendance": gi.get("attendance"),
            "weather": gi.get("weather"),
            "venue": {k: (gi.get("venue") or {}).get(k) for k in ("id", "fullName", "indoor", "grass")},
            "home_id": next((t["team"]["id"] for t in comp.get("competitors", []) if t.get("homeAway") == "home"), None),
            "away_id": next((t["team"]["id"] for t in comp.get("competitors", []) if t.get("homeAway") == "away"), None),
            "home_score": next((t.get("score") for t in comp.get("competitors", []) if t.get("homeAway") == "home"), None),
            "away_score": next((t.get("score") for t in comp.get("competitors", []) if t.get("homeAway") == "away"), None),
            "linescores": {t.get("homeAway"): [ls.get("displayValue") for ls in t.get("linescores", [])]
                           for t in comp.get("competitors", [])},
        }
    if odds:
        rec["odds"] = [compact_odds_item(it) for it in odds.get("items", [])]
    with gzip.open(out, "wt") as f:
        json.dump(rec, f)
    return "fetched"


def fetch_games(seasons, workers=10):
    ids = []
    for s in seasons:
        path = os.path.join(SCHED_DIR, f"games_{s}.json")
        if not os.path.exists(path):
            build_schedule(s)
        games = json.load(open(path))
        ids += [g["id"] for g in games if g.get("completed")]
    todo = [g for g in ids if not os.path.exists(os.path.join(GAME_DIR, f"{g}.json.gz"))]
    print(f"{len(ids)} completed games, {len(todo)} to fetch")
    done = 0
    errs = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_game, g): g for g in todo}
        for fu in as_completed(futs):
            done += 1
            try:
                fu.result()
            except Exception as e:  # noqa
                errs += 1
                print(f"  ERR {futs[fu]}: {e}")
            if done % 200 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(todo)} ({errs} errors) {el/60:.1f} min, {done/el:.1f}/s", flush=True)
    print(f"done: {done} fetched, {errs} errors, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    mode = sys.argv[1]
    s0, s1 = int(sys.argv[2]), int(sys.argv[3])
    seasons = list(range(s0, s1 + 1))
    if mode in ("schedule", "all"):
        for s in seasons:
            build_schedule(s)
    if mode in ("games", "all"):
        fetch_games(seasons)
