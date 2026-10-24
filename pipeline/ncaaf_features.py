#!/usr/bin/env python3
"""
ncaaf_features.py — turn the cached ESPN game records into two flat tables.

  data/games.csv      one row per game: ids, teams, scores, market lines (close + open where available)
  data/teamgames.csv  two rows per game (one per offense): points, drives, plays, yards/play,
                      success rate, explosive rate, points per drive, tempo, turnovers, field position ...
                      The DEFENSIVE line of a team is the opponent's offensive row in the same game.

FBS membership is inferred from the scoreboard's conferenceId (groups=80 scoreboard lists FBS games;
an opponent with no conferenceId, or an FCS conference id, is FCS).
"""
import glob
import gzip
import json
import os
import re
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

SCRIMMAGE = {
    "Rush", "Pass Reception", "Pass Incompletion", "Sack", "Rushing Touchdown", "Passing Touchdown",
    "Pass Interception Return", "Interception", "Interception Return Touchdown",
    "Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Fumble Return Touchdown", "Safety", "Pass",
}
PASS_TYPES = {"Pass Reception", "Pass Incompletion", "Sack", "Passing Touchdown", "Pass Interception Return",
              "Interception", "Interception Return Touchdown", "Pass"}
TURNOVER_TYPES = {"Pass Interception Return", "Interception", "Interception Return Touchdown",
                  "Fumble Recovery (Opponent)", "Fumble Return Touchdown"}
TD_TYPES = {"Rushing Touchdown", "Passing Touchdown"}
DEAD_RESULTS = {"END OF HALF", "END OF GAME", "END OF 4TH QUARTER", "END OF REGULATION", "KICKOFF", "UNKNOWN"}

# closing-line provider priority (first present wins).  ESPN BET carries explicit open/close since 2024.
CLOSE_PRIORITY = ["ESPN BET", "consensus", "DraftKings", "Caesars Sportsbook", "Caesars", "Westgate",
                  "BOVADA.lv", "5Dimes.eu", "BETONLINE.ag", "MGM", "PointsBet", "Unibet", "SugarHouse",
                  "Caesars Sportsbook (New Jersey)", "Wynn", "CG Technology", "teamrankings", "numberfire",
                  "accuscore", "SportsBetting.ag", "BETNOW.eu", "SportsInteraction.com", "Sportsbook.com",
                  "BetUS.COM", "Betradar", "Fantasy911.com", "FanDuel", "PinnacleSports.com"]


def clock_to_sec(s):
    if not s or ":" not in s:
        return None
    m, sec = s.split(":")
    try:
        return int(m) * 60 + int(sec)
    except ValueError:
        return None


def success(down, dist, yds, td):
    if td:
        return True
    if down is None or dist is None or yds is None:
        return None
    if down == 1:
        return yds >= 0.5 * dist
    if down == 2:
        return yds >= 0.7 * dist
    return yds >= dist


def parse_details_spread(details, home_abbr, away_abbr):
    """'UGA -54.5' -> home-perspective spread (negative = home favored). Returns None if unparseable."""
    if not details:
        return None
    m = re.match(r"^\s*([A-Za-z&'.\- ]+?)\s+([-+]?\d+(?:\.\d+)?)\s*$", details)
    if not m:
        if details.strip().upper() in ("EVEN", "PK", "PICK"):
            return 0.0
        return None
    fav, line = m.group(1).strip(), float(m.group(2))
    if fav == home_abbr:
        return line
    if fav == away_abbr:
        return -line
    return None


def pick_lines(odds, home_abbr, away_abbr):
    """Return dict with close_total, close_spread (home perspective, negative = home favored),
    open_total, open_spread, provider used, and home/away moneylines."""
    by = {}
    for o in odds:
        by.setdefault(o.get("provider"), o)
    out = {"close_total": None, "close_spread": None, "open_total": None, "open_spread": None,
           "line_provider": None, "spread_provider": None, "home_ml": None, "away_ml": None, "n_books": len(by)}

    def ok_total(x, ref=None):
        # ESPN's open/close blocks sometimes carry the ODDS (-110) instead of the line; reject those
        if x is None or not (20 <= x <= 110):
            return False
        return ref is None or abs(x - ref) <= 12

    def ok_spread(x, ref=None):
        if x is None or abs(x) > 70:
            return False
        return ref is None or abs(abs(x) - abs(ref)) <= 10

    # --- closing total: first provider with a plausible number
    for p in CLOSE_PRIORITY:
        o = by.get(p)
        if not o:
            continue
        ou = o.get("overUnder")
        ct = o.get("close_total")
        tot = ct if ok_total(ct, ou if ok_total(ou) else None) else (ou if ok_total(ou) else None)
        if tot is None:
            continue
        out["close_total"], out["line_provider"] = tot, p
        if ok_total(o.get("open_total"), tot):
            out["open_total"] = o.get("open_total")
        break
    # --- closing spread (home perspective, negative = home favored)
    for p in CLOSE_PRIORITY:
        o = by.get(p)
        if not o:
            continue
        sp = o.get("spread")
        det = parse_details_spread(o.get("details"), home_abbr, away_abbr)
        # `spread` is home-perspective on ESPN; if `details` disagrees only in sign, trust details
        if det is not None and sp is not None and abs(abs(det) - abs(sp)) < 0.01 and det != sp:
            sp = det
        if not ok_spread(sp):
            sp = det if ok_spread(det) else None
        cs = o.get("close_home_spread")
        spr = cs if ok_spread(cs, sp) else sp
        if spr is None:
            continue
        out["close_spread"], out["spread_provider"] = spr, p
        out["home_ml"], out["away_ml"] = o.get("home_ml"), o.get("away_ml")
        if ok_spread(o.get("open_home_spread"), spr):
            out["open_spread"] = o.get("open_home_spread")
        break
    # legacy 'Opening' provider (2015-16)
    if by.get("Opening"):
        o = by["Opening"]
        if out["open_total"] is None and ok_total(o.get("overUnder"), out["close_total"]):
            out["open_total"] = o.get("overUnder")
        if out["open_spread"] is None:
            det = parse_details_spread(o.get("details"), home_abbr, away_abbr)
            cand = det if det is not None else o.get("spread")
            if ok_spread(cand, out["close_spread"]):
                out["open_spread"] = cand
    return out


def team_offense_rows(rec, sched):
    """Compute offensive stats for both teams from drives/plays/box. Returns dict keyed by team id."""
    home_id, away_id = sched["home"]["id"], sched["away"]["id"]
    ids = {home_id: "home", away_id: "away"}
    agg = {tid: dict(drives=0, plays=0, yards=0, succ=0, succ_n=0, expl15=0, expl20=0, pass_plays=0, sacks=0,
                     off_pts=0, opp_pts_on_drives=0, td_drives=0, fg_drives=0, to_plays=0, start_ytg=0.0,
                     start_n=0, elapsed=0, plays_timed=0, drives_ot=0, three_out=0, rz_trips=0,
                     rush_att=0, rush_yds=0, rush_succ=0, rush_succ_n=0, rush_x15=0,
                     pass_att=0, pass_yds=0, pass_succ=0, pass_succ_n=0, pass_x15=0)
           for tid in ids}
    prev_h, prev_a = 0, 0
    for d in rec["drives"]:
        tid = d.get("team")
        if tid not in agg:
            # OT/odd rows sometimes carry no team; try abbreviation
            ab = d.get("team_abbr")
            tid = home_id if ab == sched["home"]["abbr"] else away_id if ab == sched["away"]["abbr"] else None
            if tid is None:
                continue
        a = agg[tid]
        res = (d.get("result") or "").upper()
        plays = d.get("plays") or []
        is_ot = (d.get("q0") or 0) >= 5
        # running score at end of drive
        end_h, end_a = prev_h, prev_a
        for p in plays:
            if p[10] is not None and p[11] is not None:
                end_h, end_a = p[10], p[11]
        dh, da = end_h - prev_h, end_a - prev_a
        prev_h, prev_a = end_h, end_a
        n = d.get("n") or 0
        dead = res in DEAD_RESULTS and n <= 2
        if is_ot:
            a["drives_ot"] += 1
            continue  # ratings use regulation only
        if not dead and res not in ("KICKOFF", "UNKNOWN"):
            a["drives"] += 1
            if tid == home_id:
                a["off_pts"] += max(dh, 0); a["opp_pts_on_drives"] += max(da, 0)
            else:
                a["off_pts"] += max(da, 0); a["opp_pts_on_drives"] += max(dh, 0)
            if res.startswith("TD") or res == "TD":
                a["td_drives"] += 1
            if res == "FG":
                a["fg_drives"] += 1
            if n <= 3 and res == "PUNT":
                a["three_out"] += 1
            # starting field position from first scrimmage play
            for p in plays:
                if p[4] in SCRIMMAGE and p[2] is not None:
                    a["start_ytg"] += p[2]; a["start_n"] += 1
                    break
            el = clock_to_sec(d.get("elapsed"))
            if el is not None and n:
                a["elapsed"] += el; a["plays_timed"] += n
        # plays
        reached_rz = False
        for p in plays:
            down, dist, ytg, yds, ptype, scoring, is_to, is_pen, period = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]
            if ptype not in SCRIMMAGE:
                continue
            if ytg is not None and ytg <= 20:
                reached_rz = True
            a["plays"] += 1
            if ptype in PASS_TYPES:
                a["pass_plays"] += 1
            if ptype == "Sack":
                a["sacks"] += 1
            if ptype in TURNOVER_TYPES or is_to:
                a["to_plays"] += 1
                y = 0
            else:
                y = yds if yds is not None else 0
            a["yards"] += y
            td = ptype in TD_TYPES
            s = success(down, dist, y, td)
            if s is not None:
                a["succ"] += int(s); a["succ_n"] += 1
            if y >= 15:
                a["expl15"] += 1
            if y >= 20:
                a["expl20"] += 1
            # rush / pass split (sacks and interceptions count as dropbacks; fumbles as rushes)
            is_pass = ptype in PASS_TYPES
            k = "pass" if is_pass else "rush"
            a[f"{k}_att"] += 1; a[f"{k}_yds"] += y
            if s is not None:
                a[f"{k}_succ"] += int(s); a[f"{k}_succ_n"] += 1
            if y >= 15:
                a[f"{k}_x15"] += 1
        if reached_rz and not is_ot and not dead:
            a["rz_trips"] += 1
    return agg


def build():
    sched = {}
    for f in sorted(glob.glob(os.path.join(DATA, "schedule", "games_*.json"))):
        for g in json.load(open(f)):
            sched[g["id"]] = g
    # FBS team ids = any team that appears with a non-null conferenceId in an FBS-group scoreboard
    # (FCS opponents show conf None or an FCS conference).  Build per-season membership.
    fbs_confs = set()
    conf_counts = {}
    for g in sched.values():
        for side in ("home", "away"):
            c = g[side].get("conf")
            if c is not None:
                conf_counts[c] = conf_counts.get(c, 0) + 1
    # every conference with lots of appearances is FBS; FCS conferences appear rarely in FBS scoreboards.
    # Known FBS conference ids on ESPN: 1 ACC, 4 Big12, 5 Big Ten, 8 SEC, 9 Pac-12, 12 CUSA, 15 MAC, 17 MWC,
    # 18 FBS Independents, 37 Sun Belt, 151 AAC.  FCS: 20-33ish, 40, 48 etc.
    FBS_CONF = {"1", "4", "5", "8", "9", "12", "15", "17", "18", "37", "151"}
    games, tg = [], []
    missing = 0
    for gid, g in sched.items():
        path = os.path.join(DATA, "games", f"{gid}.json.gz")
        home, away = g["home"], g["away"]
        row = {
            "game_id": gid, "season": g["season"], "stype": g["stype"], "week": g["week"], "date": g["date"],
            "neutral": g["neutral"], "conf_game": g["conf_game"], "completed": g["completed"],
            "venue": g["venue"]["name"], "venue_state": g["venue"]["state"], "indoor": g["venue"]["indoor"],
            "home_id": home["id"], "home": home["abbr"], "home_name": home["name"], "home_conf": home["conf"],
            "away_id": away["id"], "away": away["abbr"], "away_name": away["name"], "away_conf": away["conf"],
            "home_fbs": str(home["conf"]) in FBS_CONF, "away_fbs": str(away["conf"]) in FBS_CONF,
            "home_pts": home["score"], "away_pts": away["score"],
            "home_rank": home.get("rank"), "away_rank": away.get("rank"),
        }
        if g.get("odds_live"):
            row["live_total"] = g["odds_live"].get("overUnder")
            row["live_spread"] = parse_details_spread(g["odds_live"].get("details"), home["abbr"], away["abbr"])
            if row["live_spread"] is None:
                row["live_spread"] = g["odds_live"].get("spread")
        if not os.path.exists(path):
            missing += int(g["completed"])
            row.update(close_total=None, close_spread=None, open_total=None, open_spread=None)
            games.append(row)
            continue
        rec = json.load(gzip.open(path, "rt"))
        row.update(pick_lines(rec.get("odds", []), home["abbr"], away["abbr"]))
        ls = rec["meta"].get("linescores") or {}
        for side in ("home", "away"):
            q = [float(x) for x in (ls.get(side) or []) if x not in (None, "", "-")]
            row[f"{side}_reg_pts"] = sum(q[:4]) if len(q) >= 4 else row[f"{side}_pts"]
            row[f"{side}_ot"] = len(q) > 4
        games.append(row)
        if not (rec.get("drives") and rec.get("box")):
            continue
        agg = team_offense_rows(rec, g)
        box = {v["id"]: v["stats"] for v in rec["box"].values()}
        for tid, side in ((home["id"], "home"), (away["id"], "away")):
            opp_side = "away" if side == "home" else "home"
            a = agg[tid]
            b = box.get(tid, {})
            def frac(s):
                try:
                    x, y = s.split("-"); return int(x), int(y)
                except Exception:  # noqa
                    return None, None
            td3, n3 = frac(b.get("thirdDownEff", ""))
            tg.append({
                "game_id": gid, "season": g["season"], "stype": g["stype"], "week": g["week"], "date": g["date"],
                "team_id": tid, "team": g[side]["abbr"], "team_name": g[side]["name"],
                "opp_id": g[opp_side]["id"], "opp": g[opp_side]["abbr"],
                "home": int(side == "home" and not g["neutral"]), "neutral": int(g["neutral"]),
                "team_fbs": row[f"{side}_fbs"], "opp_fbs": row[f"{opp_side}_fbs"],
                "pts": row[f"{side}_pts"], "pts_reg": row.get(f"{side}_reg_pts"),
                "opp_pts": row[f"{opp_side}_pts"], "opp_pts_reg": row.get(f"{opp_side}_reg_pts"),
                "off_pts": a["off_pts"], "drives": a["drives"], "plays": a["plays"], "yards": a["yards"],
                "succ": a["succ"], "succ_n": a["succ_n"], "expl15": a["expl15"], "expl20": a["expl20"],
                "pass_plays": a["pass_plays"], "sacks_taken": a["sacks"], "td_drives": a["td_drives"],
                "fg_drives": a["fg_drives"], "three_out": a["three_out"], "rz_trips": a["rz_trips"],
                "to_plays": a["to_plays"], "start_ytg": (a["start_ytg"] / a["start_n"]) if a["start_n"] else None,
                "rush_att": a["rush_att"], "rush_yds": a["rush_yds"], "rush_succ": a["rush_succ"], "rush_succ_n": a["rush_succ_n"], "rush_x15": a["rush_x15"],
                "pass_att": a["pass_att"], "pass_yds": a["pass_yds"], "pass_succ": a["pass_succ"], "pass_succ_n": a["pass_succ_n"], "pass_x15": a["pass_x15"],
                "sec_per_play": (a["elapsed"] / a["plays_timed"]) if a["plays_timed"] else None,
                "turnovers": int(b.get("turnovers", 0) or 0),
                "opp_turnovers": int(box.get(g[opp_side]["id"], {}).get("turnovers", 0) or 0),
                "box_yards": float(b.get("totalYards", 0) or 0),
                "first_downs": int(b.get("firstDowns", 0) or 0),
                "third_conv": td3, "third_att": n3,
                "pen_yards": (frac(b.get("totalPenaltiesYards", ""))[1]),
                "poss_sec": clock_to_sec(b.get("possessionTime")),
                "close_total": row.get("close_total"),
                # team's own closing spread: negative = this team favored
                "close_spread_team": (row.get("close_spread") if side == "home"
                                      else (-row["close_spread"] if row.get("close_spread") is not None else None)),
            })
    gdf = pd.DataFrame(games).sort_values(["date", "game_id"]).reset_index(drop=True)
    tdf = pd.DataFrame(tg).sort_values(["date", "game_id", "home"]).reset_index(drop=True)
    gdf.to_csv(os.path.join(DATA, "games.csv"), index=False)
    tdf.to_csv(os.path.join(DATA, "teamgames.csv"), index=False)
    print(f"games: {len(gdf)} ({missing} completed games missing cache) | team-games: {len(tdf)}")
    return gdf, tdf


if __name__ == "__main__":
    build()
