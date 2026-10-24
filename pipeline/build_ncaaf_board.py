#!/usr/bin/env python3
"""
build_ncaaf_board.py — compute this week's NCAAF matchup board data.

  python3 build_ncaaf_board.py [season] [week]      (defaults: current season, next week with unplayed games)

Steps
  1. refresh the current season's schedule + completed game logs from ESPN (ncaaf_pull.py)
  2. rebuild teamgames.csv / features.csv (ncaaf_features.py, ncaaf_ratings.py)
  3. fit the SCORE coefficients on every completed FBS-vs-FBS game 2015..last full season
  4. fit as-of ratings for the target week from this season's completed games (+ last season's prior)
  5. pull the live scoreboard for the target week for posted spreads/totals
  6. write ncaaf_board_data.json (consumed by ncaaf_board_template.html via render)

Rating hyperparameters (chosen by CV on 2015-2024): lam=2, prior carryover c=0.75, recency rho=0.9.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
sys.path.insert(0, BASE)
from ncaaf_ratings import load_teamgames, RatingEngine, STATS, add_raw_baselines  # noqa: E402
from ncaaf_backtest import load as load_eval, fit_ols, FEATURE_SETS, SCORE_MODEL  # noqa: E402

LAM, C, RHO = 2.0, 0.75, 0.9
SITE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
COLS = FEATURE_SETS[SCORE_MODEL]


def refresh(season):
    subprocess.run([sys.executable, os.path.join(BASE, "ncaaf_pull.py"), "all", str(season), str(season)], check=True)
    subprocess.run([sys.executable, os.path.join(BASE, "ncaaf_features.py")], check=True)
    subprocess.run([sys.executable, os.path.join(BASE, "ncaaf_ratings.py"), str(LAM), str(C), str(RHO), "features"], check=True)


def season_priors(t, upto_season):
    """Chain full-season fits to get each team's final ratings entering `upto_season`."""
    prior_O, prior_D = {}, {}
    eng = RatingEngine(lam=LAM, c=C, rho=1.0)
    for s in sorted(t.season.unique()):
        if s >= upto_season:
            break
        fin = eng.fit(t[t.season == s], prior_O, prior_D, 99)
        prior_O, prior_D = fin["O"], fin["D"]
        eng.prior_h = fin["h"]
    return prior_O, prior_D, eng.prior_h


def live_scoreboard(season, week):
    d = requests.get(f"{SITE}/scoreboard?dates={season}&seasontype=2&week={week}&groups=80&limit=400", timeout=40).json()
    out = {}
    for e in d.get("events", []):
        c = e["competitions"][0]
        o = (c.get("odds") or [{}])[0]
        home = next(t for t in c["competitors"] if t["homeAway"] == "home")
        away = next(t for t in c["competitors"] if t["homeAway"] == "away")
        spread = None
        det = o.get("details")
        if det and det not in ("EVEN", "PK"):
            parts = det.rsplit(" ", 1)
            try:
                line = float(parts[1])
                fav = parts[0].strip()
                if fav == home["team"].get("abbreviation"):
                    spread = line
                elif fav == away["team"].get("abbreviation"):
                    spread = -line
            except (ValueError, IndexError):
                spread = None
        if spread is None and o.get("spread") is not None:
            spread = o.get("spread")
        out[e["id"]] = {"total": o.get("overUnder"), "spread": spread, "provider": (o.get("provider") or {}).get("name"),
                        "details": det, "status": c.get("status", {}).get("type", {}).get("name"),
                        "date": c.get("date"), "broadcast": (c.get("broadcasts") or [{}])[0].get("names", [None])[0] if c.get("broadcasts") else None,
                        "home_rank": home.get("curatedRank", {}).get("current"), "away_rank": away.get("curatedRank", {}).get("current"),
                        "home_record": (home.get("records") or [{}])[0].get("summary"), "away_record": (away.get("records") or [{}])[0].get("summary")}
    return out


def main(season=None, week=None, do_refresh=True):
    now = datetime.now(timezone.utc)
    season = season or (now.year if now.month >= 7 else now.year - 1)
    if do_refresh:
        refresh(season)
    games = pd.read_csv(os.path.join(DATA, "games.csv"))
    cur = games[(games.season == season) & (games.stype == 2)]
    if week is None:
        unplayed = cur[~cur.completed]
        week = int(unplayed.week.min()) if len(unplayed) else int(cur.week.max())
    print(f"target: {season} week {week}")

    # ---- SCORE coefficients: fit on all completed FBS-vs-FBS games before this season
    f, _ = load_eval("features")
    f = f[f.season < season]
    beta = fit_ols(f, COLS)
    coef = dict(zip(["const"] + COLS, [float(b) for b in beta]))
    print("coef:", {k: round(v, 3) for k, v in coef.items()})
    # spread of each input in the fitting sample (for the sliders' 'points per SD' display)
    sds = {c: float(f[c].std()) for c in COLS}

    # ---- as-of ratings for this season's target week
    t = load_teamgames()
    prior_O, prior_D, prior_h = season_priors(t, season)
    ts = t[(t.season == season) & (t.wk < week)]
    eng = RatingEngine(lam=LAM, c=C, rho=RHO)
    eng.prior_h = prior_h
    if len(ts):
        fit = eng.fit(ts, prior_O, prior_D, week)
        mu, h = fit["mu"], fit["h"]
        O, D = fit["O"], fit["D"]
    else:
        prev = t[t.season == season - 1]
        mu = prev[STATS].mean().values
        h = np.zeros(len(STATS))
        O = {k: C * np.array(v) for k, v in prior_O.items()}
        D = {k: C * np.array(v) for k, v in prior_D.items()}
    si = {s: i for i, s in enumerate(STATS)}
    games_played = ts.groupby("team_key").size().to_dict() if len(ts) else {}

    # raw season-to-date per team (this season)
    raw = {}
    if len(ts):
        g = ts.groupby("team_key")
        raw_df = pd.DataFrame({
            "ppg": g.pts.mean(), "pa": g.opp_pts.mean(), "ypp": g.ypp.mean(), "sr": g.sr.mean(),
            "xr": g.xr.mean(), "plays": g.plays.mean(), "spp": g.spp.mean(), "ppd": g.ppd.mean(),
            "to": g.turnovers.mean(), "n": g.size()})
        raw = raw_df.to_dict("index")
    # opponent-allowed raw (from the opponents' offensive rows)
    raw_allowed = {}
    if len(ts):
        g = ts.groupby("opp_key")
        raw_allowed = pd.DataFrame({"ypp_a": g.ypp.mean(), "sr_a": g.sr.mean(), "xr_a": g.xr.mean(),
                                    "ppd_a": g.ppd.mean(), "plays_a": g.plays.mean()}).to_dict("index")

    # team directory (names/abbr) from the schedule
    tdir = {}
    for _, r in games[games.season >= season - 1].iterrows():
        for side in ("home", "away"):
            if r[f"{side}_fbs"]:
                tdir[str(r[f"{side}_id"])] = {"abbr": r[side], "name": r[f"{side}_name"], "conf": r[f"{side}_conf"]}

    def team_block(key):
        o = O.get(key); d = D.get(key)
        if o is None:
            o = C * np.array(prior_O.get(key, np.zeros(len(STATS))))
        if d is None:
            d = C * np.array(prior_D.get(key, np.zeros(len(STATS))))
        rw = raw.get(key, {}); ra = raw_allowed.get(key, {})
        return {
            "id": key, "abbr": tdir.get(key, {}).get("abbr", key), "name": tdir.get(key, {}).get("name", key),
            "conf": tdir.get(key, {}).get("conf"), "gp": int(games_played.get(key, 0)),
            "O_pts": float(o[si["pts"]]), "D_pts": float(d[si["pts"]]),
            "O_sr": float(o[si["sr"]]), "D_sr": float(d[si["sr"]]),
            "O_xr": float(o[si["xr"]]), "D_xr": float(d[si["xr"]]),
            "O_ypp": float(o[si["ypp"]]), "D_ypp": float(d[si["ypp"]]),
            "O_ppd": float(o[si["ppd"]]), "D_ppd": float(d[si["ppd"]]),
            "O_plays": float(o[si["plays"]]), "D_plays": float(d[si["plays"]]),
            "O_to": float(o[si["to_rate"]]), "D_to": float(d[si["to_rate"]]),
            "O_rush": float(o[si["rush_ypc"]]), "D_rush": float(d[si["rush_ypc"]]),
            "O_pass": float(o[si["pass_ypa"]]), "D_pass": float(d[si["pass_ypa"]]),
            "O_prate": float(o[si["pass_rate"]]), "D_prate": float(d[si["pass_rate"]]),
            "raw": {k: (None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)) for k, v in {**rw, **ra}.items()},
        }

    def project(team, opp, home):
        rmargin = (team["O_pts"] - team["D_pts"]) - (opp["O_pts"] - opp["D_pts"])
        x = {"O_pts": team["O_pts"], "D_pts": opp["D_pts"], "O_sr": team["O_sr"], "D_sr": opp["D_sr"],
             "O_xr": team["O_xr"], "D_xr": opp["D_xr"], "home_adj": float(home),
             "blow_fav": max(rmargin - 21.0, 0.0), "blow_dog": min(rmargin + 21.0, 0.0)}
        contrib = {k: coef[k] * x[k] for k in COLS}
        return float(mu[si["pts"]] + coef["const"] + sum(contrib.values())), contrib

    ov = json.load(open(os.path.join(DATA, "overs_results.json")))
    sig = pd.read_csv(os.path.join(DATA, "game_signals.csv"), index_col=0)
    hist_tot = np.sort(sig[sig.season < season].proj_total.dropna().values)
    hist_rush = np.sort(sig[sig.season < season].rush_edge.dropna().values)
    hist_pass = np.sort(sig[sig.season < season].pass_edge.dropna().values)
    oc = ov["over_model_simple"]["coef"]
    fl = json.load(open(os.path.join(DATA, "floor_results.json")))
    fl_by_total = fl["base"]["by_total"]

    # --- price ladder: at each common alt-over price, the deepest number that clears break-even
    PRICES = [-200, -300, -400, -500, -600, -800]
    hist_res = pd.read_csv(os.path.join(DATA, "game_signals.csv"), index_col=0).dropna(subset=["close_total"])

    def price_ladder(tot):
        if tot is None:
            return None
        lo, hi = (0, 44.5) if tot < 44.5 else (44.5, 49.5) if tot < 49.5 else (49.5, 54.5) if tot < 54.5 \
            else (54.5, 59.5) if tot < 59.5 else (59.5, 64.5) if tot < 64.5 else (64.5, 200)
        x = hist_res[(hist_res.close_total >= lo) & (hist_res.close_total < hi)]
        out = []
        for pr in PRICES:
            need = (-pr) / ((-pr) + 100.0)
            k = next((k for k in range(1, 45) if (x.total > x.close_total - k).mean() >= need), None)
            out.append({"price": pr, "need": need, "pts": k, "number": (tot - k) if k else None,
                        "rate": (float((x.total > x.close_total - k).mean()) if k else None)})
        # flat-ladder value zone: at a book that prices by points-bought regardless of the number's level,
        # low totals are underpriced and high totals overpriced (measured 2015-2025, 11 seasons)
        zone = "favourable" if tot <= 49.5 else "neutral" if tot <= 59.5 else "unfavourable"
        edge20 = float((x.total > x.close_total - 20).mean() - (800 / 900.0))
        return {"n": int(len(x)), "rows": out, "zone": zone, "edge20_vs_800": edge20}

    def floor_probs(tot):
        if tot is None:
            return None
        lab = "<45" if tot < 44.5 else "45-49.5" if tot < 49.5 else "50-54.5" if tot < 54.5 else "55-59.5" if tot < 59.5 else "60-64.5" if tot < 64.5 else "65+"
        b = next(x for x in fl_by_total if x["bracket"] == lab)
        return {"bracket": lab, "n": b["n"], **{f"k{k}": b[f"hit{k}"] for k in (8, 11, 14, 16, 20)}}

    def pctl(arr, v):
        return float(np.searchsorted(arr, v) / max(len(arr), 1) * 100)

    # --- tail model: lets any (number, price) pair be priced, not just a fixed ladder
    tail_model = json.load(open(os.path.join(DATA, "tail_model.json")))

    # --- what the Shootout percentile has been worth to an over bettor, by bucket
    shoot_cal = json.load(open(os.path.join(DATA, "shoot_calibration.json")))

    # --- both-teams scoring floor (from the market's implied underdog team total)
    bf = json.load(open(os.path.join(DATA, "bothfloor_results.json")))

    def both_floor(tot, spr):
        if tot is None or spr is None:
            return None
        dog = (tot - abs(spr)) / 2.0
        ps = {}
        for X, m in bf["models"].items():
            z = m["intercept"] + m["coef"] * dog
            ps[X] = float(1.0 / (1.0 + np.exp(-z)))
        xs = sorted(int(x) for x in ps)
        f70 = max([x for x in xs if ps[str(x)] >= 0.70], default=None)
        f60 = max([x for x in xs if ps[str(x)] >= 0.60], default=None)
        return {"dog_total": dog, "p": {k: ps[k] for k in ("13", "17", "20", "24")},
                "floor70": f70, "floor60": f60}

    # --- calibration: when the model disagrees with the market on the spread, who has been right?
    gp = pd.read_csv(os.path.join(DATA, "game_projections.csv")).dropna(subset=["close_spread"])
    gp["edge"] = gp.proj_margin - gp.mkt_margin
    gp["ats"] = np.sign(gp.margin - gp.mkt_margin)
    calib = []
    for lo, hi in ((0, 3), (3, 7), (7, 10), (10, 15), (15, 99)):
        m = (gp.edge.abs() >= lo) & (gp.edge.abs() < hi) & (gp.ats != 0)
        dd = gp[m]
        if len(dd) < 10:
            continue
        calib.append({"lo": lo, "hi": hi, "n": int(len(dd)),
                      "model_ats": float((np.sign(dd.edge) == dd.ats).mean()),
                      "model_mae": float(np.abs(dd.margin - dd.proj_margin).mean()),
                      "market_mae": float(np.abs(dd.margin - dd.mkt_margin).mean())})
    big = gp[gp.mkt_margin.abs() >= 28]
    calib_big = {"n": int(len(big)), "model_margin": float(big.proj_margin.abs().mean()),
                 "market_margin": float(big.mkt_margin.abs().mean()), "actual_margin": float(big.margin.abs().mean()),
                 "model_ats": float((np.sign(big.proj_margin - big.mkt_margin) == np.sign(big.margin - big.mkt_margin))[big.margin != big.mkt_margin].mean())}

    live = live_scoreboard(season, week)
    rows = []
    slate = cur[(cur.week == week) & cur.home_fbs & cur.away_fbs]
    for _, r in slate.iterrows():
        hk, ak = str(r.home_id), str(r.away_id)
        th, ta = team_block(hk), team_block(ak)
        ph, ch = project(th, ta, 0 if r.neutral else 1)
        pa, ca = project(ta, th, 0)
        lv = live.get(str(r.game_id), {})
        tot, spr = lv.get("total"), lv.get("spread")
        if tot is None and pd.notna(r.get("live_total")):
            tot = float(r.live_total)
        if spr is None and pd.notna(r.get("live_spread")):
            spr = float(r.live_spread)
        rush_edge = (th["O_rush"] + ta["D_rush"]) + (ta["O_rush"] + th["D_rush"])
        pass_edge = (th["O_pass"] + ta["D_pass"]) + (ta["O_pass"] + th["D_pass"])
        exp_total = ph + pa
        over_read = None
        if tot is not None:
            over_read = float(oc["const"] + oc["close_total"] * tot + oc["rush_edge"] * rush_edge + oc["model_edge"] * (exp_total - tot))
        rows.append({
            "shootout": pctl(hist_tot, exp_total), "rush_edge": rush_edge, "pass_edge": pass_edge,
            "rush_edge_pct": pctl(hist_rush, rush_edge), "pass_edge_pct": pctl(hist_pass, pass_edge),
            "rush_edge_home": th["O_rush"] + ta["D_rush"], "rush_edge_away": ta["O_rush"] + th["D_rush"],
            "pass_edge_home": th["O_pass"] + ta["D_pass"], "pass_edge_away": ta["O_pass"] + th["D_pass"],
            "over_read": over_read, "floor": floor_probs(tot), "bothfloor": both_floor(tot, spr), "ladder": price_ladder(tot),
            "game_id": str(r.game_id), "date": lv.get("date") or r.date, "neutral": bool(r.neutral),
            "venue": r.venue, "completed": bool(r.completed), "status": lv.get("status"), "tv": lv.get("broadcast"),
            "home": {**th, "proj": ph, "contrib": ch, "rank": lv.get("home_rank"), "record": lv.get("home_record")},
            "away": {**ta, "proj": pa, "contrib": ca, "rank": lv.get("away_rank"), "record": lv.get("away_record")},
            "mkt_total": tot, "mkt_spread": spr, "mkt_provider": lv.get("provider"),
            "home_pts": (None if pd.isna(r.home_pts) else int(r.home_pts)),
            "away_pts": (None if pd.isna(r.away_pts) else int(r.away_pts)),
        })
    rows.sort(key=lambda x: x["date"])

    # rankings: every FBS team with a rating
    fbs_keys = sorted({k for k in list(O.keys()) + list(prior_O.keys()) if k != "FCS" and k in tdir})
    ranks = [team_block(k) for k in fbs_keys]

    bt = json.load(open(os.path.join(DATA, "backtest_results.json")))
    out = {
        "built_at": now.isoformat(), "season": int(season), "week": int(week),
        "league_mu": {s: float(mu[si[s]]) for s in STATS}, "hfa": float(h[si["pts"]]),
        "coef": coef, "input_sd": sds, "hyper": {"lam": LAM, "c": C, "rho": RHO},
        "games": rows, "teams": ranks,
        "backtest": {
            "best_model": bt["best_model"],
            "team_points": {k: {"train": v["train"], "holdout": v["holdout"]} for k, v in bt["stage_a"]["team_points"].items()},
            "directional": bt["stage_a"]["directional"], "deciles_holdout": bt["stage_a"]["deciles_holdout"],
            "per_season": bt["stage_a"]["per_season_stability"],
            "ablation": bt["stage_a"]["coefficients"]["ablation_delta_mae"],
            "univariate": bt["stage_a"]["coefficients"]["univariate_corr_with_points"],
            "standardized": bt["stage_a"]["coefficients"]["standardized"],
            "totals": {k: bt["stage_b"][k] for k in ("totals_train", "totals_holdout", "totals_per_season", "totals_incremental_train", "totals_incremental_holdout", "totals_by_games_played")},
            "spreads": {k: bt["stage_b"][k] for k in ("spread_train", "spread_holdout", "spread_per_season")},
            "opener": {"pooled_totals": bt["stage_b"]["open_vs_close_totals"], "pooled_spreads": bt["stage_b"]["open_vs_close_spreads"],
                       "holdout_totals": bt["extras"]["opener_holdout_totals"], "holdout_spreads": bt["extras"]["opener_holdout_spreads"],
                       "by_season": bt["extras"]["line_move_agreement_by_season"]},
            "total_directional": bt["total_directional"],
            "turnover_persistence": bt["turnover_persistence"], "turnover_luck": bt["extras"]["turnover_luck_regression"],
            "tempo": {"train": bt["extras"]["tempo_buckets_train"], "holdout": bt["extras"]["tempo_buckets_holdout"]},
            "team_total": {"train": bt["extras"]["team_total_train"], "holdout": bt["extras"]["team_total_holdout"]},
            "n_games": bt["n_games"],
        },
        "overs": {
            "base_over_train": ov["base_over_rate_train"], "base_over_holdout": ov["base_over_rate_holdout"],
            "by_season": ov["over_rate_by_season"],
            "screen": {k: {"label": v["label"], "corr_total_train": v["corr_total_train"],
                           "q_train": v["quintiles_train"], "q_holdout": v["quintiles_holdout"],
                           "resid_t_train": v["resid_t_train"], "resid_t_holdout": v["resid_t_holdout"],
                           "seasons_top_gt_bottom": sum(1 for p in v["per_season"] if p["top_over"] > p["bottom_over"]),
                           "n_seasons": len(v["per_season"])} for k, v in ov["screen"].items()},
            "brackets": ov["total_brackets"], "simple_model": ov["over_model_simple"], "full_model": ov["over_model"],
            "blowout_fix": ov["blowout_fix"], "rushpass": ov["rushpass"],
        },
        "calibration": {"by_disagreement": calib, "big_mismatch": calib_big},
        "tail_model": tail_model,
        "shoot_calibration": shoot_cal,
        "bothfloor": {"base_rates": bf["base_rates"], "empirical_bins": bf["empirical_bins"],
                      "predictor_corr": bf["predictor_corr"], "n": bf["n"],
                      "validation": {k: bf["validation"][k]["holdout_calibration"] for k in ("13", "17", "20", "24")}},
        "floor": {
            "base": fl["base"],
            "screen": {k: {"label": v["label"], "z11": (v["logit_k11"] or {}).get("z"), "z16": (v["logit_k16"] or {}).get("z"),
                           "s11": v["strata_k11_train"], "s16": v["strata_k16_train"], "h11": v["strata_k11_holdout"], "h16": v["strata_k16_holdout"],
                           "seasons_pos_k11": v["seasons_pos_k11"], "n_seasons_k11": v["n_seasons_k11"]} for k, v in fl["screen"].items()},
            "models": {k: {"base_rate_train": m["base_rate_train"], "base_rate_holdout": m["base_rate_holdout"],
                           "loglik_gain": m["loglik_gain_vs_total_only_holdout"], "top_holdout": m["top_holdout"],
                           "top_holdout_baseline": m["top_holdout_baseline_total_only"], "deciles_holdout": m["deciles_holdout"]}
                       for k, m in fl["models"]["models"].items()},
        },
    }
    path = os.path.join(BASE, "ncaaf_board_data.json")
    json.dump(out, open(path, "w"), default=float)
    print(f"wrote {path}: {len(rows)} games, {len(ranks)} teams")
    return out


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    s = int(pos[0]) if len(pos) > 0 else None
    w = int(pos[1]) if len(pos) > 1 else None
    main(s, w, do_refresh=("--no-refresh" not in sys.argv))
