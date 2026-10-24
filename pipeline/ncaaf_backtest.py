#!/usr/bin/env python3
"""
ncaaf_backtest.py — Stage A / Stage B evaluation of the NCAAF team-points score.

Stage A  does the as-of score predict a team's points (and the game total / margin) — vs naive baselines?
Stage B  does the projection beat the closing total / spread?  (break-even 52.4% at -110)
Holdout  2025 is never used for fitting; per-season results reported for stability.

Reads data/features.csv (from ncaaf_ratings.py). Writes data/backtest_results.json and prints a report.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from numpy.linalg import lstsq

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
TRAIN = list(range(2015, 2025))
HOLD = [2025]
BE = 0.524  # break-even hit rate at -110


def load(tag="features"):
    f = pd.read_csv(os.path.join(DATA, f"{tag}.csv"))
    g = pd.read_csv(os.path.join(DATA, "games.csv"))
    f = f[f.team_fbs & f.opp_fbs & f.season.between(2015, 2025)].copy()
    f = f.dropna(subset=["O_pts", "raw_pts", "opp_raw_pa"])
    f["pts_full"] = f["pts"].astype(float)  # regulation points (ratings target)
    # full-game points (incl. OT) come from games.csv
    gg = g.set_index("game_id")
    f["home_flag"] = f.home.astype(int)
    f["full_pts"] = np.where(f.home_flag == 1, f.game_id.map(gg.home_pts), f.game_id.map(gg.away_pts))
    # neutral-site teams: 'home' is 0 for both; use listed home to pick the right score
    listed_home = f.game_id.map(gg.home_id).astype(str)
    f["full_pts"] = np.where(f.team_id.astype(str) == listed_home, f.game_id.map(gg.home_pts), f.game_id.map(gg.away_pts))
    f["is_listed_home"] = (f.team_id.astype(str) == listed_home).astype(int)
    for col in ("close_total", "close_spread", "open_total", "open_spread", "neutral"):
        f[col] = f.game_id.map(gg[col])
    # market-implied team total: listed home = (T - S)/2, away = (T + S)/2  (S = home spread, neg = home fav)
    f["mkt_team_total"] = np.where(f.is_listed_home == 1, (f.close_total - f.close_spread) / 2,
                                   (f.close_total + f.close_spread) / 2)
    # rating margin for this team in this game, and the blowout hinge terms (the linear model's tail fix)
    f["rmargin"] = (f.O_pts - f.Dt_pts) - (f.Oo_pts - f.D_pts)
    f["blow_fav"] = np.clip(f.rmargin - 21, 0, None)
    f["blow_dog"] = np.clip(f.rmargin + 21, None, 0)
    return f, g


# ----------------------------------------------------------------------------- models

def design(d, cols):
    X = np.column_stack([np.ones(len(d))] + [d[c].values.astype(float) for c in cols])
    return X


def fit_ols(d, cols, target="full_pts"):
    """Fit points as a deviation from the as-of league average (mu_pts), so the level tracks each season."""
    X = design(d, cols)
    y = d[target].values.astype(float) - d["mu_pts"].values.astype(float)
    beta, *_ = lstsq(X, y, rcond=None)
    return beta


def predict(d, cols, beta):
    return d["mu_pts"].values.astype(float) + design(d, cols) @ beta


FEATURE_SETS = {
    "raw season PPG / PA (no adjustment)": ["raw_pts", "opp_raw_pa", "home_adj"],
    "raw last-4 PPG / PA": ["raw4_pts", "opp_raw4_pa", "home_adj"],
    "adjusted points rating (O_pts + D_pts)": ["O_pts", "D_pts", "home_adj"],
    "adjusted efficiency (ppd, ypp, sr, xr)": ["O_ppd", "D_ppd", "O_ypp", "D_ypp", "O_sr", "D_sr", "O_xr", "D_xr", "home_adj"],
    "efficiency + tempo": ["O_ppd", "D_ppd", "O_ypp", "D_ypp", "O_sr", "D_sr", "O_xr", "D_xr",
                           "O_plays", "Oo_plays", "D_plays", "home_adj"],
    "efficiency + tempo + turnovers": ["O_ppd", "D_ppd", "O_ypp", "D_ypp", "O_sr", "D_sr", "O_xr", "D_xr",
                                       "O_plays", "Oo_plays", "D_plays", "O_to_rate", "D_to_rate", "home_adj"],
    "everything (+ points rating, field pos.)": ["O_pts", "D_pts", "O_ppd", "D_ppd", "O_ypp", "D_ypp", "O_sr", "D_sr",
                                                 "O_xr", "D_xr", "O_plays", "Oo_plays", "D_plays", "O_to_rate",
                                                 "D_to_rate", "O_fp", "D_fp", "home_adj"],
    "SCORE: adj points + success rate + explosiveness + home": ["O_pts", "D_pts", "O_sr", "D_sr", "O_xr", "D_xr", "home_adj"],
    "SCORE + blowout terms": ["O_pts", "D_pts", "O_sr", "D_sr", "O_xr", "D_xr", "home_adj", "blow_fav", "blow_dog"],
}
SCORE_MODEL = "SCORE + blowout terms"


def metrics(y, p):
    e = y - p
    return {"n": int(len(y)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "corr": float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else float("nan")}


def loso(d, cols, target="full_pts"):
    """leave-one-season-out predictions within the training seasons"""
    p = pd.Series(index=d.index, dtype=float)
    for s in sorted(d.season.unique()):
        tr, te = d[d.season != s], d[d.season == s]
        beta = fit_ols(tr, cols, target)
        p.loc[te.index] = predict(te, cols, beta)
    return p


def stage_a(f):
    out = {"team_points": {}, "coefficients": {}}
    tr = f[f.season.isin(TRAIN)].copy()
    ho = f[f.season.isin(HOLD)].copy()
    mean_train = tr.full_pts.mean()
    out["team_points"]["league average (no model)"] = {
        "train": metrics(tr.full_pts.values, np.full(len(tr), mean_train)),
        "holdout": metrics(ho.full_pts.values, np.full(len(ho), mean_train))}
    mk = tr.dropna(subset=["mkt_team_total"]); mh = ho.dropna(subset=["mkt_team_total"])
    out["team_points"]["MARKET implied team total (closing)"] = {
        "train": metrics(mk.full_pts.values, mk.mkt_team_total.values),
        "holdout": metrics(mh.full_pts.values, mh.mkt_team_total.values)}
    preds = {}
    for name, cols in FEATURE_SETS.items():
        p_tr = loso(tr, cols)
        beta = fit_ols(tr, cols)
        p_ho = predict(ho, cols, beta)
        out["team_points"][name] = {"train": metrics(tr.full_pts.values, p_tr.values),
                                    "holdout": metrics(ho.full_pts.values, p_ho),
                                    "coef": dict(zip(["const"] + cols, [float(b) for b in beta]))}
        preds[name] = (p_tr, pd.Series(p_ho, index=ho.index))
    # standardized coefficients for the 'everything' model: which inputs carry the weight
    cols = FEATURE_SETS["everything (+ points rating, field pos.)"]
    beta = fit_ols(tr, cols)
    sd = tr[cols].std()
    out["coefficients"]["standardized"] = {c: float(b * sd[c]) for c, b in zip(cols, beta[1:])}
    # univariate: correlation of each as-of rating with the team's points (train) and with points allowed
    uni = {}
    for c in ["O_pts", "O_ppd", "O_ypp", "O_sr", "O_xr", "O_plays", "O_to_rate", "O_fp", "O_ypd", "O_spp",
              "raw_pts", "raw4_pts"]:
        uni[c] = float(np.corrcoef(tr[c], tr.full_pts)[0, 1])
    for c in ["D_pts", "D_ppd", "D_ypp", "D_sr", "D_xr", "D_plays", "D_to_rate", "D_fp", "D_ypd", "D_spp",
              "opp_raw_pa", "opp_raw4_pa"]:
        uni[c] = float(np.corrcoef(tr[c], tr.full_pts)[0, 1])
    out["coefficients"]["univariate_corr_with_points"] = uni
    # ablation on holdout-free LOSO: drop one input at a time from the 'everything' model
    abl = {}
    base_mae = out["team_points"]["everything (+ points rating, field pos.)"]["train"]["mae"]
    for drop in ["O_pts+D_pts", "O_ppd+D_ppd", "O_ypp+D_ypp", "O_sr+D_sr", "O_xr+D_xr", "O_plays+Oo_plays+D_plays",
                 "O_to_rate+D_to_rate", "O_fp+D_fp", "home_adj"]:
        dc = drop.split("+")
        cc = [c for c in cols if c not in dc]
        p = loso(tr, cc)
        abl[drop] = float(np.abs(tr.full_pts - p).mean() - base_mae)
    out["coefficients"]["ablation_delta_mae"] = abl
    # deciles (holdout, best model)
    best = SCORE_MODEL
    p_tr, p_ho = preds[best]
    dec = []
    q = pd.qcut(p_ho, 10, labels=False, duplicates="drop")
    for k in sorted(q.unique()):
        m = q == k
        dec.append({"decile": int(k) + 1, "pred": float(p_ho[m].mean()), "actual": float(ho.full_pts[m].mean()), "n": int(m.sum())})
    out["deciles_holdout"] = dec
    # per-season stability of the SCORE model (corr + MAE of team points, LOSO in train / fitted for holdout)
    allp = pd.concat([p_tr, p_ho])
    stab = []
    for s in sorted(f.season.unique()):
        d = f[f.season == s]
        stab.append({"season": int(s), "n": int(len(d)), "corr": float(np.corrcoef(allp[d.index], d.full_pts)[0, 1]),
                     "mae": float(np.abs(allp[d.index] - d.full_pts).mean()),
                     "mkt_corr": float(np.corrcoef(d.dropna(subset=["mkt_team_total"]).mkt_team_total,
                                                   d.dropna(subset=["mkt_team_total"]).full_pts)[0, 1]),
                     "mkt_mae": float(np.abs(d.mkt_team_total - d.full_pts).mean())})
    out["per_season_stability"] = stab
    # directional check: above/below league average
    for name in (best, "adjusted points rating (O_pts + D_pts)"):
        p_tr, p_ho = preds[name]
        res = {}
        for lbl, d, p in (("train", tr, p_tr), ("holdout", ho, p_ho)):
            mu = d.full_pts.mean()
            sgn = np.sign(p - mu); act = np.sign(d.full_pts - mu)
            m = sgn != 0
            res[lbl] = {"hit": float((sgn[m] == act[m]).mean()), "n": int(m.sum())}
            # against the team's own season average (harder): does the matchup say more/less than usual?
            own = d.raw_pts
            sgn2 = np.sign(p - own); act2 = np.sign(d.full_pts - own)
            m2 = (sgn2 != 0) & (d.games_before >= 4)
            res[lbl + "_vs_own_avg"] = {"hit": float((sgn2[m2] == act2[m2]).mean()), "n": int(m2.sum())}
        out.setdefault("directional", {})[name] = res
    return out, preds


def game_level(f, preds, name):
    """Combine the two team rows into game rows: projected total and margin."""
    p_tr, p_ho = preds[name]
    p = pd.concat([p_tr, p_ho])
    d = f.loc[p.index].copy()
    d["proj"] = p.values
    h = d[d.is_listed_home == 1].set_index("game_id")
    a = d[d.is_listed_home == 0].set_index("game_id")
    idx = h.index.intersection(a.index)
    g = pd.DataFrame({
        "season": h.loc[idx, "season"], "wk": h.loc[idx, "wk"],
        "home_proj": h.loc[idx, "proj"], "away_proj": a.loc[idx, "proj"],
        "home_pts": h.loc[idx, "full_pts"], "away_pts": a.loc[idx, "full_pts"],
        "close_total": h.loc[idx, "close_total"], "close_spread": h.loc[idx, "close_spread"],
        "open_total": h.loc[idx, "open_total"], "open_spread": h.loc[idx, "open_spread"],
        "games_before": np.minimum(h.loc[idx, "games_before"], a.loc[idx, "games_before"]),
        "home_O_plays": h.loc[idx, "O_plays"], "away_O_plays": a.loc[idx, "O_plays"],
        "home_D_plays": h.loc[idx, "Dt_plays"], "away_D_plays": a.loc[idx, "Dt_plays"],
        "home_O_spp": h.loc[idx, "O_spp"], "away_O_spp": a.loc[idx, "O_spp"],
        "mu_plays": h.loc[idx, "mu_plays"],
    })
    g["proj_total"] = g.home_proj + g.away_proj
    g["proj_margin"] = g.home_proj - g.away_proj
    g["total"] = g.home_pts + g.away_pts
    g["margin"] = g.home_pts - g.away_pts
    g["mkt_margin"] = -g.close_spread
    return g


def total_directional(gl):
    """Does the projected game total point the right way vs the league-average total, and are its deciles ordered?"""
    out = {}
    for lbl, d in (("train", gl[gl.season.isin(TRAIN)]), ("holdout", gl[gl.season.isin(HOLD)])):
        mu = d.total.mean()
        sgn = np.sign(d.proj_total - mu); act = np.sign(d.total - mu)
        m = (sgn != 0) & (act != 0)
        q = pd.qcut(d.proj_total, 10, labels=False, duplicates="drop")
        dec = [{"decile": int(k) + 1, "proj": float(d.proj_total[q == k].mean()), "actual": float(d.total[q == k].mean()),
                "close": float(d.close_total[q == k].mean())} for k in sorted(q.unique())]
        out[lbl] = {"n": int(m.sum()), "hit_vs_league_avg": float((sgn[m] == act[m]).mean()),
                    "corr_proj": float(np.corrcoef(d.proj_total, d.total)[0, 1]),
                    "mae_proj": float(np.abs(d.proj_total - d.total).mean()), "deciles": dec,
                    "margin_su": float((np.sign(d.proj_margin) == np.sign(d.margin))[d.margin != 0].mean()),
                    "margin_corr": float(np.corrcoef(d.proj_margin, d.margin)[0, 1])}
    return out


def hit_table(edge, outcome, thresholds=(0, 2, 3, 5, 7, 10)):
    """edge: model - market (signed); outcome: actual - market (signed). Hit = same sign, pushes excluded."""
    rows = []
    for th in thresholds:
        m = (np.abs(edge) >= th) & (outcome != 0) & edge.notna() & outcome.notna() & (edge != 0)
        if m.sum() == 0:
            continue
        hit = (np.sign(edge[m]) == np.sign(outcome[m])).mean()
        rows.append({"min_edge": th, "n": int(m.sum()), "hit": float(hit), "roi_at_-110": float(hit * 1.909 - 1)})
    return rows


def per_season(edge, outcome, seasons, th=0):
    res = []
    for s in sorted(seasons.unique()):
        m = (seasons == s) & (np.abs(edge) >= th) & (outcome != 0) & edge.notna() & outcome.notna() & (edge != 0)
        if m.sum() == 0:
            continue
        res.append({"season": int(s), "n": int(m.sum()), "hit": float((np.sign(edge[m]) == np.sign(outcome[m])).mean())})
    return res


def stage_b(g):
    out = {}
    tr = g[g.season.isin(TRAIN)]; ho = g[g.season.isin(HOLD)]
    # --- totals
    for lbl, d in (("train", tr), ("holdout", ho)):
        d = d.dropna(subset=["close_total"])
        edge = d.proj_total - d.close_total
        outc = d.total - d.close_total
        out[f"totals_{lbl}"] = {
            "n": int(len(d)),
            "model_total_mae": float(np.abs(d.total - d.proj_total).mean()),
            "close_total_mae": float(np.abs(d.total - d.close_total).mean()),
            "corr_model": float(np.corrcoef(d.total, d.proj_total)[0, 1]),
            "corr_close": float(np.corrcoef(d.total, d.close_total)[0, 1]),
            "mean_edge": float(edge.mean()),
            "hits": hit_table(edge, outc),
            "over_rate_when_model_over": float(((outc > 0) & (edge > 0)).sum() / max(((outc != 0) & (edge > 0)).sum(), 1)),
            "under_rate_when_model_under": float(((outc < 0) & (edge < 0)).sum() / max(((outc != 0) & (edge < 0)).sum(), 1)),
        }
    d = tr.dropna(subset=["close_total"])
    out["totals_per_season"] = per_season(d.proj_total - d.close_total, d.total - d.close_total, d.season)
    d = ho.dropna(subset=["close_total"])
    out["totals_per_season"] += per_season(d.proj_total - d.close_total, d.total - d.close_total, d.season)
    # does the model add information to the closing total?  (actual - close) ~ (model - close), fit on train
    d = tr.dropna(subset=["close_total"])
    x = (d.proj_total - d.close_total).values; y = (d.total - d.close_total).values
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.sum(resid ** 2) / (len(y) - 2) / np.sum((x - x.mean()) ** 2))
    out["totals_incremental_train"] = {"slope": float(beta[1]), "t": float(beta[1] / se), "n": int(len(y))}
    d = ho.dropna(subset=["close_total"])
    x = (d.proj_total - d.close_total).values; y = (d.total - d.close_total).values
    X = np.column_stack([np.ones(len(x)), x]); b2, *_ = lstsq(X, y, rcond=None)
    r2 = y - X @ b2; se2 = np.sqrt(np.sum(r2 ** 2) / (len(y) - 2) / np.sum((x - x.mean()) ** 2))
    out["totals_incremental_holdout"] = {"slope": float(b2[1]), "t": float(b2[1] / se2), "n": int(len(y))}
    # --- spreads
    for lbl, d in (("train", tr), ("holdout", ho)):
        d = d.dropna(subset=["close_spread"])
        edge = d.proj_margin - d.mkt_margin
        outc = d.margin - d.mkt_margin
        out[f"spread_{lbl}"] = {
            "n": int(len(d)),
            "model_margin_mae": float(np.abs(d.margin - d.proj_margin).mean()),
            "close_margin_mae": float(np.abs(d.margin - d.mkt_margin).mean()),
            "corr_model": float(np.corrcoef(d.margin, d.proj_margin)[0, 1]),
            "corr_close": float(np.corrcoef(d.margin, d.mkt_margin)[0, 1]),
            "su_model": float((np.sign(d.proj_margin) == np.sign(d.margin))[d.margin != 0].mean()),
            "su_market": float((np.sign(d.mkt_margin) == np.sign(d.margin))[(d.margin != 0) & (d.mkt_margin != 0)].mean()),
            "hits": hit_table(edge, outc),
        }
    d = tr.dropna(subset=["close_spread"])
    out["spread_per_season"] = per_season(d.proj_margin - d.mkt_margin, d.margin - d.mkt_margin, d.season)
    d = ho.dropna(subset=["close_spread"])
    out["spread_per_season"] += per_season(d.proj_margin - d.mkt_margin, d.margin - d.mkt_margin, d.season)
    # --- opening lines (where available)
    d = g.dropna(subset=["open_total", "close_total"])
    out["open_vs_close_totals"] = {
        "n": int(len(d)),
        "seasons": sorted(int(s) for s in d.season.unique()),
        "hits_vs_open": hit_table(d.proj_total - d.open_total, d.total - d.open_total),
        "hits_vs_close_same_games": hit_table(d.proj_total - d.close_total, d.total - d.close_total),
        "line_moves_toward_model": float((np.sign(d.close_total - d.open_total) == np.sign(d.proj_total - d.open_total))[(d.close_total != d.open_total)].mean()),
        "n_moved": int((d.close_total != d.open_total).sum()),
    }
    d = g.dropna(subset=["open_spread", "close_spread"])
    out["open_vs_close_spreads"] = {
        "n": int(len(d)),
        "hits_vs_open": hit_table(d.proj_margin + d.open_spread, d.margin + d.open_spread),
        "hits_vs_close_same_games": hit_table(d.proj_margin - d.mkt_margin, d.margin - d.mkt_margin),
        "line_moves_toward_model": float((np.sign(-d.close_spread + d.open_spread) == np.sign(d.proj_margin + d.open_spread))[(d.close_spread != d.open_spread)].mean()),
        "n_moved": int((d.close_spread != d.open_spread).sum()),
    }
    # --- early-season vs late (games_before buckets), totals, holdout+train pooled but per-season available
    d = g.dropna(subset=["close_total"])
    buck = []
    for lo, hi in ((0, 2), (3, 5), (6, 8), (9, 20)):
        m = d.games_before.between(lo, hi)
        e = d.proj_total[m] - d.close_total[m]; o = d.total[m] - d.close_total[m]
        mm = (o != 0) & (e != 0)
        buck.append({"games_before": f"{lo}-{hi}", "n": int(mm.sum()), "hit": float((np.sign(e[mm]) == np.sign(o[mm])).mean()),
                     "model_mae": float(np.abs(d.total[m] - d.proj_total[m]).mean()), "close_mae": float(np.abs(o[m]).mean())})
    out["totals_by_games_played"] = buck
    return out


def extras(f, preds, gl, best):
    out = {}
    tr = f[f.season.isin(TRAIN)]; ho = f[f.season.isin(HOLD)]
    p_tr, p_ho = preds[best]
    # 1. TEAM-total information: (actual - market team total) ~ (model - market team total)
    for lbl, d, p in (("train", tr, p_tr), ("holdout", ho, p_ho)):
        d = d.dropna(subset=["mkt_team_total"])
        x = (p[d.index] - d.mkt_team_total).values; y = (d.full_pts - d.mkt_team_total).values
        X = np.column_stack([np.ones(len(x)), x]); b, *_ = lstsq(X, y, rcond=None)
        r = y - X @ b; se = np.sqrt(np.sum(r ** 2) / (len(y) - 2) / np.sum((x - x.mean()) ** 2))
        m = (x != 0) & (y != 0)
        out[f"team_total_{lbl}"] = {"n": int(len(y)), "slope": float(b[1]), "t": float(b[1] / se),
                                    "hit_sign": float((np.sign(x[m]) == np.sign(y[m])).mean()),
                                    "hits": hit_table(pd.Series(x), pd.Series(y))}
    # 2. tempo buckets on game totals (train + holdout separately): combined tempo rating tercile
    for lbl, d in (("train", gl[gl.season.isin(TRAIN)]), ("holdout", gl[gl.season.isin(HOLD)])):
        d = d.dropna(subset=["close_total"]).copy()
        d["tempo"] = d.home_O_plays + d.away_O_plays + d.home_D_plays + d.away_D_plays
        d["bucket"] = pd.qcut(d.tempo, 3, labels=["slow", "mid", "fast"])
        rows = []
        for b, dd in d.groupby("bucket", observed=True):
            o = dd.total - dd.close_total
            e = dd.proj_total - dd.close_total
            mm = (o != 0) & (e != 0)
            rows.append({"bucket": str(b), "n": int(len(dd)), "avg_total": float(dd.total.mean()),
                         "avg_close": float(dd.close_total.mean()),
                         "over_rate": float((o[o != 0] > 0).mean()),
                         "model_hit": float((np.sign(e[mm]) == np.sign(o[mm])).mean())})
        out[f"tempo_buckets_{lbl}"] = rows
    # 3. turnover luck: early-season turnover margin vs later residual (actual - model) — do lucky teams regress?
    t = pd.read_csv(os.path.join(DATA, "teamgames.csv"))
    t = t[t.team_fbs & (t.stype == 2)]
    t["to_margin"] = t.opp_turnovers - t.turnovers
    early = t[t.week <= 6].groupby(["season", "team_id"]).to_margin.mean().rename("early_to_margin")
    allp = pd.concat([p_tr, p_ho])
    late = f.loc[allp.index]
    late = late[late.wk >= 8].copy()
    late["resid"] = late.full_pts - allp[late.index]
    late = late.merge(early.reset_index(), on=["season", "team_id"], how="inner")
    late["luck"] = pd.qcut(late.early_to_margin, 3, labels=["unlucky", "neutral", "lucky"])
    rows = []
    for b, dd in late.groupby("luck", observed=True):
        rows.append({"bucket": str(b), "n": int(len(dd)), "early_to_margin": float(dd.early_to_margin.mean()),
                     "later_resid_pts": float(dd.resid.mean()),
                     "later_resid_t": float(dd.resid.mean() / (dd.resid.std() / np.sqrt(len(dd))))})
    # the market version: does the market also under-rate lucky teams? actual - mkt team total by bucket
    late2 = late.dropna(subset=["mkt_team_total"])
    for r in rows:
        dd = late2[late2.luck == r["bucket"]]
        r["later_vs_market_pts"] = float((dd.full_pts - dd.mkt_team_total).mean())
    out["turnover_luck_regression"] = rows
    # 4. opener, holdout only
    d = gl[gl.season.isin(HOLD)].dropna(subset=["open_total", "close_total"])
    out["opener_holdout_totals"] = {"n": int(len(d)),
                                    "hits_vs_open": hit_table(d.proj_total - d.open_total, d.total - d.open_total),
                                    "hits_vs_close": hit_table(d.proj_total - d.close_total, d.total - d.close_total),
                                    "line_moves_toward_model": float((np.sign(d.close_total - d.open_total) == np.sign(d.proj_total - d.open_total))[(d.close_total != d.open_total)].mean()),
                                    "n_moved": int((d.close_total != d.open_total).sum())}
    d = gl[gl.season.isin(HOLD)].dropna(subset=["open_spread", "close_spread"])
    out["opener_holdout_spreads"] = {"n": int(len(d)),
                                     "hits_vs_open": hit_table(d.proj_margin + d.open_spread, d.margin + d.open_spread),
                                     "hits_vs_close": hit_table(d.proj_margin - d.mkt_margin, d.margin - d.mkt_margin),
                                     "line_moves_toward_model": float((np.sign(-d.close_spread + d.open_spread) == np.sign(d.proj_margin + d.open_spread))[(d.close_spread != d.open_spread)].mean()),
                                     "n_moved": int((d.close_spread != d.open_spread).sum())}
    # per-season line-move agreement (totals)
    d = gl.dropna(subset=["open_total", "close_total"])
    d = d[d.close_total != d.open_total]
    out["line_move_agreement_by_season"] = [
        {"season": int(s), "n_moved": int(len(dd)),
         "toward_model": float((np.sign(dd.close_total - dd.open_total) == np.sign(dd.proj_total - dd.open_total)).mean())}
        for s, dd in d.groupby("season")]
    return out


def turnover_luck(f):
    """Persistence test: correlation of a team's first-half-season stat with its second-half stat."""
    t = pd.read_csv(os.path.join(DATA, "teamgames.csv"))
    t = t[t.team_fbs & (t.stype == 2) & (t.plays >= 30)].copy()
    t["ppd"] = (7 * t.td_drives + 3 * t.fg_drives) / t.drives
    t["ypp"] = t.yards / t.plays; t["sr"] = t.succ / t.succ_n; t["xr"] = t.expl15 / t.plays
    t["to_rate"] = t.turnovers / t.drives; t["to_margin"] = t.opp_turnovers - t.turnovers
    t["pts"] = t.pts_reg
    res = {}
    for s in ["pts", "ppd", "ypp", "sr", "xr", "to_rate", "to_margin", "plays"]:
        a = t[t.week <= 6].groupby(["season", "team_id"])[s].mean()
        b = t[t.week >= 8].groupby(["season", "team_id"])[s].mean()
        j = pd.concat([a, b], axis=1, keys=["h1", "h2"]).dropna()
        res[s] = {"corr_h1_h2": float(np.corrcoef(j.h1, j.h2)[0, 1]), "n_team_seasons": int(len(j))}
    return res


def main(tag="features"):
    f, g = load(tag)
    print(f"evaluation rows: {len(f)} team-games, seasons {f.season.min()}-{f.season.max()}")
    A, preds = stage_a(f)
    best = SCORE_MODEL
    gl = game_level(f, preds, best)
    B = stage_b(gl)
    # spread/total using the simple points-rating model too, for comparison
    gl2 = game_level(f, preds, "adjusted points rating (O_pts + D_pts)")
    B2 = stage_b(gl2)
    T = turnover_luck(f)
    E = extras(f, preds, gl, best)
    TD = total_directional(gl)
    res = {"stage_a": A, "stage_b": B, "stage_b_points_rating_model": B2, "turnover_persistence": T,
           "extras": E, "total_directional": TD, "best_model": best, "n_games": int(len(gl))}
    json.dump(res, open(os.path.join(DATA, "backtest_results.json"), "w"), indent=1, default=float)
    gl.to_csv(os.path.join(DATA, "game_projections.csv"))
    # ------------------------------------------------------------------ report
    print("\n=== STAGE A: team points, MAE (lower is better) / corr ===")
    print(f"{'model':<48} {'train MAE':>9} {'corr':>6} | {'2025 MAE':>9} {'corr':>6}")
    for k, v in A["team_points"].items():
        print(f"{k:<48} {v['train']['mae']:>9.2f} {v['train']['corr']:>6.3f} | {v['holdout']['mae']:>9.2f} {v['holdout']['corr']:>6.3f}")
    print("\nstandardized coefficients (points per 1 SD of input), 'everything' model:")
    for k, v in sorted(A["coefficients"]["standardized"].items(), key=lambda kv: -abs(kv[1])):
        print(f"   {k:<12} {v:+.2f}")
    print("\nablation (LOSO MAE change when dropped; positive = the input helps):")
    for k, v in sorted(A["coefficients"]["ablation_delta_mae"].items(), key=lambda kv: -kv[1]):
        print(f"   {k:<28} {v:+.3f}")
    print("\nunivariate corr with team points (train):")
    for k, v in sorted(A["coefficients"]["univariate_corr_with_points"].items(), key=lambda kv: -abs(kv[1])):
        print(f"   {k:<12} {v:+.3f}")
    print("\ndirectional:", json.dumps(A["directional"], indent=1))
    print("\nholdout deciles:", [(d['decile'], round(d['pred'], 1), round(d['actual'], 1)) for d in A["deciles_holdout"]])
    print("\n=== STAGE B: totals ===")
    for k in ("totals_train", "totals_holdout"):
        v = B[k]; print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'hits'})
        for h in v["hits"]:
            print(f"   edge>={h['min_edge']:>2}: n={h['n']:>5} hit={h['hit']:.3f} roi={h['roi_at_-110']:+.3f}")
    print("per season:", [(r['season'], r['n'], round(r['hit'], 3)) for r in B["totals_per_season"]])
    print("incremental:", B["totals_incremental_train"], B["totals_incremental_holdout"])
    print("by games played:", B["totals_by_games_played"])
    print("\n=== STAGE B: spreads ===")
    for k in ("spread_train", "spread_holdout"):
        v = B[k]; print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'hits'})
        for h in v["hits"]:
            print(f"   edge>={h['min_edge']:>2}: n={h['n']:>5} hit={h['hit']:.3f} roi={h['roi_at_-110']:+.3f}")
    print("per season:", [(r['season'], r['n'], round(r['hit'], 3)) for r in B["spread_per_season"]])
    print("\n=== opening vs closing ===")
    print(json.dumps(B["open_vs_close_totals"], indent=1)); print(json.dumps(B["open_vs_close_spreads"], indent=1))
    print("\n=== turnover luck / persistence (first 6 weeks vs week 8+) ===")
    for k, v in T.items():
        print(f"   {k:<10} r={v['corr_h1_h2']:+.3f}  (n={v['n_team_seasons']})")
    print("\n=== TOTAL directional ===")
    for k, v in TD.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "deciles"})
        print("   deciles:", [(x["decile"], round(x["proj"], 1), round(x["actual"], 1), round(x["close"], 1)) for x in v["deciles"]])
    print("\nper-season stability:", [(s["season"], round(s["corr"], 3), round(s["mae"], 2), round(s["mkt_corr"], 3)) for s in A["per_season_stability"]])
    print("\n=== EXTRAS ===")
    print(json.dumps(E, indent=1, default=float))
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "features")
