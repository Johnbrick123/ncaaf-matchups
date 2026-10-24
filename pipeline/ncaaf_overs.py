#!/usr/bin/env python3
"""
ncaaf_overs.py — what predicts high-scoring games, and what (if anything) predicts OVERS vs the closing total.

Game-level signals built from the as-of ratings of both teams (features.csv):
  shoot_pts    both offenses' points ratings + both defenses' allowed ratings  (absolute scoring index)
  off_sum / def_sum   the two halves of that
  pace         combined tempo rating (plays above average, both offenses and both defenses)
  pass_mix     combined pass-rate rating (pass-heavy teams stop the clock)
  pass_edge / rush_edge   offense strength + opposing defense weakness, per side, in yards/attempt
  align        does the offense's lean (pass vs rush) line up with the defense's weakness?  (z-scored product)
  worst_def    the better of the two defenses (min of D_pts) — high means BOTH defenses are bad
  blowout      |rating margin| beyond 21 points (the model's known tail weakness)
  plus context: indoor, conference game, neutral, week, posted total level, open->close move

Stage A: correlation with the actual total (does the signal find high-scoring games?)
Stage B: over rate vs the closing total by quintile, 2015-24 pooled and 2025 alone; slope of (actual-close) on signal.
Then a fitted over-model on all signals (train) evaluated on 2025.
Writes data/overs_results.json.
"""
import json
import os
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from ncaaf_backtest import load, fit_ols, predict, loso, FEATURE_SETS, SCORE_MODEL, TRAIN, HOLD

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")


def z(s):
    return (s - s.mean()) / s.std()


def assemble(f):
    """one row per game with both teams' as-of ratings + SCORE projections (LOSO in train, fitted for holdout)."""
    tr = f[f.season.isin(TRAIN)].copy(); ho = f[f.season.isin(HOLD)].copy()
    cols = FEATURE_SETS[SCORE_MODEL]
    p_tr = loso(tr, cols)
    beta = fit_ols(tr, cols)
    p_ho = pd.Series(predict(ho, cols, beta), index=ho.index)
    f = f.copy()
    f["proj"] = pd.concat([p_tr, p_ho])
    # blowout term test at team level: rating margin for this team in this game
    f["rmargin"] = (f.O_pts - f.Dt_pts) - (f.Oo_pts - f.D_pts)  # team strength minus opponent strength (pts)
    f["blow_fav"] = np.clip(f.rmargin - 21, 0, None)
    f["blow_dog"] = np.clip(f.rmargin + 21, None, 0)
    h = f[f.is_listed_home == 1].set_index("game_id"); a = f[f.is_listed_home == 0].set_index("game_id")
    idx = h.index.intersection(a.index)
    h, a = h.loc[idx], a.loc[idx]
    g = pd.read_csv(os.path.join(DATA, "games.csv")).set_index("game_id").loc[idx]
    d = pd.DataFrame(index=idx)
    d["season"] = h.season; d["wk"] = h.wk
    d["total"] = h.full_pts + a.full_pts; d["margin"] = h.full_pts - a.full_pts
    d["close_total"] = h.close_total; d["open_total"] = h.open_total; d["close_spread"] = h.close_spread
    d["proj_total"] = h.proj + a.proj; d["proj_margin"] = h.proj - a.proj
    d["mu_pts"] = h.mu_pts
    d["shoot_pts"] = h.O_pts + a.O_pts + h.Dt_pts + a.Dt_pts
    d["off_sum"] = h.O_pts + a.O_pts; d["def_sum"] = h.Dt_pts + a.Dt_pts
    d["worst_def"] = np.minimum(h.Dt_pts, a.Dt_pts)
    d["best_off"] = np.maximum(h.O_pts, a.O_pts)
    d["pace"] = h.O_plays + a.O_plays + h.Dt_plays + a.Dt_plays
    d["pass_mix"] = h.O_pass_rate + a.O_pass_rate
    d["sr_sum"] = h.O_sr + a.O_sr + h.Dt_sr + a.Dt_sr
    d["xr_sum"] = h.O_xr + a.O_xr + h.Dt_xr + a.Dt_xr
    # per-side rush/pass edges (offense strength + opposing defense weakness), yards per attempt
    d["pass_edge_h"] = h.O_pass_ypa + a.Dt_pass_ypa; d["pass_edge_a"] = a.O_pass_ypa + h.Dt_pass_ypa
    d["rush_edge_h"] = h.O_rush_ypc + a.Dt_rush_ypc; d["rush_edge_a"] = a.O_rush_ypc + h.Dt_rush_ypc
    d["pass_edge"] = d.pass_edge_h + d.pass_edge_a; d["rush_edge"] = d.rush_edge_h + d.rush_edge_a
    d["max_edge"] = np.maximum(np.maximum(d.pass_edge_h, d.pass_edge_a), np.maximum(d.rush_edge_h, d.rush_edge_a))
    # alignment: offense lean x defense weakness lean (z-scored within the full sample)
    zo_h = z(h.O_pass_ypa) - z(h.O_rush_ypc); zd_a = z(a.Dt_pass_ypa) - z(a.Dt_rush_ypc)
    zo_a = z(a.O_pass_ypa) - z(a.O_rush_ypc); zd_h = z(h.Dt_pass_ypa) - z(h.Dt_rush_ypc)
    d["align"] = zo_h * zd_a + zo_a * zd_h
    d["align_max"] = np.maximum(zo_h * zd_a, zo_a * zd_h)
    d["blowout"] = np.clip(np.abs(d.proj_margin) - 21, 0, None)
    d["indoor"] = g.indoor.fillna(False).astype(float)
    d["conf_game"] = g.conf_game.astype(float); d["neutral"] = g.neutral.astype(float)
    d["games_before"] = np.minimum(h.games_before, a.games_before)
    d["open_move"] = d.close_total - d.open_total
    d["model_edge"] = d.proj_total - d.close_total
    d["over"] = np.where(d.total > d.close_total, 1.0, np.where(d.total < d.close_total, 0.0, np.nan))
    return d, f


SIGNALS = [
    ("proj_total", "Projected total (the SCORE)"),
    ("shoot_pts", "Shootout index: both offenses + both defenses (points ratings)"),
    ("off_sum", "Both offenses' ratings"),
    ("def_sum", "Both defenses' allowed ratings"),
    ("worst_def", "The BETTER of the two defenses (high = both bad)"),
    ("best_off", "The better offense"),
    ("pace", "Combined tempo (plays above avg)"),
    ("pass_mix", "Combined pass-rate rating"),
    ("sr_sum", "Combined success-rate ratings"),
    ("xr_sum", "Combined explosive-rate ratings"),
    ("pass_edge", "Pass edge, both sides (O pass yds/att + D pass yds/att allowed)"),
    ("rush_edge", "Rush edge, both sides"),
    ("max_edge", "Biggest single-side rush or pass edge"),
    ("align", "Run/pass alignment: offense lean × defense weakness lean"),
    ("align_max", "Best single-side run/pass alignment"),
    ("blowout", "Blowout size (|rating margin| beyond 21)"),
    ("close_total", "Posted total (level)"),
    ("open_move", "Open→close move of the total"),
    ("model_edge", "Model edge vs closing total"),
    ("wk", "Week of season"),
    ("indoor", "Indoor venue"),
    ("conf_game", "Conference game"),
]


def quintiles(d, col, k=5):
    dd = d.dropna(subset=[col, "over"])
    if dd[col].nunique() < k:
        # binary / discrete
        rows = []
        for v, x in dd.groupby(col):
            rows.append({"bucket": str(v), "n": int(len(x)), "over_rate": float(x.over.mean()),
                         "avg_total": float(x.total.mean()), "avg_close": float(x.close_total.mean())})
        return rows
    q = pd.qcut(dd[col], k, labels=False, duplicates="drop")
    rows = []
    for b in sorted(q.unique()):
        x = dd[q == b]
        rows.append({"bucket": int(b) + 1, "n": int(len(x)), "over_rate": float(x.over.mean()),
                     "avg_total": float(x.total.mean()), "avg_close": float(x.close_total.mean()),
                     "signal_mean": float(x[col].mean())})
    return rows


def slope_t(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    X = np.column_stack([np.ones(len(x)), x]); b, *_ = lstsq(X, y, rcond=None)
    r = y - X @ b; se = np.sqrt(np.sum(r ** 2) / max(len(y) - 2, 1) / max(np.sum((x - x.mean()) ** 2), 1e-9))
    return float(b[1]), float(b[1] / se) if se > 0 else 0.0, int(len(y))


def screen(d):
    out = {}
    tr = d[d.season.isin(TRAIN)]; ho = d[d.season.isin(HOLD)]
    for col, label in SIGNALS:
        res = {"label": label}
        for lbl, x in (("train", tr), ("holdout", ho)):
            xx = x.dropna(subset=[col, "close_total"])
            res[f"corr_total_{lbl}"] = float(np.corrcoef(xx[col], xx.total)[0, 1]) if xx[col].nunique() > 1 else 0.0
            res[f"corr_close_{lbl}"] = float(np.corrcoef(xx[col], xx.close_total)[0, 1]) if xx[col].nunique() > 1 else 0.0
            s, t, n = slope_t(xx[col], xx.total - xx.close_total)
            res[f"resid_slope_{lbl}"] = s; res[f"resid_t_{lbl}"] = t
            res[f"quintiles_{lbl}"] = quintiles(xx, col)
        # per-season: over rate in the top quintile (defined within season) minus bottom quintile
        ps = []
        for s, x in d.dropna(subset=[col, "over"]).groupby("season"):
            if x[col].nunique() < 5:
                continue
            q = pd.qcut(x[col], 5, labels=False, duplicates="drop")
            top = x[q == q.max()].over.mean(); bot = x[q == 0].over.mean()
            ps.append({"season": int(s), "top_over": float(top), "bottom_over": float(bot), "n_top": int((q == q.max()).sum())})
        res["per_season"] = ps
        out[col] = res
    return out


def fitted_over_model(d):
    """(actual - close) ~ signals, fit on train, evaluate sign-hit on holdout by predicted-edge bucket."""
    cols = ["model_edge", "shoot_pts", "pace", "pass_mix", "pass_edge", "rush_edge", "align", "blowout",
            "close_total", "indoor", "conf_game", "wk"]
    dd = d.dropna(subset=cols + ["close_total", "over"]).copy()
    tr = dd[dd.season.isin(TRAIN)]; ho = dd[dd.season.isin(HOLD)]
    X = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cols])
    y = (tr.total - tr.close_total).values
    b, *_ = lstsq(X, y, rcond=None)
    Xh = np.column_stack([np.ones(len(ho))] + [ho[c].values for c in cols])
    ph = Xh @ b
    # in-sample LOSO too
    p_tr = np.zeros(len(tr))
    for s in tr.season.unique():
        m = tr.season.values == s
        bb, *_ = lstsq(X[~m], y[~m], rcond=None)
        p_tr[m] = X[m] @ bb
    def hits(p, yy):
        rows = []
        for th in (0, 1, 2, 3, 4):
            m = (np.abs(p) >= th) & (yy != 0)
            if m.sum() == 0:
                continue
            rows.append({"min_pred_edge": th, "n": int(m.sum()), "hit": float((np.sign(p[m]) == np.sign(yy[m])).mean()),
                         "over_share": float((p[m] > 0).mean())})
        return rows
    yh = (ho.total - ho.close_total).values
    return {"coef": dict(zip(["const"] + cols, [float(v) for v in b])),
            "train_loso_hits": hits(p_tr, y), "holdout_hits": hits(ph, yh),
            "holdout_corr_pred_vs_resid": float(np.corrcoef(ph, yh)[0, 1]) if len(ho) > 2 else None,
            "n_train": int(len(tr)), "n_holdout": int(len(ho))}


def parsimonious_over_model(d):
    """(actual - close) ~ posted total level + rush edge + model edge, the three things that showed any stability."""
    cols = ["close_total", "rush_edge", "model_edge"]
    dd = d.dropna(subset=cols + ["over"]).copy()
    tr = dd[dd.season.isin(TRAIN)]; ho = dd[dd.season.isin(HOLD)]
    X = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cols]); y = (tr.total - tr.close_total).values
    b, *_ = lstsq(X, y, rcond=None)
    p_tr = np.zeros(len(tr))
    for s in tr.season.unique():
        m = tr.season.values == s
        bb, *_ = lstsq(X[~m], y[~m], rcond=None); p_tr[m] = X[m] @ bb
    Xh = np.column_stack([np.ones(len(ho))] + [ho[c].values for c in cols]); yh = (ho.total - ho.close_total).values
    ph = Xh @ b
    def hits(p, yy):
        rows = []
        for th in (0, 0.5, 1, 1.5, 2, 3):
            m = (np.abs(p) >= th) & (yy != 0)
            if m.sum() == 0:
                continue
            rows.append({"min_pred_edge": th, "n": int(m.sum()), "hit": float((np.sign(p[m]) == np.sign(yy[m])).mean()),
                         "over_share": float((p[m] > 0).mean()),
                         "over_hit": float((yy[m][p[m] > 0] > 0).mean()) if (p[m] > 0).sum() else None,
                         "under_hit": float((yy[m][p[m] < 0] < 0).mean()) if (p[m] < 0).sum() else None})
        return rows
    per_season = []
    allp = np.concatenate([p_tr, ph]); ally = np.concatenate([y, yh]); alls = np.concatenate([tr.season.values, ho.season.values])
    for s in sorted(set(alls)):
        m = (alls == s) & (ally != 0) & (np.abs(allp) >= 1)
        per_season.append({"season": int(s), "n": int(m.sum()), "hit": float((np.sign(allp[m]) == np.sign(ally[m])).mean()) if m.sum() else None})
    return {"coef": dict(zip(["const"] + cols, [float(v) for v in b])), "train_loso_hits": hits(p_tr, y),
            "holdout_hits": hits(ph, yh), "per_season_edge1": per_season, "sd_pred_train": float(np.std(p_tr))}


def total_brackets(d):
    bins = [0, 44.5, 49.5, 54.5, 59.5, 64.5, 200]; labels = ["<45", "45-49.5", "50-54.5", "55-59.5", "60-64.5", "65+"]
    dd = d.dropna(subset=["close_total", "over"]).copy()
    dd["tb"] = pd.cut(dd.close_total, bins, labels=labels)
    out = []
    for lab in labels:
        tr = dd[(dd.tb == lab) & dd.season.isin(TRAIN)]; ho = dd[(dd.tb == lab) & dd.season.isin(HOLD)]
        out.append({"bracket": lab, "n_train": int(len(tr)), "over_train": float(tr.over.mean()) if len(tr) else None,
                    "n_holdout": int(len(ho)), "over_holdout": float(ho.over.mean()) if len(ho) else None})
    hi = [{"season": int(s), "n": int((x.close_total >= 60).sum()), "over": float(x[x.close_total >= 60].over.mean())} for s, x in dd.groupby("season")]
    lo = [{"season": int(s), "n": int((x.close_total < 50).sum()), "over": float(x[x.close_total < 50].over.mean())} for s, x in dd.groupby("season")]
    return {"brackets": out, "hi60_by_season": hi, "lo50_by_season": lo}


def blowout_fix(f):
    """Does a blowout term fix the tail?  Team-level: SCORE + blow_fav + blow_dog."""
    tr = f[f.season.isin(TRAIN)].copy(); ho = f[f.season.isin(HOLD)].copy()
    base = FEATURE_SETS[SCORE_MODEL]; ext = base + ["blow_fav", "blow_dog"]
    out = {}
    for name, cols in (("SCORE", base), ("SCORE + blowout terms", ext)):
        p_tr = loso(tr, cols); beta = fit_ols(tr, cols); p_ho = predict(ho, cols, beta)
        big_tr = tr.rmargin.abs() >= 28; big_ho = ho.rmargin.abs() >= 28
        out[name] = {"mae_train": float(np.abs(tr.full_pts - p_tr).mean()), "mae_holdout": float(np.abs(ho.full_pts - p_ho).mean()),
                     "mae_train_big": float(np.abs(tr.full_pts - p_tr)[big_tr].mean()), "mae_holdout_big": float(np.abs(ho.full_pts - p_ho)[big_ho].mean()),
                     "bias_train_big_fav": float((tr.full_pts - p_tr)[tr.rmargin >= 28].mean()),
                     "bias_holdout_big_fav": float((ho.full_pts - p_ho)[ho.rmargin >= 28].mean()),
                     "n_big_train": int(big_tr.sum()), "n_big_holdout": int(big_ho.sum()),
                     "coef": dict(zip(["const"] + cols, [float(v) for v in beta]))}
    return out


def rushpass_stage_a(f):
    """Do rush/pass ratings add to the SCORE for team points?"""
    tr = f[f.season.isin(TRAIN)].copy(); ho = f[f.season.isin(HOLD)].copy()
    base = FEATURE_SETS[SCORE_MODEL]
    # interaction: offense pass strength x defense pass weakness (z-scored), same for rush
    for d in (tr, ho, f):
        d["pass_int"] = z(d.O_pass_ypa) * z(d.D_pass_ypa)
        d["rush_int"] = z(d.O_rush_ypc) * z(d.D_rush_ypc)
        d["pass_edge_t"] = d.O_pass_ypa + d.D_pass_ypa
        d["rush_edge_t"] = d.O_rush_ypc + d.D_rush_ypc
        d["align_t"] = (z(d.O_pass_ypa) - z(d.O_rush_ypc)) * (z(d.D_pass_ypa) - z(d.D_rush_ypc))
    sets = {
        "SCORE": base,
        "SCORE + rush/pass yds per attempt (O and D)": base + ["O_rush_ypc", "D_rush_ypc", "O_pass_ypa", "D_pass_ypa"],
        "SCORE + rush/pass success + explosive": base + ["O_rush_sr", "D_rush_sr", "O_pass_sr", "D_pass_sr", "O_rush_xr", "D_rush_xr", "O_pass_xr", "D_pass_xr"],
        "SCORE + pass rate (O and D)": base + ["O_pass_rate", "D_pass_rate"],
        "SCORE + run/pass matchup interactions": base + ["pass_int", "rush_int", "align_t"],
        "SCORE + everything rush/pass": base + ["O_rush_ypc", "D_rush_ypc", "O_pass_ypa", "D_pass_ypa", "O_pass_rate", "D_pass_rate", "pass_int", "rush_int", "align_t"],
    }
    out = {}
    for name, cols in sets.items():
        p_tr = loso(tr, cols); beta = fit_ols(tr, cols); p_ho = predict(ho, cols, beta)
        out[name] = {"mae_train": float(np.abs(tr.full_pts - p_tr).mean()), "corr_train": float(np.corrcoef(tr.full_pts, p_tr)[0, 1]),
                     "mae_holdout": float(np.abs(ho.full_pts - p_ho).mean()), "corr_holdout": float(np.corrcoef(ho.full_pts, p_ho)[0, 1]),
                     "coef": dict(zip(["const"] + cols, [float(v) for v in beta]))}
    # univariate on the splits
    uni = {c: float(np.corrcoef(tr[c], tr.full_pts)[0, 1]) for c in ["O_rush_ypc", "O_pass_ypa", "D_rush_ypc", "D_pass_ypa", "O_rush_sr", "O_pass_sr", "D_rush_sr", "D_pass_sr", "O_pass_rate", "D_pass_rate", "pass_int", "rush_int", "align_t", "pass_edge_t", "rush_edge_t"]}
    # persistence of the split stats (first 6 weeks vs 8+), from teamgames
    t = pd.read_csv(os.path.join(DATA, "teamgames.csv")); t = t[t.team_fbs & (t.stype == 2) & (t.plays >= 30)].copy()
    t["rush_ypc"] = t.rush_yds / t.rush_att.clip(lower=1); t["pass_ypa"] = t.pass_yds / t.pass_att.clip(lower=1)
    t["pass_rate"] = t.pass_att / (t.rush_att + t.pass_att)
    pers = {}
    for s in ("rush_ypc", "pass_ypa", "pass_rate"):
        a = t[t.week <= 6].groupby(["season", "team_id"])[s].mean(); b = t[t.week >= 8].groupby(["season", "team_id"])[s].mean()
        j = pd.concat([a, b], axis=1, keys=["h1", "h2"]).dropna()
        pers[s] = float(np.corrcoef(j.h1, j.h2)[0, 1])
    return {"models": out, "univariate": uni, "persistence": pers}


def main():
    f, _ = load("features")
    d, f2 = assemble(f)
    res = {"n_games": int(len(d)), "screen": screen(d), "over_model": fitted_over_model(d),
           "over_model_simple": parsimonious_over_model(d), "total_brackets": total_brackets(d),
           "blowout_fix": blowout_fix(f2), "rushpass": rushpass_stage_a(f2)}
    # baseline over rate
    for lbl, x in (("train", d[d.season.isin(TRAIN)]), ("holdout", d[d.season.isin(HOLD)])):
        res[f"base_over_rate_{lbl}"] = float(x.over.mean())
    res["over_rate_by_season"] = [{"season": int(s), "over": float(x.over.mean()), "n": int(x.over.notna().sum())} for s, x in d.groupby("season")]
    json.dump(res, open(os.path.join(DATA, "overs_results.json"), "w"), indent=1, default=float)
    d.to_csv(os.path.join(DATA, "game_signals.csv"))
    # ---- report
    print(f"games: {len(d)} | base over rate train {res['base_over_rate_train']:.3f} holdout {res['base_over_rate_holdout']:.3f}")
    print("over rate by season:", [(r['season'], round(r['over'], 3)) for r in res["over_rate_by_season"]])
    print("\n=== SIGNAL SCREEN  (corr with actual total | over-rate top vs bottom quintile, train | holdout | resid t train/holdout) ===")
    for col, r in res["screen"].items():
        qt = r["quintiles_train"]; qh = r["quintiles_holdout"]
        top_t = qt[-1]["over_rate"] if qt else float("nan"); bot_t = qt[0]["over_rate"] if qt else float("nan")
        top_h = qh[-1]["over_rate"] if qh else float("nan"); bot_h = qh[0]["over_rate"] if qh else float("nan")
        seasons_pos = sum(1 for p in r["per_season"] if p["top_over"] > p["bottom_over"])
        print(f"{r['label']:<70} corr {r['corr_total_train']:+.3f} | Q5 {top_t:.3f} vs Q1 {bot_t:.3f} | 2025: Q5 {top_h:.3f} vs Q1 {bot_h:.3f} | t {r['resid_t_train']:+.1f} / {r['resid_t_holdout']:+.1f} | top>bottom in {seasons_pos}/{len(r['per_season'])} seasons")
    print("\n=== FITTED OVER MODEL (actual - close ~ signals) ===")
    om = res["over_model"]
    print("train LOSO:", [(h['min_pred_edge'], h['n'], round(h['hit'], 3)) for h in om["train_loso_hits"]])
    print("holdout   :", [(h['min_pred_edge'], h['n'], round(h['hit'], 3)) for h in om["holdout_hits"]], "| corr(pred, resid) holdout:", round(om["holdout_corr_pred_vs_resid"], 3))
    om = res["over_model_simple"]
    print("\n=== SIMPLE OVER MODEL (posted level + rush edge + model edge) ===", {k: round(v, 3) for k, v in om["coef"].items()})
    print("train LOSO:", [(h['min_pred_edge'], h['n'], round(h['hit'], 3), 'over-share', round(h['over_share'], 2)) for h in om["train_loso_hits"]])
    print("holdout   :", [(h['min_pred_edge'], h['n'], round(h['hit'], 3), 'over-share', round(h['over_share'], 2)) for h in om["holdout_hits"]])
    print("per season (|edge|>=1):", [(r['season'], r['n'], round(r['hit'], 3) if r['hit'] is not None else None) for r in om["per_season_edge1"]])
    print("\n=== BLOWOUT FIX ===")
    for k, v in res["blowout_fix"].items():
        print(f"{k:<26} MAE train {v['mae_train']:.3f} holdout {v['mae_holdout']:.3f} | big-mismatch MAE train {v['mae_train_big']:.2f} holdout {v['mae_holdout_big']:.2f} | fav bias (actual-pred) train {v['bias_train_big_fav']:+.2f} holdout {v['bias_holdout_big_fav']:+.2f} (n={v['n_big_train']}/{v['n_big_holdout']})")
    print("\n=== RUSH / PASS, Stage A (team points) ===")
    for k, v in res["rushpass"]["models"].items():
        print(f"{k:<50} MAE train {v['mae_train']:.3f} corr {v['corr_train']:.3f} | holdout {v['mae_holdout']:.3f} corr {v['corr_holdout']:.3f}")
    print("univariate corr with team points:", {k: round(v, 3) for k, v in res["rushpass"]["univariate"].items()})
    print("persistence (wk<=6 vs wk>=8):", {k: round(v, 3) for k, v in res["rushpass"]["persistence"].items()})
    return res


if __name__ == "__main__":
    main()
