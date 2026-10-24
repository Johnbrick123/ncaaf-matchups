#!/usr/bin/env python3
"""
ncaaf_floor.py — the scoring FLOOR: what predicts P(total > posted − k), i.e. whether an alternate over
bought k points below the posted total clears.

Base facts (2015–2025, 7,625 FBS-vs-FBS games with closing totals): total − close has SD ≈ 16.3 and is
almost exactly normal in the lower tail. P(clear −11) = 74.8% (fair −297), P(clear −16) = 84.4% (fair −542).
The tail is tighter on low posted totals and wider on high ones. So the question is: which matchups have a
lower tail thinner than that curve implies?

Features (all as-of, both teams): offense ratings (sum and the weaker side), defense ratings (sum and the
better side), success-rate and explosive-rate sums, tempo, pass mix, rush/pass edge, spread size, each
team's scoring VOLATILITY (SD of its residuals vs the rating expectation over prior games) and raw FLOOR
(20th percentile of points in prior games), model gap to the posted total, indoor, week.

Outputs data/floor_results.json and data/game_floor.csv. Fits logistic models for k = 11 and 16 on
2015–24 with close_total as a control, evaluates 2025 by predicted-probability decile.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from ncaaf_backtest import load, TRAIN, HOLD

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
KS = (8, 11, 14, 16, 20)


def team_volatility(f):
    """as-of scoring volatility and raw floor per team-game (prior games this season only)."""
    f = f.copy()
    f["expect"] = f.mu_pts + f.O_pts + f.D_pts + f.h_pts * f.home_adj
    f["resid"] = f.full_pts - f.expect
    f = f.sort_values(["season", "team_id", "date"])
    g = f.groupby(["season", "team_id"])
    f["vol"] = g.resid.transform(lambda x: x.shift(1).expanding(min_periods=3).std())
    f["absres"] = g.resid.transform(lambda x: x.shift(1).expanding(min_periods=3).apply(lambda v: np.abs(v).mean(), raw=True))
    f["floor20"] = g.full_pts.transform(lambda x: x.shift(1).expanding(min_periods=3).quantile(0.2))
    f["minpts"] = g.full_pts.transform(lambda x: x.shift(1).expanding(min_periods=3).min())
    f["pts_sd"] = g.full_pts.transform(lambda x: x.shift(1).expanding(min_periods=3).std())
    return f


def assemble(f):
    f = team_volatility(f)
    lg_vol = f.vol.mean(); lg_floor = f.floor20.mean(); lg_sd = f.pts_sd.mean(); lg_abs = f.absres.mean()
    h = f[f.is_listed_home == 1].set_index("game_id"); a = f[f.is_listed_home == 0].set_index("game_id")
    idx = h.index.intersection(a.index); h, a = h.loc[idx], a.loc[idx]
    s = pd.read_csv(os.path.join(DATA, "game_signals.csv"), index_col=0)
    s = s.loc[s.index.intersection(idx)]
    d = s.copy()
    d["off_min"] = np.minimum(h.loc[d.index, "O_pts"], a.loc[d.index, "O_pts"])
    d["off_max"] = np.maximum(h.loc[d.index, "O_pts"], a.loc[d.index, "O_pts"])
    d["def_max"] = np.maximum(h.loc[d.index, "Dt_pts"], a.loc[d.index, "Dt_pts"])   # the WORSE defense
    d["vol_sum"] = h.loc[d.index, "vol"].fillna(lg_vol) + a.loc[d.index, "vol"].fillna(lg_vol)
    d["vol_max"] = np.maximum(h.loc[d.index, "vol"].fillna(lg_vol), a.loc[d.index, "vol"].fillna(lg_vol))
    d["absres_sum"] = h.loc[d.index, "absres"].fillna(lg_abs) + a.loc[d.index, "absres"].fillna(lg_abs)
    d["floor_sum"] = h.loc[d.index, "floor20"].fillna(lg_floor) + a.loc[d.index, "floor20"].fillna(lg_floor)
    d["floor_min"] = np.minimum(h.loc[d.index, "floor20"].fillna(lg_floor), a.loc[d.index, "floor20"].fillna(lg_floor))
    d["ptsd_sum"] = h.loc[d.index, "pts_sd"].fillna(lg_sd) + a.loc[d.index, "pts_sd"].fillna(lg_sd)
    d["has_vol"] = (h.loc[d.index, "vol"].notna() & a.loc[d.index, "vol"].notna()).astype(float)
    d["sr_min"] = np.minimum(h.loc[d.index, "O_sr"], a.loc[d.index, "O_sr"])
    d["spread_abs"] = d.close_spread.abs()
    d["mkt_dog_total"] = (d.close_total - d.spread_abs) / 2
    d["mkt_fav_total"] = (d.close_total + d.spread_abs) / 2
    d["res"] = d.total - d.close_total
    for k in KS:
        d[f"hit{k}"] = (d.total > d.close_total - k).astype(float)
    return d


FEATURES = [
    ("off_sum", "Both offenses' points ratings"),
    ("off_min", "The WEAKER offense's rating"),
    ("def_sum", "Both defenses' allowed ratings"),
    ("worst_def", "The BETTER defense's allowed rating (high = both bad)"),
    ("def_max", "The worse defense's allowed rating"),
    ("sr_sum", "Combined success-rate ratings (consistency)"),
    ("sr_min", "The weaker offense's success rate"),
    ("xr_sum", "Combined explosive-rate ratings (boom/bust)"),
    ("pace", "Combined tempo"),
    ("pass_mix", "Combined pass-rate rating"),
    ("rush_edge", "Rush edge, both sides"),
    ("pass_edge", "Pass edge, both sides"),
    ("vol_sum", "Both teams' scoring volatility (as-of SD of residuals)"),
    ("vol_max", "The more volatile team"),
    ("floor_sum", "Both teams' raw 20th-percentile points so far"),
    ("floor_min", "The lower team's raw 20th-percentile points"),
    ("ptsd_sum", "Both teams' raw SD of points so far"),
    ("spread_abs", "Spread size"),
    ("mkt_dog_total", "Market-implied underdog team total"),
    ("model_edge", "Model total minus posted total"),
    ("proj_total", "Model projected total"),
    ("close_total", "Posted total"),
    ("wk", "Week of season"),
    ("indoor", "Indoor"),
    ("conf_game", "Conference game"),
]


def logit_effect(d, col, k, controls=("close_total",)):
    """logistic hit_k ~ close_total + feature; returns standardized coefficient and z."""
    dd = d.dropna(subset=[col, f"hit{k}"] + list(controls))
    if dd[col].nunique() < 2:
        return None
    X = dd[list(controls) + [col]].values.astype(float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Xs = (X - mu) / sd
    m = LogisticRegression(C=1e6, max_iter=2000).fit(Xs, dd[f"hit{k}"].values)
    b = m.coef_[0][-1]
    # crude z via one-step SE from the Hessian
    p = m.predict_proba(Xs)[:, 1]; W = p * (1 - p)
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    try:
        cov = np.linalg.inv((Xd * W[:, None]).T @ Xd)
        se = np.sqrt(cov[-1, -1])
    except np.linalg.LinAlgError:
        se = np.nan
    return {"coef_std": float(b), "z": float(b / se) if se else None, "n": int(len(dd))}


def strata_quintiles(d, col, k):
    """hit rate top vs bottom quintile of the feature WITHIN posted-total brackets, averaged (equal weight)."""
    dd = d.dropna(subset=[col, f"hit{k}"]).copy()
    dd["tb"] = pd.cut(dd.close_total, [0, 44.5, 49.5, 54.5, 59.5, 64.5, 200])
    tops, bots = [], []
    for _, x in dd.groupby("tb", observed=True):
        if len(x) < 50 or x[col].nunique() < 5:
            continue
        q = pd.qcut(x[col], 5, labels=False, duplicates="drop")
        tops.append(x[f"hit{k}"][q == q.max()].mean()); bots.append(x[f"hit{k}"][q == 0].mean())
    return {"top": float(np.mean(tops)) if tops else None, "bottom": float(np.mean(bots)) if bots else None}


def screen(d):
    out = {}
    tr = d[d.season.isin(TRAIN)]; ho = d[d.season.isin(HOLD)]
    for col, label in FEATURES:
        r = {"label": label}
        for k in (11, 16):
            e = logit_effect(tr, col, k)
            r[f"logit_k{k}"] = e
            r[f"strata_k{k}_train"] = strata_quintiles(tr, col, k)
            r[f"strata_k{k}_holdout"] = strata_quintiles(ho, col, k)
            # per-season sign consistency (top vs bottom quintile within season, no strata)
            ps = []
            for s, x in d.dropna(subset=[col]).groupby("season"):
                if x[col].nunique() < 5:
                    continue
                q = pd.qcut(x[col], 5, labels=False, duplicates="drop")
                ps.append(float(x[f"hit{k}"][q == q.max()].mean() - x[f"hit{k}"][q == 0].mean()))
            r[f"seasons_pos_k{k}"] = int(sum(1 for v in ps if v > 0)); r[f"n_seasons_k{k}"] = len(ps)
            r[f"mean_gap_k{k}"] = float(np.mean(ps)) if ps else None
        out[col] = r
    return out


def fit_floor_models(d):
    """Logistic P(hit_k) on a handful of inputs; evaluate calibration on 2025 by predicted decile."""
    cols = ["close_total", "off_min", "worst_def", "sr_sum", "xr_sum", "pace", "vol_sum", "spread_abs", "model_edge", "rush_edge"]
    dd = d.dropna(subset=cols + [f"hit{k}" for k in KS]).copy()
    tr = dd[dd.season.isin(TRAIN)]; ho = dd[dd.season.isin(HOLD)]
    out = {"cols": cols, "n_train": int(len(tr)), "n_holdout": int(len(ho)), "models": {}}
    mu, sd = tr[cols].mean(), tr[cols].std()
    for k in KS:
        y = tr[f"hit{k}"].values
        Xs = ((tr[cols] - mu) / sd).values
        m = LogisticRegression(C=1e6, max_iter=3000).fit(Xs, y)
        base = LogisticRegression(C=1e6, max_iter=3000).fit(((tr[["close_total"]] - mu["close_total"]) / sd["close_total"]).values, y)
        # LOSO in train
        p_tr = np.zeros(len(tr))
        for s in tr.season.unique():
            msk = tr.season.values == s
            mm = LogisticRegression(C=1e6, max_iter=3000).fit(Xs[~msk], y[~msk]); p_tr[msk] = mm.predict_proba(Xs[msk])[:, 1]
        Xh = ((ho[cols] - mu) / sd).values; yh = ho[f"hit{k}"].values
        p_ho = m.predict_proba(Xh)[:, 1]
        pb_ho = base.predict_proba(((ho[["close_total"]] - mu["close_total"]) / sd["close_total"]).values)[:, 1]
        pb_tr = base.predict_proba(((tr[["close_total"]] - mu["close_total"]) / sd["close_total"]).values)[:, 1]
        def deciles(p, yy, n=10):
            q = pd.qcut(pd.Series(p), n, labels=False, duplicates="drop")
            return [{"decile": int(b) + 1, "pred": float(p[q == b].mean()), "actual": float(yy[q == b].mean()), "n": int((q == b).sum())} for b in sorted(q.unique())]
        def top_share(p, yy, frac):
            th = np.quantile(p, 1 - frac); msk = p >= th
            return {"share": frac, "n": int(msk.sum()), "pred": float(p[msk].mean()), "actual": float(yy[msk].mean())}
        out["models"][str(k)] = {
            "coef_std": dict(zip(cols, [float(v) for v in m.coef_[0]])), "intercept": float(m.intercept_[0]),
            "mu": {c: float(mu[c]) for c in cols}, "sd": {c: float(sd[c]) for c in cols},
            "base_rate_train": float(y.mean()), "base_rate_holdout": float(yh.mean()),
            "deciles_train_loso": deciles(p_tr, y), "deciles_holdout": deciles(p_ho, yh),
            "top_train": [top_share(p_tr, y, f) for f in (0.1, 0.2, 0.33)],
            "top_holdout": [top_share(p_ho, yh, f) for f in (0.1, 0.2, 0.33)],
            "top_holdout_baseline_total_only": [top_share(pb_ho, yh, f) for f in (0.1, 0.2, 0.33)],
            "top_train_baseline_total_only": [top_share(pb_tr, y, f) for f in (0.1, 0.2, 0.33)],
            "loglik_gain_vs_total_only_holdout": float(np.sum(yh * np.log(p_ho) + (1 - yh) * np.log(1 - p_ho)) - np.sum(yh * np.log(pb_ho) + (1 - yh) * np.log(1 - pb_ho))),
        }
        # per-season: hit rate in the top third of predicted probability, all seasons (LOSO/holdout preds)
        allp = np.concatenate([p_tr, p_ho]); ally = np.concatenate([y, yh]); alls = np.concatenate([tr.season.values, ho.season.values])
        ps = []
        for s in sorted(set(alls)):
            msk = alls == s; pp = allp[msk]; yy = ally[msk]
            th = np.quantile(pp, 2 / 3); m2 = pp >= th
            ps.append({"season": int(s), "n": int(m2.sum()), "top_third_actual": float(yy[m2].mean()), "top_third_pred": float(pp[m2].mean()), "season_base": float(yy.mean())})
        out["models"][str(k)]["per_season_top_third"] = ps
    return out


def base_table(d):
    sd = d.res.std()
    rows = []
    for k in (5, 8, 10, 11, 12, 14, 15, 16, 18, 20, 25):
        hit = (d.total > d.close_total - k)
        p = float(hit.mean()); ps = d.groupby("season").apply(lambda x: (x.total > x.close_total - k).mean())
        rows.append({"k": k, "empirical": p, "normal": float(norm.cdf(k / sd)), "holdout": float(hit[d.season == 2025].mean()),
                     "season_min": float(ps.min()), "season_max": float(ps.max()),
                     "fair_american": float(-100 * p / (1 - p))})
    by_total = []
    d2 = d.copy(); d2["tb"] = pd.cut(d2.close_total, [0, 44.5, 49.5, 54.5, 59.5, 64.5, 200], labels=["<45", "45-49.5", "50-54.5", "55-59.5", "60-64.5", "65+"])
    for lab, x in d2.groupby("tb", observed=True):
        by_total.append({"bracket": str(lab), "n": int(len(x)), "sd": float(x.res.std()), "p10": float(x.res.quantile(.1)),
                         **{f"hit{k}": float((x.total > x.close_total - k).mean()) for k in KS}})
    by_spread = []
    d2["sb"] = pd.cut(d2.close_spread.abs(), [-1, 3.5, 7.5, 14.5, 21.5, 28.5, 99], labels=["0-3.5", "4-7.5", "8-14.5", "15-21.5", "22-28.5", "29+"])
    for lab, x in d2.groupby("sb", observed=True):
        by_spread.append({"bracket": str(lab), "n": int(len(x)), "sd": float(x.res.std()),
                          **{f"hit{k}": float((x.total > x.close_total - k).mean()) for k in KS}})
    return {"sd": float(sd), "skew": float(d.res.skew()), "n": int(len(d)), "offsets": rows, "by_total": by_total, "by_spread": by_spread,
            "quantiles": {str(q): float(d.res.quantile(q)) for q in (0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.9, 0.95)}}


def main():
    f, _ = load("features")
    d = assemble(f)
    d = d.dropna(subset=["close_total"])
    res = {"base": base_table(d), "screen": screen(d), "models": fit_floor_models(d)}
    json.dump(res, open(os.path.join(DATA, "floor_results.json"), "w"), indent=1, default=float)
    d.to_csv(os.path.join(DATA, "game_floor.csv"))
    # report
    print(f"games {len(d)} | SD {res['base']['sd']:.2f} skew {res['base']['skew']:.2f}")
    print("\n=== FEATURE SCREEN for the floor (k=11 | k=16): logistic z (controlling for posted total) | top vs bottom fifth within total brackets, train | 2025 | seasons top>bottom ===")
    for col, r in res["screen"].items():
        l11 = r["logit_k11"]; l16 = r["logit_k16"]
        s11 = r["strata_k11_train"]; s16 = r["strata_k16_train"]; h11 = r["strata_k11_holdout"]; h16 = r["strata_k16_holdout"]
        z11 = (l11 or {}).get("z") or float('nan'); z16 = (l16 or {}).get("z") or float('nan')
        for q in (s11, s16, h11, h16):
            for kk in ("top", "bottom"):
                if q.get(kk) is None: q[kk] = float('nan')
        print(f"{r['label']:<58} z {z11:+5.1f} | {z16:+5.1f} || k11 {s11['top']:.3f} vs {s11['bottom']:.3f} (2025 {h11['top']:.3f} vs {h11['bottom']:.3f}) {r['seasons_pos_k11']}/{r['n_seasons_k11']} || k16 {s16['top']:.3f} vs {s16['bottom']:.3f} (2025 {h16['top']:.3f} vs {h16['bottom']:.3f}) {r['seasons_pos_k16']}/{r['n_seasons_k16']}")
    print("\n=== FLOOR MODELS ===")
    for k, m in res["models"]["models"].items():
        print(f"\nk={k}: base rate train {m['base_rate_train']:.3f} holdout {m['base_rate_holdout']:.3f} | loglik gain vs total-only (holdout) {m['loglik_gain_vs_total_only_holdout']:+.1f}")
        print("  coef (std):", {c: round(v, 2) for c, v in m["coef_std"].items()})
        print("  holdout deciles (pred, actual, n):", [(round(x['pred'], 3), round(x['actual'], 3), x['n']) for x in m["deciles_holdout"]])
        print("  top share holdout  :", [(x['share'], x['n'], round(x['pred'], 3), round(x['actual'], 3)) for x in m["top_holdout"]])
        print("  top share baseline :", [(x['share'], x['n'], round(x['pred'], 3), round(x['actual'], 3)) for x in m["top_holdout_baseline_total_only"]])
        print("  top share train    :", [(x['share'], x['n'], round(x['pred'], 3), round(x['actual'], 3)) for x in m["top_train"]])
        print("  per season top third:", [(x['season'], x['n'], round(x['top_third_actual'], 3), round(x['season_base'], 3)) for x in m["per_season_top_third"]])
    return res


if __name__ == "__main__":
    main()
