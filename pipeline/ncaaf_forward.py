#!/usr/bin/env python3
"""
ncaaf_forward.py — the 2026 forward test: how the live model has done against the closing line.

  python3 ncaaf_forward.py [through_week]

Each completed FBS-vs-FBS game is projected with the as-of ratings the board would have shown that
week (features.csv) and THAT WEEK's as-of league mean (mu_pts), with SCORE coefficients fitted on
seasons before the current one. Compared with ESPN's closing total/spread.

Correction (Oct 22 2026): the ad-hoc version used Sep 24 – Oct 8 projected every past game with the
CURRENT week's league mean. The as-of mean fell from 26.2 (week 1) to ~24, so week-1 totals were
projected ~4.5 points too low and the early weeks looked far worse than they were. Spreads were
unaffected (the mean cancels in the margin). This script is the fixed, reusable version.
"""
import json, os, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(BASE, "data")
sys.path.insert(0, BASE)
from ncaaf_backtest import load as load_eval, fit_ols, FEATURE_SETS, SCORE_MODEL  # noqa: E402

from datetime import date as _date
_t = _date.today()
SEASON = int(os.environ.get("SEASON") or (_t.year if _t.month >= 7 else _t.year - 1))
COLS = FEATURE_SETS[SCORE_MODEL]


def main(through=None):
    hist, _ = load_eval("features")
    beta = fit_ols(hist[hist.season < SEASON], COLS)
    C = dict(zip(["const"] + COLS, beta))
    f = pd.read_csv(os.path.join(DATA, "features.csv"))
    f = f[(f.season == SEASON) & f.team_fbs & f.opp_fbs & f.pts.notna()].copy()
    rm = (f.O_pts - f.Dt_pts) - (f.Oo_pts - f.D_pts)
    x = {"O_pts": f.O_pts, "D_pts": f.D_pts, "O_sr": f.O_sr, "D_sr": f.D_sr, "O_xr": f.O_xr, "D_xr": f.D_xr,
         "home_adj": ((f.home == 1) & (f.neutral == 0)).astype(float),
         "blow_fav": np.clip(rm - 21, 0, None), "blow_dog": np.clip(rm + 21, None, 0)}
    f["proj"] = f.mu_pts + C["const"] + sum(C[c] * x[c] for c in COLS)
    keep = f.groupby("game_id").size().eq(2); f = f[f.game_id.isin(keep[keep].index)]
    rows = []
    for gid, g in f.groupby("game_id"):
        g = g.sort_values("home"); a, h = g.iloc[0], g.iloc[1]
        rows.append(dict(wk=int(h.wk), pt=a.proj + h.proj, act=a.pts + h.pts, pm=h.proj - a.proj,
                         am=h.pts - a.pts, ct=h.close_total, cs=h.close_spread_team))
    G = pd.DataFrame(rows)
    if through:
        G = G[G.wk <= through]
    L = G.dropna(subset=["ct"]); S = L.dropna(subset=["cs"])
    tot_hit = lambda z: float(((z.pt > z.ct) == (z.act > z.ct))[z.act != z.ct].mean())
    spr_hit = lambda z: float(((z.pm > -z.cs) == (z.am > -z.cs))[z.am != -z.cs].mean())
    n = len(L); th = tot_hit(L)
    big = L[(L.pt - L.ct).abs() >= 7]
    out = {
        "through_week": int(G.wk.max()), "n": n,
        "total_mae": {"model": float((L.pt - L.act).abs().mean()), "market": float((L.ct - L.act).abs().mean())},
        "margin_mae": {"model": float((S.pm - S.am).abs().mean()), "market": float((-S.cs - S.am).abs().mean())},
        "su": {"model": float((np.sign(G.pm) == np.sign(G.am)).mean()), "market": float((np.sign(-S.cs) == np.sign(S.am)).mean())},
        "totals_vs_close": th, "totals_se": float(np.sqrt(th * (1 - th) / n)), "spreads_vs_close": spr_hit(S),
        "over_rate": float((L.act > L.ct).mean()),
        "totals_by_week": {int(w): tot_hit(z) for w, z in L.groupby("wk")},
        "spreads_by_week": {int(w): spr_hit(z) for w, z in S.groupby("wk")},
        "big_total_gap": {"n": int(len(big)), "hit": tot_hit(big) if len(big) else None},
        "brackets": {},
    }
    L = L.assign(br=pd.cut(L.ct, [0, 44.5, 49.5, 54.5, 59.5, 64.5, 200], labels=["<45", "45-49.5", "50-54.5", "55-59.5", "60-64.5", "65+"]))
    for b, z in L.groupby("br", observed=True):
        out["brackets"][str(b)] = {"n": int(len(z)), "over": float((z.act > z.ct).mean()),
                                   "clear16": float((z.act > z.ct - 16).mean()), "clear20": float((z.act > z.ct - 20).mean())}
    lo, hi = L[L.ct < 50], L[L.ct >= 55]
    out["low_vs_high"] = {"lt50": {"n": int(len(lo)), "clear16": float((lo.act > lo.ct - 16).mean())},
                          "ge55": {"n": int(len(hi)), "clear16": float((hi.act > hi.ct - 16).mean())}}
    json.dump(out, open(os.path.join(DATA, "forward_test.json"), "w"), indent=1)
    p = lambda v: f"{v*100:.1f}%"
    print(f"2026 forward test through week {out['through_week']}: {n} games")
    print(f"  total MAE  model {out['total_mae']['model']:.2f} | market {out['total_mae']['market']:.2f}")
    print(f"  margin MAE model {out['margin_mae']['model']:.2f} | market {out['margin_mae']['market']:.2f}")
    print(f"  winner SU  model {p(out['su']['model'])} | market {p(out['su']['market'])}")
    print(f"  totals vs closer {p(th)} (SE {out['totals_se']*100:.1f}) | spreads vs closer {p(out['spreads_vs_close'])} | over rate {p(out['over_rate'])}")
    print("  totals by week:  " + "  ".join(f"wk{w} {v*100:.0f}%" for w, v in out["totals_by_week"].items()))
    print("  spreads by week: " + "  ".join(f"wk{w} {v*100:.0f}%" for w, v in out["spreads_by_week"].items()))
    print(f"  model 7+ off the closing total: n={out['big_total_gap']['n']}, right {p(out['big_total_gap']['hit'])}")
    print("  clears -16 by posted bracket: " + " | ".join(f"{b} {v['clear16']*100:.0f}% (n={v['n']})" for b, v in out["brackets"].items()))
    print(f"  posted <50: {p(out['low_vs_high']['lt50']['clear16'])} (n={out['low_vs_high']['lt50']['n']}) | posted 55+: {p(out['low_vs_high']['ge55']['clear16'])} (n={out['low_vs_high']['ge55']['n']})")
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
