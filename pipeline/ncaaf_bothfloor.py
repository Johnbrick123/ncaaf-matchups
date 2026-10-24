#!/usr/bin/env python3
"""
ncaaf_bothfloor.py — the BOTH-TEAMS scoring floor.

Different question from the game total: not "how many points combined" but "how high is the number
that BOTH teams clear?"  A 60-point total split 48-12 has a terrible floor; split 32-28 it has a good one.

Metric per game: P(min(home pts, away pts) > X) for X in 10..27, plus the "floor level" — the highest X
that clears 70% (and 60%).  Fitted as a logistic on the MARKET's implied underdog team total, which
predicts the lower score better than our own model does (corr .408 vs .366) and which our model adds
nothing to out of sample.  So this is a descriptive read on the market's number, not an edge over it.

Writes data/bothfloor_results.json.
"""
import json, os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

BASE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(BASE, "data")
TR = list(range(2015, 2025)); HO = [2025]
XS = list(range(10, 28))

d = pd.read_csv(os.path.join(DATA, "game_minpts.csv"), index_col=0).dropna(subset=["mkt_dog", "minp"])
tr, ho = d[d.season.isin(TR)], d[d.season.isin(HO)]

models, valid = {}, {}
for X in XS:
    y = (tr.minp > X).astype(int).values
    m = LogisticRegression(C=1e6, max_iter=2000).fit(tr[["mkt_dog"]].values, y)
    ph = m.predict_proba(ho[["mkt_dog"]].values)[:, 1]
    yh = (ho.minp > X).astype(int).values
    # calibration on the holdout, by predicted tercile
    q = pd.qcut(pd.Series(ph), 3, labels=False, duplicates="drop")
    cal = [{"bucket": int(b) + 1, "pred": float(ph[q == b].mean()), "actual": float(yh[q == b].mean()),
            "n": int((q == b).sum())} for b in sorted(pd.Series(q).unique())]
    # deployed model: refit on everything
    mf = LogisticRegression(C=1e6, max_iter=2000).fit(d[["mkt_dog"]].values, (d.minp > X).astype(int).values)
    models[str(X)] = {"coef": float(mf.coef_[0][0]), "intercept": float(mf.intercept_[0]),
                      "base_train": float(y.mean()), "base_holdout": float(yh.mean())}
    valid[str(X)] = {"holdout_calibration": cal}

# empirical table by bin, for display
d["bin"] = pd.cut(d.mkt_dog, [0, 15, 18, 20, 22, 24, 26, 28, 100])
tbl = []
for b, x in d.groupby("bin", observed=True):
    row = {"bin": str(b), "n": int(len(x)), "median_min": float(x.minp.median())}
    for X in (10, 13, 17, 20, 24):
        row[f"p{X}"] = float((x.minp > X).mean())
    row["floor70"] = max([X for X in range(0, 40) if (x.minp > X).mean() >= 0.70], default=0)
    row["floor60"] = max([X for X in range(0, 40) if (x.minp > X).mean() >= 0.60], default=0)
    tbl.append(row)

out = {"n": int(len(d)), "xs": XS, "models": models, "validation": valid, "empirical_bins": tbl,
       "base_rates": {str(X): float((d.minp > X).mean()) for X in (10, 13, 17, 20, 24)},
       "predictor_corr": {"market_dog_total": float(np.corrcoef(d.mkt_dog, d.minp)[0, 1]),
                          "model_weaker_side": float(np.corrcoef(d.proj_min, d.minp)[0, 1]),
                          "posted_total": float(np.corrcoef(d.close_total, d.minp)[0, 1])}}
json.dump(out, open(os.path.join(DATA, "bothfloor_results.json"), "w"), indent=1)
print(f"wrote bothfloor_results.json | n={len(d)} | base P(both>20)={out['base_rates']['20']:.3f}")
for X in (13, 17, 20, 24):
    c = valid[str(X)]["holdout_calibration"]
    print(f"  X={X}: 2025 calibration by tercile " + " | ".join(f"pred {r['pred']:.2f} act {r['actual']:.2f}" for r in c))
