#!/usr/bin/env python3
"""
ncaaf_shootcal.py — what the Shootout percentile has actually been worth to an over bettor.

For every historical game, compute its Shootout percentile exactly the way the live board does
(rank the model's projected total against the pool of ALL PRIOR seasons' projected totals), then
bucket and measure: straight over rate vs the closing total, and the clear rate for alt-overs
bought 12 / 16 / 20 points down. Writes data/shoot_calibration.json.
"""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
BUCK = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 90), (90, 101)]


def main():
    s = pd.read_csv(os.path.join(DATA, "game_signals.csv"), index_col=0).dropna(subset=["close_total"])
    pct = np.full(len(s), np.nan)
    for yr in sorted(s.season.unique()):
        if yr == s.season.min():
            continue
        pool = np.sort(s[s.season < yr].proj_total.values)
        m = (s.season == yr).values
        pct[m] = np.searchsorted(pool, s.proj_total.values[m]) / len(pool) * 100
    s = s.assign(shoot=pct).dropna(subset=["shoot"])

    rows = []
    for lo, hi in BUCK:
        x = s[(s.shoot >= lo) & (s.shoot < hi)]
        r = {"lo": lo, "hi": hi, "n": int(len(x)),
             "avg_posted": float(x.close_total.mean()),
             "avg_actual": float(x.total.mean()),
             "over_rate": float((x.total > x.close_total).mean())}
        for k in (12, 16, 20):
            r[f"clear{k}"] = float((x.total > x.close_total - k).mean())
        r["need600"] = 600 / 700.0
        r["roi16_at_600"] = float((r["clear16"] * 100 - (1 - r["clear16"]) * 600) / 600)
        rows.append(r)

    # correlation check: is the relationship monotone, or a couple of noisy buckets?
    sp = float(pd.Series(s.shoot).corr(pd.Series((s.total > s.close_total - 16).astype(float)), method="spearman"))
    out = {"buckets": rows, "n": int(len(s)), "spearman_shoot_vs_clear16": sp,
           "seasons": [int(s.season.min()), int(s.season.max())]}
    json.dump(out, open(os.path.join(DATA, "shoot_calibration.json"), "w"), indent=1)
    print(f"wrote shoot_calibration.json  n={len(s)}  spearman(shoot, clears-16) = {sp:+.3f}")
    for r in rows:
        print(f"  {r['lo']:>3}-{r['hi']:<3} n={r['n']:>5}  posted {r['avg_posted']:.1f}  actual {r['avg_actual']:.1f}  "
              f"over {r['over_rate']*100:.1f}%  clear-16 {r['clear16']*100:.1f}%  ROI@-600 {r['roi16_at_600']*100:+.1f}%")


if __name__ == "__main__":
    main()
