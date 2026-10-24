#!/usr/bin/env python3
"""
ncaaf_minpts.py — builds data/game_minpts.csv, the input ncaaf_bothfloor.py reads.

One row per FBS-vs-FBS game (index = game_id) from data/game_projections.csv (written by
ncaaf_backtest.py):
  minp      the LOWER of the two teams' final scores (the "both teams clear X" target)
  mkt_dog   the market's implied underdog team total, (close_total - |close_spread|) / 2
  proj_min  this model's weaker-side projection, min(home_proj, away_proj)
  close_total, season
Restored Oct 8 2026: the original was an ad-hoc step during the Sep 16 study and was never saved.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

g = pd.read_csv(os.path.join(DATA, "game_projections.csv"), index_col=0)
d = pd.DataFrame(index=g.index)
d["season"] = g.season
d["minp"] = np.minimum(g.home_pts, g.away_pts)
d["mkt_dog"] = (g.close_total - g.close_spread.abs()) / 2
d["proj_min"] = np.minimum(g.home_proj, g.away_proj)
d["close_total"] = g.close_total
d.to_csv(os.path.join(DATA, "game_minpts.csv"))
dd = d.dropna(subset=["mkt_dog", "minp"])
print(f"wrote game_minpts.csv: {len(d)} games, {len(dd)} with a market dog total | "
      f"corr(mkt_dog, minp) {np.corrcoef(dd.mkt_dog, dd.minp)[0,1]:.4f} | "
      f"corr(proj_min, minp) {np.corrcoef(dd.proj_min, dd.minp)[0,1]:.4f}")
