#!/usr/bin/env python3
"""
ncaaf_bigfav.py — the alt-SPREAD buy-down question.

John buys big favourites down: instead of -45, take -27. Two things have to be true for that to
make money: (1) the clear rate at the number bought has to beat the price, and (2) if we want to
choose BETWEEN two big favourites at the same price, something has to predict which one blows the
doors off and which one scrapes by.

This builds a favourite-oriented game table (favourite's own ratings, the dog's ratings, context),
then runs the standing method: Stage A = does the effect exist against a self-control, Stage B =
does it beat the closing number, with 2025 kept untouched, plus leave-one-season-out.
Writes data/bigfav_results.json.
"""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MINSPR = 14.0


def build():
    f = pd.read_csv(os.path.join(DATA, "features.csv"))
    f = f[f.team_fbs & f.opp_fbs & f.pts.notna() & f.close_spread_team.notna()].copy()
    keep = f.groupby("game_id").size().eq(2)
    f = f[f.game_id.isin(keep[keep].index)]
    # one row per team-game; close_spread_team is this team's line (negative = this team favoured)
    f["is_fav"] = f.close_spread_team < 0
    rows = []
    for gid, x in f.groupby("game_id"):
        if x.is_fav.sum() != 1:
            continue                                    # pick'em or bad data
        fav = x[x.is_fav].iloc[0]
        dog = x[~x.is_fav].iloc[0]
        posted = abs(float(fav.close_spread_team))
        if posted < MINSPR:
            continue
        rows.append({
            "game_id": gid, "season": int(fav.season), "wk": int(fav.wk), "posted": posted,
            "fav_margin": float(fav.pts - dog.pts), "total": float(fav.pts + dog.pts),
            "close_total": float(fav.close_total) if pd.notna(fav.close_total) else np.nan,
            "fav_pts": float(fav.pts), "dog_pts": float(dog.pts),
            # favourite's own side
            "f_O": float(fav.O_pts), "f_D": float(fav.Dt_pts),
            "f_Osr": float(fav.O_sr), "f_Dsr": float(fav.Dt_sr),
            "f_Oxr": float(fav.O_xr), "f_Dxr": float(fav.Dt_xr),
            "f_plays": float(fav.O_plays), "f_to": float(fav.O_to_rate),
            "f_home": 1.0 if (fav.home == 1 and not fav.neutral) else 0.0,
            "f_gb": float(fav.games_before),
            # dog's side
            "d_O": float(dog.O_pts), "d_D": float(dog.Dt_pts),
            "d_Osr": float(dog.O_sr), "d_Dsr": float(dog.Dt_sr),
            "d_Oxr": float(dog.O_xr), "d_Dxr": float(dog.Dt_xr),
            "d_plays": float(dog.O_plays), "d_to": float(dog.O_to_rate),
            # context
            "neutral": float(bool(fav.neutral)),
        })
    g = pd.DataFrame(rows)
    # derived, all expressed so that "bigger = the favourite should win by more"
    g["rating_margin"] = (g.f_O - g.f_D) - (g.d_O - g.d_D)
    g["model_edge"] = g.rating_margin - g.posted        # how much the model likes the favourite vs the line
    g["fav_def"] = -g.f_D                               # positive = favourite's defense is good
    g["dog_off"] = g.d_O                                # positive = dog can score
    g["dog_floor"] = g.d_O - g.f_D                      # dog offense vs favourite defense
    g["pace"] = g.f_plays + g.d_plays
    g["fav_expl"] = g.f_Oxr
    g["dog_to"] = g.d_to
    g["err"] = g.fav_margin - g.posted
    return g



def analyse(g):
    """The three ways to bet 'the big favourite will score', priced against each other."""
    g = g.dropna(subset=["close_total"]).copy()
    g["imp_fav"] = (g.close_total + g.posted) / 2
    g["imp_dog"] = (g.close_total - g.posted) / 2
    tr, ho = g[g.season < 2025], g[g.season == 2025]
    KS = (9, 12, 15, 18, 21, 24)

    def rates(x):
        return {
            "spread": {k: float((x.fav_margin > x.posted - k).mean()) for k in KS},
            "fav_total": {k: float((x.fav_pts > x.imp_fav - k).mean()) for k in KS},
            "game_total": {k: float((x.total > x.close_total - k).mean()) for k in KS},
        }

    out = {"train": rates(tr), "holdout": rates(ho),
           "n_train": int(len(tr)), "n_holdout": int(len(ho)),
           "sd": {"spread": float((tr.fav_margin - tr.posted).std()),
                  "fav_total": float((tr.fav_pts - tr.imp_fav).std()),
                  "dog_total": float((tr.dog_pts - tr.imp_dog).std()),
                  "game_total": float((tr.total - tr.close_total).std())},
           "bias": {"fav_total": float((tr.fav_pts - tr.imp_fav).mean()),
                    "dog_total": float((tr.dog_pts - tr.imp_dog).mean())},
           "by_spread": {}, "by_season": {}}
    for lo, hi in [(14, 21), (21, 28), (28, 60)]:
        x = tr[(tr.posted >= lo) & (tr.posted < hi)]
        out["by_spread"][f"{lo}-{hi}"] = {
            "n": int(len(x)), "imp_fav_avg": float(x.imp_fav.mean()),
            "spread": {k: float((x.fav_margin > x.posted - k).mean()) for k in KS},
            "fav_total": {k: float((x.fav_pts > x.imp_fav - k).mean()) for k in KS}}
    for yr, x in g.groupby("season"):
        out["by_season"][int(yr)] = {
            "n": int(len(x)),
            "spread15": float((x.fav_margin > x.posted - 15).mean()),
            "spread18": float((x.fav_margin > x.posted - 18).mean()),
            "fav_total15": float((x.fav_pts > x.imp_fav - 15).mean()),
            "fav_total18": float((x.fav_pts > x.imp_fav - 18).mean())}

    # why the failures happen: favourite short, or dog long?
    f = tr[tr.fav_margin <= tr.posted - 18]
    out["failure_anatomy"] = {
        "n_fail": int(len(f)), "fail_rate": float((tr.fav_margin <= tr.posted - 18).mean()),
        "fav_short_share": float((f.fav_pts < f.imp_fav).mean()),
        "dog_long_share": float((f.dog_pts > f.imp_dog).mean()),
        "both_share": float(((f.fav_pts < f.imp_fav) & (f.dog_pts > f.imp_dog)).mean()),
        "fav_avg_short": float((f.imp_fav - f.fav_pts).mean()),
        "dog_avg_long": float((f.dog_pts - f.imp_dog).mean())}
    return out


def ratetab(x, ks=(12, 15, 18, 21)):
    return {f"k{k}": float((x.fav_margin > x.posted - k).mean()) for k in ks}


def main():
    g = build()
    g.to_csv(os.path.join(DATA, "bigfav.csv"), index=False)
    tr, ho = g[g.season < 2025], g[g.season == 2025]
    out = {"n": int(len(g)), "n_train": int(len(tr)), "n_holdout": int(len(ho)),
           "min_spread": MINSPR, "base": ratetab(tr), "base_holdout": ratetab(ho)}
    print(f"built {len(g)} favourite-games at -{MINSPR:.0f} or bigger "
          f"(train {len(tr)}, holdout {len(ho)})")
    out["markets"] = analyse(g)
    json.dump(out, open(os.path.join(DATA, "bigfav_results.json"), "w"), indent=1)
    m = out["markets"]
    print(f'  SD: spread {m["sd"]["spread"]:.2f} | favourite team total {m["sd"]["fav_total"]:.2f} '
          f'| dog team total {m["sd"]["dog_total"]:.2f}')
    for k in (15, 18):
        print(f'  buy {k} down: spread clears {m["train"]["spread"][k]*100:.1f}% '
              f'(2025 {m["holdout"]["spread"][k]*100:.1f}%) | favourite team total '
              f'{m["train"]["fav_total"][k]*100:.1f}% (2025 {m["holdout"]["fav_total"][k]*100:.1f}%)')
    return g


if __name__ == "__main__":
    main()
