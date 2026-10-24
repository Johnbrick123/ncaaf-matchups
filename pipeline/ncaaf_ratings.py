#!/usr/bin/env python3
"""
ncaaf_ratings.py — opponent-adjusted "as of" ratings for every team-game.

For each stat X (points, points/drive, yards/play, success rate, explosive rate, plays, turnovers/drive,
seconds/play) fit, using ONLY games played before the week in question:

    X_game = mu + O_team + D_opp + h*home            (ridge / least squares)

  O_team : how much more of X this offense produces than average (opponent-adjusted)
  D_opp  : how much more of X this DEFENSE allows than average  (positive = bad defense)

Prior: each O/D is shrunk toward c * (the team's final rating last season), with weight lam expressed in
"pseudo-games". Games are recency-weighted by rho**(weeks ago). All FCS opponents are pooled into one
pseudo-team "FCS". Output: data/features.csv — one row per team-game with the as-of ratings of the team's
offense and the opponent's defense, plus the raw pre-game season-to-date averages for comparison.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

STATS = ["pts", "ppd", "ypp", "sr", "xr", "plays", "to_rate", "spp", "ypd", "fp",
         "rush_ypc", "pass_ypa", "rush_sr", "pass_sr", "rush_xr", "pass_xr", "pass_rate"]


def load_teamgames():
    t = pd.read_csv(os.path.join(DATA, "teamgames.csv"))
    t = t[t.plays >= 30].copy()  # a handful of broken/partial logs
    t["off_pts7"] = 7 * t.td_drives + 3 * t.fg_drives
    t["ppd"] = t.off_pts7 / t.drives
    t["ypp"] = t.yards / t.plays
    t["sr"] = t.succ / t.succ_n
    t["xr"] = t.expl15 / t.plays
    t["to_rate"] = t.turnovers / t.drives
    t["spp"] = t.sec_per_play
    t["ypd"] = t.yards / t.drives
    t["fp"] = t.start_ytg          # average yards-to-go at drive start (lower = better field position)
    t["pts"] = t.pts_reg.where(t.pts_reg.notna(), t.pts)
    # rush / pass splits (per attempt; pass attempts include sacks and interceptions)
    ra = t.rush_att.clip(lower=1); pa = t.pass_att.clip(lower=1)
    t["rush_ypc"] = t.rush_yds / ra
    t["pass_ypa"] = t.pass_yds / pa
    t["rush_sr"] = t.rush_succ / t.rush_succ_n.clip(lower=1)
    t["pass_sr"] = t.pass_succ / t.pass_succ_n.clip(lower=1)
    t["rush_xr"] = t.rush_x15 / ra
    t["pass_xr"] = t.pass_x15 / pa
    t["pass_rate"] = t.pass_att / (t.rush_att + t.pass_att).clip(lower=1)
    t["home_adj"] = t.home.astype(float)  # 1 home, 0 away or neutral
    # pool FCS into one pseudo team
    t["team_key"] = np.where(t.team_fbs, t.team_id.astype(str), "FCS")
    t["opp_key"] = np.where(t.opp_fbs, t.opp_id.astype(str), "FCS")
    t["date"] = pd.to_datetime(t.date, utc=True)
    # ordinal week within season: regular weeks 1..16, bowls = 20
    t["wk"] = np.where(t.stype == 3, 20, t.week)
    return t


class RatingEngine:
    def __init__(self, lam=4.0, c=0.5, rho=1.0, stats=STATS, lam_h=50.0, fcs_w=1.0):
        self.lam, self.c, self.rho, self.stats, self.lam_h = lam, c, rho, stats, lam_h
        self.fcs_w = fcs_w  # weight on games against the pooled FCS opponent (opponent quality is unmeasured)
        self.prior_h = None  # previous season's fitted home-field effect per stat

    def fit(self, rows, prior_O, prior_D, as_of_week):
        """rows: DataFrame of games before as_of_week (one row per offense). Returns dict of ratings.
        Home field gets its own prior (self.prior_h, weight lam_h): early in a season 'home' is confounded
        with 'the better program hosting a guarantee game', and an unpenalized h soaks that up."""
        teams = sorted(set(rows.team_key) | set(rows.opp_key) | set(prior_O.keys()))
        idx = {k: i for i, k in enumerate(teams)}
        n_t, n_r = len(teams), len(rows)
        S = len(self.stats)
        w = self.rho ** (as_of_week - rows.wk.values).clip(0)
        if self.fcs_w != 1.0:
            fcs_game = (rows.opp_key.values == "FCS") | (rows.team_key.values == "FCS")
            w = w * np.where(fcs_game, self.fcs_w, 1.0)
        w = np.sqrt(w)
        extra = 1 if self.prior_h is not None else 0
        X = np.zeros((n_r + 2 * n_t + extra, 2 + 2 * n_t))
        Y = np.zeros((n_r + 2 * n_t + extra, S))
        ti = rows.team_key.map(idx).values
        oi = rows.opp_key.map(idx).values
        r = np.arange(n_r)
        X[r, 0] = w
        X[r, 1] = w * rows.home_adj.values
        X[r, 2 + ti] = w
        X[r, 2 + n_t + oi] = w
        Y[:n_r] = rows[self.stats].values * w[:, None]
        sl = np.sqrt(self.lam)
        for k, i in idx.items():
            X[n_r + i, 2 + i] = sl
            X[n_r + n_t + i, 2 + n_t + i] = sl
            po = prior_O.get(k)
            pdf = prior_D.get(k)
            Y[n_r + i] = sl * self.c * (np.array(po) if po is not None else 0.0)
            Y[n_r + n_t + i] = sl * self.c * (np.array(pdf) if pdf is not None else 0.0)
        if extra:
            X[n_r + 2 * n_t, 1] = np.sqrt(self.lam_h)
            Y[n_r + 2 * n_t] = np.sqrt(self.lam_h) * np.array(self.prior_h)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        mu, h = beta[0], beta[1]
        O = {k: beta[2 + i] for k, i in idx.items()}
        D = {k: beta[2 + n_t + i] for k, i in idx.items()}
        return {"mu": mu, "h": h, "O": O, "D": D}

    def run(self, t, seasons=None, verbose=True):
        """Produce as-of features for every team-game. Ratings for week w use games with wk < w."""
        seasons = seasons or sorted(t.season.unique())
        prior_O, prior_D = {}, {}
        out = []
        S = len(self.stats)
        zero = np.zeros(S)
        for season in seasons:
            ts = t[t.season == season]
            weeks = sorted(ts.wk.unique())
            final = None
            for wk in weeks:
                hist = ts[ts.wk < wk]
                fit = self.fit(hist, prior_O, prior_D, wk) if len(hist) else None
                cur = ts[ts.wk == wk]
                for _, g in cur.iterrows():
                    if fit is None:
                        O_t = self.c * np.array(prior_O.get(g.team_key, zero))
                        D_o = self.c * np.array(prior_D.get(g.opp_key, zero))
                        D_t = self.c * np.array(prior_D.get(g.team_key, zero))
                        O_o = self.c * np.array(prior_O.get(g.opp_key, zero))
                        mu = np.array(self._season_mu(t, season))
                        h = np.zeros(S)
                    else:
                        O_t = fit["O"].get(g.team_key, self.c * np.array(prior_O.get(g.team_key, zero)))
                        D_o = fit["D"].get(g.opp_key, self.c * np.array(prior_D.get(g.opp_key, zero)))
                        D_t = fit["D"].get(g.team_key, self.c * np.array(prior_D.get(g.team_key, zero)))
                        O_o = fit["O"].get(g.opp_key, self.c * np.array(prior_O.get(g.opp_key, zero)))
                        mu, h = fit["mu"], fit["h"]
                    n_prev = int((hist.team_key == g.team_key).sum())
                    rec = {"game_id": g.game_id, "team_id": g.team_id, "season": season, "wk": wk,
                           "games_before": n_prev}
                    for j, s in enumerate(self.stats):
                        rec[f"O_{s}"] = O_t[j]; rec[f"D_{s}"] = D_o[j]
                        rec[f"Dt_{s}"] = D_t[j]; rec[f"Oo_{s}"] = O_o[j]
                        rec[f"mu_{s}"] = mu[j]; rec[f"h_{s}"] = h[j]
                    out.append(rec)
                if verbose and wk in (1, 5, 10):
                    print(f"  {season} wk{wk:>2}: fit on {len(hist)} rows", flush=True)
            # final ratings for next season's prior — full-season fit with rho=1 (no recency)
            eng = RatingEngine(lam=self.lam, c=self.c, rho=1.0, stats=self.stats, lam_h=self.lam_h, fcs_w=self.fcs_w)
            eng.prior_h = self.prior_h
            fin = eng.fit(ts, prior_O, prior_D, 99)
            prior_O, prior_D = fin["O"], fin["D"]
            self.prior_h = fin["h"]
        return pd.DataFrame(out)

    def _season_mu(self, t, season):
        # league mean of the previous season (used only when no games have been played yet)
        prev = t[t.season == season - 1]
        if not len(prev):
            prev = t
        return prev[self.stats].mean().values


def add_raw_baselines(t):
    """Season-to-date raw averages (no opponent adjustment), computed before each game."""
    t = t.sort_values(["season", "team_key", "date"]).copy()
    grp = t.groupby(["season", "team_key"])
    for s in ("pts", "ppd", "ypp", "plays"):
        t[f"raw_{s}"] = grp[s].transform(lambda x: x.shift(1).expanding().mean())
        t[f"raw4_{s}"] = grp[s].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    # opponent's raw allowed: need the opponent's season-to-date allowed averages
    allowed = t[["season", "game_id", "team_key", "opp_pts", "date"]].copy()
    allowed = allowed.sort_values(["season", "team_key", "date"])
    allowed["raw_pa"] = allowed.groupby(["season", "team_key"])["opp_pts"].transform(lambda x: x.shift(1).expanding().mean())
    allowed["raw4_pa"] = allowed.groupby(["season", "team_key"])["opp_pts"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    m = allowed[["game_id", "team_key", "raw_pa", "raw4_pa"]].rename(columns={"team_key": "opp_key", "raw_pa": "opp_raw_pa", "raw4_pa": "opp_raw4_pa"})
    t = t.merge(m, on=["game_id", "opp_key"], how="left")
    return t


if __name__ == "__main__":
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    c = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    rho = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    tag = sys.argv[4] if len(sys.argv) > 4 else "features"
    t = load_teamgames()
    t = add_raw_baselines(t)
    eng = RatingEngine(lam=lam, c=c, rho=rho)
    f = eng.run(t)
    full = t.merge(f, on=["game_id", "team_id", "season", "wk"], how="left")
    full.to_csv(os.path.join(DATA, f"{tag}.csv"), index=False)
    print("wrote", tag, full.shape)
