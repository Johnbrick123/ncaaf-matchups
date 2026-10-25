# NCAAF Matchup Board

A weekly college-football (FBS) board: projected points for every team from opponent-adjusted
ratings built on ESPN's public play-by-play, a Shootout ranking of every game's expected scoring,
and the posted line beside it for reference. Same family as the MLB and NHL matchup sites.

Tabs: **This week** (one card per game, one row per team: adjusted offense, the opponent's adjusted
defense, tempo, projected points, market implied, Δ, model line vs posted line, rush/pass edge) ·
**Shootout** (every game ranked by expected total) · **Rankings** (every FBS team's ratings) ·
**Backtest** (what the model does and does not do, 2015–2025).

- **Run it:** `pip install -r requirements.txt && bash pipeline/rebuild_all.sh`, then open `index.html`.
- **Publish it:** push to GitHub, Settings → Pages → Deploy from branch `main` / `/ (root)`. Live: https://johnbrick123.github.io/ncaaf-matchups/
- **Automate it:** `.github/workflows/weekly.yml` rebuilds Tue/Thu/Sat at 12:00 UTC.
- **How it works and what it can't do:** `BUILD_SPEC.md` (start here) and `reference/ncaaf-matchup-runbook.md`.

The model predicts games about as well as a good public model but does not beat the closing line.
The page says so at the top; keep it that way.
