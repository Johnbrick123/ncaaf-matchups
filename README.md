# NCAAF Matchup Board

A weekly college-football (FBS) board: projected points for every team from opponent-adjusted
ratings built on ESPN's public play-by-play, compared with the betting market, plus the price math
for bought-down alternate overs. Same family as the MLB matchup site.

- **Run it:** `pip install -r requirements.txt && bash pipeline/rebuild_all.sh`, then open `docs/index.html`.
- **Publish it:** push to GitHub, Settings → Pages → Deploy from branch `main` / `/docs`.
- **Automate it:** `.github/workflows/weekly.yml` rebuilds Tue/Thu/Sat at 12:00 UTC.
- **How it works and what it can't do:** `BUILD_SPEC.md` (start here) and `reference/ncaaf-matchup-runbook.md`.

The model predicts games about as well as a good public model but does not beat the closing line.
The page says so at the top; keep it that way.
