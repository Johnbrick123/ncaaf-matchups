# NCAAF Matchup Board — standing rules

Weekly college-football matchup board: Python pipeline (ESPN public data) → one self-contained HTML page
published to GitHub Pages from `index.html` at the repo root (`docs/history/` keeps each build's data). **Full spec: `BUILD_SPEC.md` — read it before any non-trivial
change.** Research history: `reference/ncaaf-matchup-runbook.md`.

## Commands
- Rebuild everything and publish locally: `bash pipeline/rebuild_all.sh` (optional `WEEK=9`, `SEASON=2027`)
- Smoke test a rendered page: `node tests/smoke.js index.html` (needs `npm i playwright`)
- Single steps run from `pipeline/` (scripts import each other by module name — keep them in one folder)

## Must-follow rules
- The pipeline is a working reference implementation. Port and extend; don't rewrite the model.
- After any pipeline change, re-run the acceptance checks in `BUILD_SPEC.md` §3. Unexplained moves = stop.
- Ask John before changing any modeling choice, threshold, coefficient, colour meaning or caveat wording.
- Honesty (BUILD_SPEC §7): the model does **not** beat the closing line — never present Δ, leans or
  Shootout as an edge; keep the "model unreliable here" flag.
- Scope (John, Oct 24 2026): the board scores games from data inputs, like the NHL and MLB sites. No
  alt-line / bought-down-total / floor / price-ladder / price-check features — those scripts
  (`ncaaf_minpts.py`, `ncaaf_floor.py`, `ncaaf_bothfloor.py`, `ncaaf_bigfav.py`) are research-only
  and are not run by `rebuild_all.sh`.
- Spreads are home-perspective: negative = home favoured. Market implied: home (T−S)/2, away (T+S)/2.
- Any projection of a past week must use that week's as-of league average (`mu_pts`), never today's.
- Everything in `pipeline/data/` is generated; nothing there is committed.
- Build before kickoff — ESPN drops a game's odds from the live scoreboard once it starts.
- Colour: red = pushes points up, blue = pushes points down, gray = average; every coloured cell prints its number.
