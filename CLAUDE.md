# NCAAF Matchup Board — standing rules

Weekly college-football matchup board: Python pipeline (ESPN public data) → one self-contained HTML page
published to GitHub Pages from `docs/`. **Full spec: `BUILD_SPEC.md` — read it before any non-trivial
change.** Research history: `reference/ncaaf-matchup-runbook.md`.

## Commands
- Rebuild everything and publish locally: `bash pipeline/rebuild_all.sh` (optional `WEEK=9`, `SEASON=2027`)
- Smoke test a rendered page: `node tests/smoke.js docs/index.html` (needs `npm i playwright`)
- Single steps run from `pipeline/` (scripts import each other by module name — keep them in one folder)

## Must-follow rules
- The pipeline is a working reference implementation. Port and extend; don't rewrite the model.
- After any pipeline change, re-run the acceptance checks in `BUILD_SPEC.md` §3. Unexplained moves = stop.
- Ask John before changing any modeling choice, threshold, coefficient, colour meaning or caveat wording.
- Honesty (BUILD_SPEC §7): the model does **not** beat the closing line — never present Δ, leans or
  Shootout as an edge; keep the "model unreliable here" flag; every ROI carries the unverified
  flat-pricing assumption; floor numbers describe the posted total's level, not the matchup.
- Spreads are home-perspective: negative = home favoured. Market implied: home (T−S)/2, away (T+S)/2.
- Any projection of a past week must use that week's as-of league average (`mu_pts`), never today's.
- `pipeline/data/tail_model.json` is a committed input; everything else in `pipeline/data/` is generated.
- Build before kickoff — ESPN drops a game's odds from the live scoreboard once it starts.
- Colour: red = pushes points up, blue = pushes points down, gray = average; every coloured cell prints its number.
