# NCAAF Matchup Board — build brief for Claude Code

> **Scope change, Oct 24 2026 (John):** the board now *scores the games from data inputs*, the way the
> NHL and MLB sites do. The alt-line machinery described below — the Floor chip, both-teams floor,
> bracket over rate, over read, price ladder, price-check calculator, the Shootout tab's bucket/ROI
> columns and calibration table, Backtest §8–9 and `data/tail_model.json` — has been **removed from the
> page and the build**. The research scripts that produced it (`ncaaf_minpts.py`, `ncaaf_floor.py`,
> `ncaaf_bothfloor.py`, `ncaaf_bigfav.py`) stay in `pipeline/` for reference but `rebuild_all.sh` does
> not run them. Sections marked ~~like this~~ describe the removed pieces and are kept only as history.

**Read this whole document before writing any code.** It describes a system that already exists and
works. The files in this repo are the reference implementation: every script has been run end to end
and reproduces the published backtest to the decimal. Your job is to stand it up as a website, put it
on a weekly schedule, and keep it honest. **Do not redesign the model or re-derive it from this text —
port and wire up the existing code, then extend it one step at a time with the owner's approval.**

Owner: John (Chicago, CFA). Sister site for reference: `johnbrick123.github.io/mlb-matchups` — same
family, same look (row per team, compact, colour-coded, every coloured cell prints its number).

---

## 0. How to work on this (rules for Claude Code)

1. **Reuse, don't rewrite.** `pipeline/` holds the working pipeline. Keep the script names, the
   flat layout (scripts import each other by module name) and the `data/` folder next to them.
2. **Verify before publishing.** Run the acceptance checks in §3 after any change to the pipeline.
   If a number moves and you did not intend it to, stop and find out why.
3. **The honesty rules in §7 are product requirements**, not decoration. Never remove a caveat,
   restyle a model-vs-market delta as an "edge", or present an ROI without its pricing assumption.
4. **Ask John before changing** any modeling choice, threshold, coefficient, colour meaning, or the
   wording of a caveat. Layout, performance, accessibility and mobile fixes are fine to just do.
5. Plain English in everything user-facing. John prefers single-page, scannable output.

## 1. What this is and who it's for

A weekly college-football (FBS) board. For every FBS-vs-FBS game it projects each team's points from
public play-by-play data and shows the posted line beside it for reference.

**What John wants from it (Oct 24 2026):** score every game from the data inputs, the way the NHL
and MLB boards do — projected points per team, the Shootout ranking, the ratings — and make the
scoring picture easy to scan each week. Not alt lines, not bought-down totals.

**What the research found (keep this framing everywhere):**

- The model **predicts games** about as well as a good public model: team-points miss 8.89 in the
  untouched 2025 season vs 8.59 for the closing line; picks winners ~72%.
- It **does not beat the closing line** on totals or spreads (2025: 50.6% / 48.7%; 2026 live forward
  test through week 8: totals 54.0% ± 2.5, spreads 46.3%). The market is slightly *more* accurate.
- The one structural thing with money in it is **the level of the posted total**: low posted totals
  clear a bought-down alt-over far more often than high ones (16 points down: ~88% on totals under 45
  vs ~79% on 65+, 2015–25; 2026 so far 91.1% under 50 vs 78.8% at 55+). That only pays **if a book
  charges the same price per point bought regardless of the total's level — still unverified.**
- Nothing about the *matchup* (offense/defense quality, tempo, run/pass styles, volatility…) moves
  the chance of clearing a bought-down number once the posted total is known. Tested many ways; null.

## 2. Repo layout (the starter zip)

```
ncaaf-board/
├── BUILD_SPEC.md              ← this file
├── CLAUDE.md                  ← short standing rules (Claude Code loads it automatically)
├── README.md
├── requirements.txt           ← requests, pandas, numpy, scipy, scikit-learn
├── .gitignore                 ← ignores everything generated under pipeline/data
├── .github/workflows/weekly.yml  ← Tue/Thu/Sat rebuild + publish to GitHub Pages (/docs)
├── pipeline/                  ← the working pipeline (run from anywhere: rebuild_all.sh cds into it)
│   ├── rebuild_all.sh         ← runs everything below in order and publishes docs/index.html
│   ├── ncaaf_pull.py          ← ESPN schedules + per-game summaries + odds archive → data/games/*.json.gz
│   ├── ncaaf_features.py      ← → data/games.csv, data/teamgames.csv
│   ├── ncaaf_ratings.py       ← opponent-adjusted as-of ratings → data/features.csv
│   ├── ncaaf_backtest.py      ← Stage A/B backtest → data/backtest_results.json, data/game_projections.csv
│   ├── ncaaf_overs.py         ← overs study, game signals → data/overs_results.json, data/game_signals.csv
│   ├── ncaaf_minpts.py        ← research only (not run): → data/game_minpts.csv
│   ├── ncaaf_floor.py         ← research only (not run): alt-over floor by posted-total bracket
│   ├── ncaaf_bothfloor.py     ← research only (not run): both-teams floor
│   ├── ncaaf_shootcal.py      ← Shootout percentile vs over results → data/shoot_calibration.json (Shootout tab caveat)
│   ├── ncaaf_bigfav.py        ← research only (not run): big-favourite buy-down
│   ├── ncaaf_forward.py       ← 2026 live forward test → data/forward_test.json
│   ├── build_ncaaf_board.py   ← this week's board data → ncaaf_board_data.json
│   ├── render_ncaaf_board.py  ← bakes the JSON into the template → NCAAF_Matchup_Scores_<season>_wk<week>.html
│   ├── ncaaf_board_template.html  ← the whole front end (HTML/CSS/JS, one file, data injected at /*DATA*/)
├── index.html                 ← the published board (GitHub Pages serves main / root, like the MLB site)
├── .nojekyll                  ← tells Pages to serve files as-is
├── docs/history/              ← board_<date>.json archived from every build
├── reference/
│   ├── ncaaf-matchup-runbook.md       ← the full research log (why every choice was made)
│   └── example_board_2026_wk8.html    ← what the finished page looks like — open it in a browser
└── tests/smoke.js             ← Playwright smoke test of a rendered page
```

## 3. Quick start and acceptance checks

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11+ (tested on 3.13, pandas 3)
pip install -r requirements.txt
bash pipeline/rebuild_all.sh          # first run ≈ 6 min (ESPN pull of ~11,000 games), later runs ≈ 1–2 min
open index.html                       # macOS; or just double-click it
# optional browser smoke test:
npm i playwright && npx playwright install chromium && node tests/smoke.js index.html
```

`WEEK=9 bash pipeline/rebuild_all.sh` forces a week; `SEASON=2027` forces a season. With no `WEEK`,
the build picks the first regular-season week that still has unplayed games.

**Acceptance checks — these must reproduce exactly** (they did on Oct 24 2026; numbers marked † grow
as 2026 games are played):

| Step | Must print |
|---|---|
| `ncaaf_features.py` | `games: 11500 … team-games: ~21,4xx †` |
| `ncaaf_backtest.py` (backtest_report.txt) | `evaluation rows: 15292 team-games, seasons 2015-2025`; `SCORE + blowout terms` 9.41 / corr 0.535 (2015–24 LOSO) and 8.89 / 0.529 (2025); `MARKET implied team total` 9.08 / 0.576 and 8.59 / 0.572 |
| `ncaaf_overs.py` | `games: 7646 \| base over rate train 0.492 holdout 0.489` |
| `ncaaf_shootcal.py` | `n=6910`; bucket 0–20 over 51.8%; bucket 90–101 over 44.3% |
| `build_ncaaf_board.py` (2026 season) | `coef: const 0.547, O_pts 0.632, D_pts 0.745, O_sr 32.645, D_sr 20.877, O_xr 22.402, D_xr 18.849, home_adj 2.673, blow_fav 0.172, blow_dog 0.133` |
| `tests/smoke.js` | 56 cards (wk 8), 44 Shootout rows, 56 line-move rows, 138 ranking rows, 7 backtest sections, `no errors` |

(The research-only scripts, when run by hand: `ncaaf_minpts.py` → `corr(mkt_dog, minp) 0.4084`; `ncaaf_floor.py` → `games 7625 | SD 16.29 skew 0.35`; `ncaaf_bothfloor.py` → `n=7623 | base P(both>20)=0.443`; `ncaaf_bigfav.py` → `SD: spread 15.84 | favourite team total 12.07 | dog team total 10.01`.)

(Coefficients refit automatically when a new season starts — they will differ in 2027.)

## 4. Data

### 4.1 ESPN endpoints (all public, no key)

| What | URL |
|---|---|
| Season calendar | `https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates=YYYY&groups=80&limit=1` |
| Week scoreboard (FBS) | `…/scoreboard?dates=YYYY&seasontype=2&week=W&groups=80&limit=400` (`seasontype=3` = bowls) |
| Current week + live odds | `…/scoreboard?groups=80&limit=400` → `competitions[0].odds[0]` (`details` like `"MRSH -4.5"`, `overUnder`) |
| Game summary | `…/summary?event=ID` → box score, drives with full play-by-play, gameInfo (venue, weather) |
| Odds archive | `https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/events/ID/competitions/ID/odds` → 4–14 books, with `open` / `close` / `current` blocks |

- `groups=80` = FBS. FBS conference IDs: `{1 ACC, 4 Big 12, 5 Big Ten, 8 SEC, 9 Pac-12, 12 C-USA,
  15 MAC, 17 Mountain West, 18 FBS Independents, 37 Sun Belt, 151 American}`. Every other opponent is
  pooled into one pseudo-team `"FCS"`. Re-check this set each preseason for realignment (North Dakota
  State joined the Mountain West in 2026; before that it was FCS, so it has no prior-season FBS rating).
- The pull runs 10 threads, retries with backoff, and caches each completed game as
  `data/games/<id>.json.gz` (≈46 MB for 2014–2026). Completed games never change, so the cache is safe
  to keep forever; schedules are re-pulled every run.
- **User-agent gotcha:** in our sandbox a browser-style User-Agent got HTTP 403 through a proxy, so the
  pull deliberately uses the default `python-requests` UA. On GitHub Actions there's no proxy; if ESPN
  ever starts returning 403s, try a browser UA instead.
- ESPN removes a game's odds from the scoreboard **once it kicks off** (and during weather delays).
  Build before kickoff — the schedule is Tue/Thu/Sat 12:00 UTC for that reason.

### 4.2 From raw games to stats (`ncaaf_features.py`)

- Scrimmage plays are a fixed set of ESPN play types (rush, pass reception/incompletion, sack, rushing
  /passing TD, interception variants, fumble recoveries, safety). Turnover plays count 0 yards.
- **Success rate:** 1st down gains ≥50% of the distance, 2nd ≥70%, 3rd/4th ≥100%; TDs always succeed.
- **Explosive rate:** scrimmage plays of 15+ yards ÷ plays (20+ also stored).
- **Offensive points = 7 × TD drives + 3 × FG drives** (from 2021 ESPN logs the PAT outside the drive,
  so the running score is wrong for this). Ratings use **regulation** points (sum of quarters 1–4);
  the backtest target is full-game points.
- OT drives are excluded from ratings. "Dead" drives (end of half/game with ≤2 plays) are dropped.
- Rush/pass split: sacks and interceptions count as pass attempts; fumbles as rushes.
- Field position = yards to go at the first scrimmage play of each drive; tempo = drive seconds ÷ plays.
- **Closing lines** — the odds archive is messy; these rules are load-bearing:
  - Provider priority for the close: ESPN BET → consensus → DraftKings → Caesars → … (list in code).
    Totals and spreads are chosen independently (consensus carries only spreads in 2015–16).
  - **2022–23 ESPN BET stores the odds (−110) in the open/close *line* fields.** Every open/close is
    validated: totals must be 20–110 and within 12 of the headline `overUnder`; spreads |x| ≤ 70 and
    within 10 of the headline.
  - `spread` is home-perspective (negative = home favoured); cross-check against the `details`
    string and trust `details` when only the sign disagrees.
  - Openers: ESPN BET `open` (2023+), legacy `"Opening"` provider (2015–16).
- Coverage: 11,500 games 2014–2026; ~9,450 completed FBS-vs-FBS with closing total and spread.

### 4.3 Ratings (`ncaaf_ratings.py`)

For each of 17 stats (points, pts/drive, yds/play, success rate, explosive rate, plays, turnovers/drive,
sec/play, yds/drive, field position, rush yds/carry, pass yds/att, rush & pass success, rush & pass
explosive, pass rate), fit by ridge least squares:

```
stat_game = league + O(offense team) + D(defense opponent) + h · home
```

- **As-of:** ratings used for week *w* come only from games with week < *w* (no look-ahead). Bowls = week 20.
- **Prior:** each team's O and D are shrunk toward **0.75 × last season's final rating**, with the
  weight of **2 games** (`lam=2, c=0.75`). Final ratings for the next season's prior are a full-season
  fit with no recency weighting.
- **Recency:** games weighted 0.9^(weeks ago) (`rho=0.9`). Heavier recency made things worse.
- **Home field** gets its own prior pinned to last season's value with the weight of **50 games** —
  without it, week 1 ("big program hosts a cupcake") inflates home field to ~11 points.
- Team-games with fewer than 30 plays are dropped (broken logs).
- Hyperparameters were chosen by grid search on 2015–24 only; the top five combos were within 0.05 points.
- Down-weighting FCS games was tested and rejected (made the 2025 early-season holdout worse).

### 4.4 The projection — "the SCORE" (`ncaaf_backtest.py`, used by `build_ncaaf_board.py`)

Ordinary least squares of **team points minus the as-of league average** on:

| Input | Meaning |
|---|---|
| `O_pts`, `D_pts` | team's offensive points rating; opponent's defensive points-allowed rating (positive = bad defense) |
| `O_sr`, `D_sr` | success rate, offense / opponent defense allowed |
| `O_xr`, `D_xr` | explosive rate, offense / opponent defense allowed |
| `home_adj` | 1 for a true home team, 0 away or neutral |
| `blow_fav` = max(rmargin − 21, 0), `blow_dog` = min(rmargin + 21, 0) | hinge terms fixing big-mismatch under-projection; rmargin = (team O_pts − team D_pts) − (opp O_pts − opp D_pts) |

`projected points = league_avg_as_of + const + Σ coef × input`. Fitted on every completed FBS-vs-FBS game
from seasons **before** the current one. 2026 coefficients: const 0.547, O_pts 0.632, D_pts 0.745,
O_sr 32.645, D_sr 20.877, O_xr 22.402, D_xr 18.849, home 2.673, blow_fav 0.172, blow_dog 0.133.
Projected total = sum of the two teams; projected margin = home − away.

## 5. Every number on the board, and how it's computed

All of this already exists in `build_ncaaf_board.py` and the template — this is the spec for reading
and maintaining it.

| Metric | Definition |
|---|---|
| **Market implied team totals** | home = (T − S)/2, away = (T + S)/2, T = posted total, S = home spread (negative = home favoured) |
| **Δ** | projected points − market implied team total (shown, colour-coded, **never labelled an edge**) |
| **Over/under lean, side lean** | model total − posted (flat if \|x\| < 2); model margin − market margin (flat if < 2) |
| **Model unreliable here** (warn pill) | \|model margin − market margin\| ≥ 10 **or** market favourite ≥ 28. Suppresses the lean pills and prints the calibration: in 405 games with a 28+ favourite the market averaged 33.9 vs an actual 33.7 while the model said 29.2 and went 49.5% ATS. Calibration buckets come from `game_projections.csv`. |
| **Shootout** | percentile of the projected total among the projected totals of every FBS-vs-FBS game in **prior** seasons (pool median 55.0, 90th pct 66.3). Pure scoring read, no market input. |
| **Rush edge / pass edge** | (home O_rush + away D_rush) + (away O_rush + home D_rush), yds/carry vs average; pass the same in yds/att; each with its percentile vs prior seasons |
| ~~REMOVED Oct 24 2026~~ **Over read** | 7.223 − 0.121 × posted + 0.648 × rush edge − 0.043 × (model total − posted). Flat if \|read\| < 1. Break-even historically — a filter, not a signal. |
| ~~REMOVED Oct 24 2026~~ **Floor (alt-over by bracket)** | for the posted-total bracket (<45, 45–49.5, 50–54.5, 55–59.5, 60–64.5, 65+): share of historical games whose total beat posted − k, k = 8/11/14/16/20, with fair prices |
| ~~REMOVED Oct 24 2026~~ **Price ladder** | for −200/−300/−400/−500/−600/−800: the fewest points to buy (k = 1…44) whose bracket clear rate ≥ break-even; tags posted ≤ 49.5 "flat-ladder value zone", ≥ 60 "overpriced zone" |
| ~~REMOVED Oct 24 2026~~ **Price check calculator** | P(total > N \| posted T) = 1 − F(z), z = ((N − T) − μ(T)) / σ(T), μ(T) = 4.838 − 0.0786·T, σ(T) = 8.978 + 0.1322·T, F = empirical CDF from the 999 quantiles in `data/tail_model.json` (right-skewed residual, so deep buys are a bit safer than a normal curve says). Break-even for American A < 0: −A/(−A + 100). ROI on risk = (p·100 − (1−p)·\|A\|)/\|A\|. Fair price = −100p/(1−p). Warn past 26 points bought (only 1.3% of games missed by that much). Defaults: over posted − 16 at −600. |
| ~~REMOVED Oct 24 2026~~ **Both-teams floor** | dog = (T − \|S\|)/2. For X = 10…27, P(both teams score > X) = logistic(a_X + b_X · dog) from `bothfloor_results.json`. Shows P at 13/17/20/24; "70% floor" = highest X with P ≥ 0.70 ("<10" if none); also 60%. League base rate: both top 20 in 44.3% of games. It's a read on the market's number (beats our own model as a predictor), not an edge. |
| ~~REMOVED Oct 24 2026~~ **Shootout bucket record** (Shootout tab) | from `shoot_calibration.json`: per bucket 0–20/20–40/40–60/60–80/80–90/90–100, over rate, clear −12/−16/−20, ROI of −16 at −600 |

**Big-favourite research** (`ncaaf_bigfav.py`, research only, not on the page): when betting a big
favourite to "just score", the favourite's **own team total** bought down beats buying down the spread,
because the spread carries both teams' randomness: miss SD 15.84 vs 12.07 for the favourite's implied
team total (12.07² + 10.01² ≈ 15.8²). Favourites −14 or bigger, 15 points bought: team total cleared
88.8% (fair −795) vs spread 82.3% (fair −464); it cleared the −600 break-even in 13 of 13 seasons.
Nothing about the matchup predicted *which* big favourite blows the doors off (complete null).

## 6. The website

One self-contained HTML file (`ncaaf_board_template.html`, ~78 KB + baked JSON ≈ 0.5 MB). No build
step, no framework, no server. Data is injected at the literal `/*DATA*/{}` by `render_ncaaf_board.py`.

**Header:** sticky; title "NCAAF Matchup Scores"; subline `season · Week N · N games · built <time> CT`;
five tabs.

**Tab 1 — This week.** Caveat block (see §7), controls, weight sliders, legend, then one card per game.
- Sort buttons: Spread gap · Combined total · Edge vs posted total · Edge vs posted spread · Shootout ·
  Kickoff. Min-edge filter. "Raw inputs" toggle (raw season PPG/allowed).
- Weight sliders 0–2× (default 1.0× = the backtested fit) for points rating, success, explosive, home.
- Card header: matchup, Shootout chip (percentile + expected total), kickoff (Central
  time), TV, venue, posted line and provider.
- Card table, one row per team (away first): Team (rank, full name, record, home/away/neutral), GP,
  Offense — points rating / success / explosive; vs opponent's defense — allows / success / explosive;
  tempo; projected points; market implied; Δ.
- Read lines: model line & total vs market line & total with pills; reliability note; rush/pass edges
  with percentiles. Completed games show a `final` pill.

**Tab 2 — Shootout.** One row per game still to play (games already played this week are hidden with a
caption). Columns: #, matchup, kickoff, expected total, posted, Shootout (number + bar), model line.
Sorts: Shootout · Kickoff. Toggle "only games with a posted total". The caveat above the table keeps
the over rate of the top vs bottom Shootout bucket (from `shoot_calibration.json`) so the ranking is
never read as an over signal.

**Tab 3 — Line moves** (added Oct 25 2026 at John's request; baseline corrected the same day). One row per
game with a line. **Week open** = the line at the *first build of the week* (the Sunday 10 pm CT run, added
for this purpose: openers are up by then); **Now** = the line at this build; **Move this week** (red up /
blue down); model total and model vs week open (coloured at 3+) with a `toward model` / `away from model`
tag; the same for the spread (move shown as "→ TEAM x.x"); and a **trail** of every build's number.
ESPN's archive "open" is shown only as a muted *ESPN first #* column and nothing is measured against it:
it has no date and for marquee games it is a look-ahead line posted a week or more early (IU at MICH wk 8:
52.5 vs a 43.5 close). Sorts: Biggest total move this week · Biggest spread move · Model vs week open ·
Kickoff. Toggle: hide games already played.
Data: `build_ncaaf_board.py` calls the ESPN odds archive for every slate game (`line_archive()`: open /
current per book, validated like close lines, provider = the one the scoreboard shows, else the close
priority list) and reads earlier builds' lines from `docs/history/board_*.json` (`history_trail()`); the
first trail point is the week open. The same archive call **keeps a game's line after kickoff** (Phase 4
item 5, done). The caveat prints the line-move agreement by season and the model-vs-opener record (pooled
and 2025) and notes that those backtest openers are ESPN's, with the same look-ahead caveat. Never style a
move as a pick.

**Tab 4 — Rankings.** Every FBS team's current ratings, sortable by any column (strength, offense pts,
success, explosive, tempo, defense allows, success/explosive allowed, run/pass offense and defense,
pass rate, raw PPG/PA).

**Tab 5 — Backtest.** Sections 1–7 in plain English with tables and small inline-SVG charts: does it
predict the game; what predicts points; recent form / tempo / turnover luck; does it beat the closer;
opener & line movement; method; overs.

**Visual system** (keep it — it matches the MLB site):
- Colour means one thing everywhere: **red = pushes points up** (strong offense, leaky defense, over
  lean), **blue = pushes points down**, gray = league average. Light red `#c4432e` / blue `#2d6db5`;
  dark mode `#e0654f` / `#5b93d8`. Cell tint alpha ∝ |value ÷ scale| (cap 0.42 light, 0.55 dark).
  Scales: points 12, success 8 pp, explosive 4 pp, run 1.2, pass 1.6, Δ 6.
- Every coloured cell prints its number. Tabular figures.
- Fonts (Google Fonts): Barlow Condensed (team codes, headers), IBM Plex Sans (body), IBM Plex Mono (numbers).
- Full light/dark support via CSS custom properties on `:root` with `prefers-color-scheme`.
- Tables scroll horizontally inside their card on narrow screens; the page itself must not.

**Hosting:** GitHub Pages from `main` → `/ (root)`, live at https://johnbrick123.github.io/ncaaf-matchups/. Note a GitHub Pages site is **public** (even from a
private repo on paid plans) — John should decide if that's OK before turning it on.

## 7. Honesty rules — do not drop these

1. The caveat at the top of *This week* must keep saying, with live numbers: the score predicts the game;
   it does **not** beat the closing line; the closing line moves toward the model ~57–68% of the time
   (the one live thread); the record against openers is positive but small-sample; the model breaks in
   lopsided games; high-scoring matchups are the *worst* place for overs.
2. Never restyle Δ, the lean pills, or Shootout as an "edge" or a "pick".
3. Keep the **model unreliable here** behaviour (spread disagreement ≥ 10 or favourite ≥ 28) — without
   it, early-season mismatches are the biggest-looking "edges" on the page and they're all artifacts.
4. ~~ROI / price figures~~ — removed from the page Oct 24 2026. If any price math ever comes back, it
   must carry the unverified flat-pricing assumption and the "level of the total, not the matchup" note.
5. Forward-test numbers are shown with their standard error and never as "the model is hot".

## 8. Research findings to show (and not contradict)

| | 2015–24 (LOSO) | 2025 holdout | 2026 live (through wk 8) |
|---|---|---|---|
| Team-points miss: model / closing line | 9.41 / 9.08 | 8.89 / 8.59 | — |
| Totals vs closing line | 50.8% | 50.6% | 54.0% ± 2.5 |
| Spreads vs closing line | 49.8% | 48.7% | 46.3% |
| Closing total moved toward the model | 57–67% (each season with an opener on file) | 67.6% | — |

- **Overs:** posted totals of 60+ went over 46.8% in 2015–24 and under 50% in all eleven seasons.
  The Shootout percentile is monotone *against* the over: bucket 0–20 over 51.8%, clears −16 88.0%
  (ROI +2.6% at −600); bucket 90–100 over 44.3%, clears −16 78.8% (−8.0%).
- **Alt-over floor:** total − close has SD 16.3, skew +0.35. Clears −16: 88.2% on <45, 84.9% on
  50–54.5, 79.4% on 65+. 2026: under 50 → 91.1%, 55+ → 78.8%.
- **Run/pass matchups:** null in every form. "Bad run defense" is the more persistent weakness marker.
- **Tempo, turnover luck, recent form, weather/dome, conference games:** priced or null.
- **When the model is 7+ points off the posted total** it was right 46.8% (2015–24) and 40.0% (2026,
  n=35). Big total disagreements are the model failing (e.g. 2026 wk 6 UGA@ALA model +14 over, final 37).

The full log, with every test and number, is `reference/ncaaf-matchup-runbook.md`.

## 9. Automation & operations

- `.github/workflows/weekly.yml` runs `pipeline/rebuild_all.sh` Sun 10 pm CT (Line-moves baseline), Tue/Thu 12:00 UTC and Sat 08/10/12 UTC (before
  kickoffs) plus on demand (Actions → ncaaf-board → Run workflow, optional week). It restores the ESPN
  game cache with `actions/cache`, rebuilds, copies the page to `index.html`, archives the board
  JSON to `docs/history/board_<date>.json`, and commits. First run ≈ 6 min; later runs ≈ 2 min.
- Turn on Pages: repo Settings → Pages → Deploy from a branch → `main` / `/ (root)` (done Oct 25 2026).
- **Season rollover (each August):** nothing to edit for the live board — the season is detected from
  the date, the pull range extends, and coefficients refit on all prior seasons. Do check: FBS
  conference IDs (realignment), and whether to roll the research holdout (`TRAIN`/`HOLD` constants in
  `ncaaf_backtest.py`, `TR`/`HO` in `ncaaf_bothfloor.py`, `season < 2025` in `ncaaf_bigfav.py`) and the
  years in the caveat text. Ask John before re-running the research with a new holdout.
- Bowls/postseason are not supported (regular season only).

## 10. Pitfalls we already hit (don't repeat them)

1. **Forward-test bug:** projecting past weeks with the *current* week's league average instead of each
   week's own as-of average biased early weeks by ~4.5 points on totals. `ncaaf_forward.py` does it right
   (uses each row's `mu_pts`). Any new backtest must do the same.
2. Odds stored in line fields (2022–23) — validate every line (see §4.2).
3. Spread sign conventions (home-perspective, negative = home favoured) — check every new consumer.
4. PAT logged outside the drive from 2021 → use 7×TD + 3×FG for drive points.
5. Home field inflated early season → keep the home-field prior.
6. Linear model under-projects big favourites by ~5 → keep the blowout hinges.
7. `game_minpts.csv` must exist before `ncaaf_bothfloor.py` (research only; run `ncaaf_minpts.py` first).
8. `build_ncaaf_board.py` filters `--flags` out of positional args — keep that if you touch the CLI.
9. ~~Floor "<10" display~~ (removed with the floor).
10. Midweek games already played sorted to the top of the Shootout tab → now hidden there.
11. Games in progress lose their posted line in the live scoreboard → build before kickoff.
12. New FBS members (pooled as FCS before) have no prior-season rating — less reliable early.

## 11. Suggested build plan for Claude Code

**Phase 1 — Run it.** Create the venv, run `bash pipeline/rebuild_all.sh`, confirm every acceptance
check in §3, open `docs/index.html`, run the smoke test. Don't change anything yet.

**Phase 2 — Publish.** `git init`, commit (the `.gitignore` keeps generated data out), create the GitHub
repo (`johnbrick123/ncaaf-matchups`), push, enable Pages on `/ (root)`, confirm the live URL loads.
Confirm with John that a public URL is OK first.

**Phase 3 — Automate.** Enable the workflow, trigger it once by hand, confirm it commits a new
`index.html` and a history file, and that the cache step restores on the second run.

**Phase 4 — Improvements (one at a time, ask John first).** Ideas, in rough priority order
(items 3 and 7 are alt-line features John has set aside — keep them off unless he asks):
1. **Results tab** from `data/forward_test.json` and `docs/history/` — grade each week's board vs what
   happened (totals/spreads vs closer with SE, bracket clear rates, the model-7+-off record).
2. **Total-disagreement flag:** add a `model unreliable here` pill when the model is 7+ points off the
   posted *total* (today only spread disagreement and 28+ favourites are flagged).
3. **Big-favourite panel** from `ncaaf_bigfav.py`: for favourites −14 or bigger, show the team-total
   route (favourite over implied total − k) next to the spread route, with clear rates and fair prices.
4. ~~Line-move tracker~~ — done Oct 25 2026 (Tab 3).
5. ~~Keep lines through kickoff~~ — done Oct 25 2026 (archive fallback in `build_ncaaf_board.py`).
6. **Mobile pass:** card tables are wide (min-width ~980 px) — consider a stacked layout under 600 px.
7. **Book-pricing check:** if John supplies his book's alt-total prices for a few games at different
   total levels, add them so the flat-pricing assumption can be tested instead of assumed.
