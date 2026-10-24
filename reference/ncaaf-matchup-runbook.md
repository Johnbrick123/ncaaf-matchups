> **Reference copy — research log, not build instructions.** For building, follow `BUILD_SPEC.md` at the
> repo root. Paths like `claude/code/...` below refer to the Claude project where this was developed;
> in this repo the code lives in `pipeline/`. Artifact URLs below are the private Claude-hosted board.

# NCAAF Matchup Scores — runbook

How to rebuild John's college-football matchup board and its backtest in any new session.
All code is in `claude/code/` in this project (files prefixed `ncaaf_` plus `build_ncaaf_board.py`,
`render_ncaaf_board.py`, `ncaaf_board_template.html`).

**Live artifact URL:** https://claude.ai/artifact/CnQo8wzaq2GApWgNWMsWqe (Version 13 = 2026 week 8).
(An older link, claude.ai/code/artifact/5f6ba8cd-…, appears in early notes — do not publish there.)
Republish to that same URL by passing it as `url` — do not create a second artifact.
Standalone copy delivered each week as `NCAAF_Matchup_Scores_<season>_wk<week>.html` (data baked in —
John's browser blocks local files from fetching).

## What it is

One page, three views, same family as the MLB matchup site and the NFL weak-spot board
(row per team, compact, colour-coded, every coloured cell prints its number):

1. **This week** — one card per FBS-vs-FBS game, one row per team: opponent-adjusted offense
   (points rating, success rate, explosive rate), the opponent's adjusted defense (points allowed
   above average = positive is a bad defense, success/explosive allowed), tempo (context only),
   projected points, market-implied team total, delta. Footer line gives model line + total vs the
   posted line with an over/under lean and side-vs-line pill. Sort: spread gap / combined total /
   edge vs posted total / edge vs posted spread / kickoff. Min-edge filter. Raw-inputs toggle.
   Four weight sliders (points rating, success, explosive, home field; 1.0× = backtested fit).
2. **Rankings** — all FBS teams' current adjusted ratings, sortable.
3. **Backtest** — the 2015–2025 findings, charts, and the method, in plain English.

Colour: one diverging scale everywhere — **red = pushes points up** (strong offense, leaky defense,
over lean), **blue = pushes points down**, gray = league average. Fonts: Barlow Condensed (team codes,
headers), IBM Plex Sans (body), IBM Plex Mono (figures).

## How to run it (weekly, Tuesday morning is the natural slot — lines post Sunday night)

```
pip install requests pandas numpy scipy scikit-learn --break-system-packages   # if missing
mkdir ncaaf && cd ncaaf   # copy the claude/code/ncaaf_* files + build/render/template here
python3 ncaaf_pull.py all 2014 2026          # first time only: ~4 min for 10,500 games (cached in data/games/)
python3 build_ncaaf_board.py                  # refreshes current season, rebuilds features, writes ncaaf_board_data.json
python3 render_ncaaf_board.py                 # writes artifact_body.html + NCAAF_Matchup_Scores_<season>_wk<week>.html
```
`build_ncaaf_board.py [season] [week] [--no-refresh]` — flags may appear in any position (positional
args are filtered); week defaults to the first week with unplayed
games. It needs `data/backtest_results.json` (from `ncaaf_backtest.py`) for the Backtest tab; re-run
the backtest only when the model changes. The pull uses the default python-requests user agent on
purpose: browser-style UAs get 403 through the sandbox proxy.

**Fresh session (the container was reset on Oct 8 2026 and everything rebuilt from this project):** restore every `claude/code/ncaaf_*` file plus `build_ncaaf_board.py`, `render_ncaaf_board.py`, `ncaaf_board_template.html`, `ncaaf_rebuild_all.sh`, and `ncaaf_tail_model.json` → `data/tail_model.json`, then `WEEK=<n> bash rebuild_all.sh` (~6 min: 5 for the ESPN pull, under a minute for everything else). The Oct 8 rebuild reproduced all 1,335 values in `backtest_results.json` and all 395 in `bothfloor_results.json` exactly, and the SCORE coefficients to the third decimal. `ncaaf_minpts.py` (new) writes the `data/game_minpts.csv` that `ncaaf_bothfloor.py` needs — the original was an unsaved ad-hoc step.

Full rebuild of the study: `ncaaf_pull.py all 2014 2026` → `ncaaf_features.py` →
`ncaaf_ratings.py 2 0.75 0.9 features` → `ncaaf_backtest.py` (≈3 min) → `ncaaf_overs.py` (≈2 min;
the board build reads its `overs_results.json` and `game_signals.csv`) → `ncaaf_floor.py` (≈1 min) →
`ncaaf_bothfloor.py` (needs `data/game_minpts.csv`; the build reads `floor_results.json` and
`bothfloor_results.json`). `refresh_all.sh` chains the
weekly steps. Hyperparameter grid:
`ncaaf_cv.py <lam> <c> <rho>` per combo (training seasons only).

## Data (ESPN, keyless, all free)

- Schedule by week: `site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates=YYYY&seasontype=2|3&week=W&groups=80&limit=400`
  (calendar from the same endpoint with `dates=YYYY`). Upcoming games carry `competitions[].odds`.
- Game summary: `.../summary?event=ID` → box score (15 stats/team) + drives with full play-by-play
  (down, distance, yards to end zone, yardage, type, turnover flag).
- Odds archive: `sports.core.api.espn.com/v2/sports/football/leagues/college-football/events/ID/competitions/ID/odds`
  → 4–14 books; ESPN BET carries explicit open/close (2024+, partial 2023); 2015–16 have an
  "Opening" feed; 2017–2022 have `consensus`.
- Gotchas handled in `ncaaf_features.py`: (1) 2022–23 ESPN BET stores the *odds* (−110) in the
  open/close line fields — every open/close is validated against the headline number and a plausible
  range; (2) totals and spreads are picked independently across books because consensus carries spreads
  only in 2015–16; (3) `spread` is home-perspective (negative = home favoured), cross-checked against
  `details`; (4) from 2021 the PAT is logged outside the drive, so offensive points = 7×TD drives +
  3×FG drives, not the running score; (5) FBS membership = ESPN conference ids
  {1,4,5,8,9,12,15,17,18,37,151}; every other opponent is pooled into one "FCS" team.

Coverage: 2014–2025 ≈ 10,500 games; 9,400 with closing totals, 9,500 with spreads; ≈3,950 with both
an opening and a closing total.

## The score

For each stat (points, pts/drive, yds/play, success rate, explosive rate, plays, turnovers/drive,
sec/play, yds/drive, field position):
`stat = league + O(team) + D(opponent) + h·home`, ridge least squares on games **before** the week in
question, recency weight 0.9^weeks-ago, each team shrunk toward **0.75 × last season's final rating**
with the weight of **2 games**, home field pinned to last season's value (weight 50 games — without
that pin, Week 1's "big program hosts a cupcake" pattern inflates h to 11 points). Hyperparameters from a
grid on 2015–2024 only; the top five combos are within 0.05 points of each other.

Projection = as-of league average + OLS on
`O_pts, D_pts, O_sr, D_sr, O_xr, D_xr, home` (fit on all completed seasons before the current one; the
board's footer prints the live coefficients). The sliders scale those coefficient pairs.

## Findings (2015–2024 leave-one-season-out, 2025 untouched) — keep these on the page

| | 2015–24 | 2025 |
|---|---|---|
| Team-points miss, the score | 9.42 (corr .535) | 8.89 (corr .528) |
| Team-points miss, closing line implied | 9.08 (.576) | 8.59 (.572) |
| Team-points miss, raw PPG vs raw PA | 10.13 | 9.49 |
| Team-points miss, raw last-4 games | 10.21 | 9.60 |
| Direction vs league avg (team points) | 68% | 66% |
| Game total direction vs league avg | 62% | 57% |
| Straight-up winner, model / market | 72.3% / 73.6% | 72.0% / 72.9% |
| Totals vs closer (all disagreements) | 50.9% | 50.1% |
| Spreads vs closer | 50.0% | 49.1% |
| Closing total moved toward model | 57–65% each season | 67.4% (n=614) |
| Closing spread moved toward model | 57–58% (2023–24) | 58.0% (n=579) |
| Totals vs OPENER, edge ≥3 | 55.2% (n=1,478, 2015-16/23-24) | 52.2% (n=345) |
| Spreads vs OPENER, edge ≥5 | 54.4% (n=719) | 57.1% (n=161) |

- **What predicts points:** the opponent-adjusted points ratings do most of the work, and the
  *opponent's defense* rating matters at least as much as the team's own offense rating. Success rate
  adds a little (+0.044 miss if removed), explosiveness a hair (+0.014). Points/drive, yards/play,
  tempo, turnovers, field position add nothing once the points ratings are in. Raw unadjusted PPG is
  clearly worse; last-4-games is worse than season-to-date.
- **Recent form:** recency hurts. Heavy recency (0.6/week) raised the miss 9.46 → 9.83; mild (0.9)
  is neutral. Carrying 75% of last season's rating into Week 1 cut the Week 1–3 miss 11.3 → 9.8.
- **Tempo:** priced. Over rate by tempo tercile 49/50/48%.
- **Turnover luck:** real but priced. Turnover rate persists r=0.13 (margin 0.17) vs 0.50 for
  pts/drive, 0.48 success rate; lucky-early teams later scored −0.03 vs the score and +0.27 vs the
  market, unlucky −0.16 / +0.35 — no correction needed, no edge.
- **Line movement** is the one live thread: demeaning both sides leaves agreement at 57–66%;
  Spearman between market move and model deviation +0.23 to +0.38 every season (p<1e-8); a shuffled
  model scores 50.6% ± 1.0. Betting the opener is positive but thin — 2026 is the forward test. ESPN's
  archive keeps ESPN BET open/close automatically, so the sample grows without manual logging.

## Overs, Shootout score, run/pass (added Sep 6 2026 — `ncaaf_overs.py`)

John asked what to focus on for betting OVERS, for one combined "scoring" score, and whether
run-vs-pass matchups can be exploited. Screened 22 game-level signals against the closing total
(Stage A = correlation with the actual total; Stage B = over rate in the top vs bottom fifth,
per season, 2025 holdout). Results in `claude/code/ncaaf_overs_results.json`.

- **Base rate:** overs hit 49.2% (2015–24) and 48.9% (2025). Blind overs lose.
- **The posted total level is the strongest over/under signal, and it runs the "wrong" way:**
  totals 60+ went over 46.8% in 2015–24 and were under 50% in **all eleven seasons**; the 45–49.5
  bracket went over 51.7% (56.3% in 2025). The market over-prices scoring in shootouts.
- **Shootout score** (on the board) = percentile of the game's expected total against every
  FBS-vs-FBS game since 2015. It finds high-scoring games (corr 0.37 with the actual total) — and
  the games it ranks highest go UNDER more often (top fifth 46.7% overs). It ranks scoring, not value.
- **Rush edge** (both offenses' yds/carry above avg + both defenses' yds/carry allowed above avg)
  is the only matchup component with an over lean: top fifth 51.6% vs 46.8%, top > bottom in 9 of 11
  seasons, ≈ +1 point vs the closer. Weak but consistent in direction.
- **Run/pass matchup is a null in every form:** efficiency interaction, style alignment (offense
  lean × defense weakness), volume version (pass-heavy offense vs weak pass D) — all corr ≈ 0.00
  with points and with overs. Splitting the ratings into rush/pass barely moves the team-points
  miss (8.89 → 8.85). A defense's rushing yards allowed predicts opponent points better than its
  passing yards allowed (corr .27 vs .16) and persists more (.37 vs .23) — "bad run defense" is the
  more reliable exploitable-defense marker. Tempo, pass rate, explosiveness, domes, week, conference
  games: flat vs the line.
- **Over read** (on each card) = the parsimonious fit `(actual − posted) ≈ 7.2 − 0.121×posted +
  0.65×rush edge − 0.043×(model − posted)`: 52.1% at ±1 (n=2,309, LOSO), 52.6% in 2025 (n=369),
  56% at ±1.5 in 2025 (n=168), above 50% in 8 of 11 seasons. Break-even, not an edge; it mostly
  says over on low totals and under on high ones. Labeled that way on the page.
- **Blowout fix:** the linear score under-projected 35+ point favorites by ~5 (model 44.7, market
  49.7, actual 49.7). Two hinge terms (rating margin beyond ±21) cut the 2025 favorite bias in 28+
  point mismatches from +3.3 to +1.8 with no cost elsewhere; now part of SCORE_MODEL
  ("SCORE + blowout terms", coefficients ≈ 0.17 / 0.13). Model-says-under in 35+ spreads hit 43%
  historically — treat those unders as the model's blind spot.
- Board additions: Shootout chip in each card header, a second read line (rush/pass edge per side
  with percentiles, posted-total bracket over rate, over-read pill), sorts for Shootout score and
  Over read, run/pass columns in Rankings, Backtest §7 "Overs".

## The floor — buying the total down / alternate overs (added Sep 6 2026 — `ncaaf_floor.py`)

John's actual over strategy: with a posted total of 56, bet over 45 or over 40 — i.e. buy 11–16 points
at a price, in games where "both teams can put up points with confidence". Tested whether any
matchup trait thins the lower tail of the total relative to the closing number.

- Base curve (7,625 games): total − close has SD 16.3, skew +0.35, almost exactly normal in the lower
  tail. Clears −8: 68.7% (fair −220); −11: 74.8% (−297); −14: 80.6% (−416); −16: 84.4% (−542);
  −20: 90.1% (−911). Per-season range at −11: 70.5–78.4%. 2025: 75.4% / 85.9%.
- The tail scales with the **level** of the posted total: at −11, 78.5% on totals <45 vs 70.0% on
  65+ (SD 14.3 vs 18.7). Spread size barely matters (74–77%; 29+ spreads slightly tighter).
- **Nothing about the matchup moves it** (posted total held constant; logistic z and top-vs-bottom
  fifth within total brackets): both offenses' ratings, the weaker offense, both defenses, the better
  defense, success-rate consistency, explosiveness, tempo, pass rate, rush/pass edge, each team's
  as-of scoring volatility (SD of residuals vs its rating expectation), each team's raw
  20th-percentile points so far, the model's gap to the number — all z between −1.6 and +2.2, all
  flat within brackets (e.g. 74.4% vs 75.0%). A logistic model with all inputs gained nothing over
  posted-total-only in 2025 (log-lik −1.1/−0.8/−0.6/−0.1/+0.1 at 8/11/14/16/20); its top decile
  cleared −11 at 76.3% vs a 75.4% base rate.
- Board: each card prints the bracket's clear rates at −8/−11/−14/−16/−20 with fair prices; Backtest
  §8 has the tables. Message to John: the floor is set by the number, not the teams; compare the
  bracket rate to the book's alt price; the only exploitable structure is if a book prices the buy
  the same across total levels (then buy on low totals, never on 65+), and very deep buys are a touch
  safer than a normal curve implies (right skew).

## Model reliability vs the market — the disagreement calibration (added Sep 12 2026)

Asked of the data directly: when the model disagrees with the posted spread, who turns out right?
Measured on all 7,500+ FBS-vs-FBS games with closing spreads (`data/game_projections.csv`).

| model − market | n | model ATS | model's miss on the real margin | market's miss |
|---|---|---|---|---|
| 0–3 | 4,117 | 49.7% | 12.59 | 12.53 |
| 3–7 | 2,523 | 49.1% | 13.05 | 12.50 |
| 7–10 | 581 | 51.6% | 13.36 | 11.86 |
| 10–15 | 248 | **46.4%** | 16.60 | 12.97 |
| 15+ | 47 | 59.6% | 18.24 | 14.62 |

The 15+ cell looks good but is n=47 and sits on top of a much better-powered fact: **in the 405 games
where the market made a 28+ point favourite, the market's number averaged 33.9 against an actual
margin of 33.7 — nearly exact — while the model averaged 29.2 (about 4.8 short) and went 49.5% ATS.**
A large disagreement is the model losing accuracy, not finding value. In weeks 2–4 specifically the
model's miss at 7–15 points of disagreement is 14.70 vs the market's 12.38.

**On the board:** any game where the model is 10+ points off the market, or the market has a 28+ point
favourite, now carries a `model unreliable here` pill, and the over/side pills are suppressed for it,
with the calibration numbers printed underneath. Week 2 of 2026 flagged 12 of 49 games this way.
Keep this behaviour — without it the early-season mismatch games are the biggest-looking "edges" on
the page and they are all artifacts.

**Tested and rejected: down-weighting FCS games.** The suspicion was that blowouts of FCS opponents
(Louisiana Tech 80–6 over Northwestern State in 2026 wk 1, which moved their offense rating from −4.1
to +7.4) poison the early-season ratings. Weighting FCS games at 0.5 or 0.25 in the rating fit changed
the team-points miss by less than 0.03 and made the 2025 weeks 2–4 holdout **worse** (9.06 → 9.13 at
0.5, → 9.23 at 0.25). The blowout carries real information. `RatingEngine(fcs_w=...)` exists but stays
at 1.0.

## The both-teams scoring floor (added Sep 16 2026 — `ncaaf_bothfloor.py`)

John's framing: total points is one metric, but "how high is the number BOTH teams clear" is a second,
separate one — a 60-point game split 48-12 has a terrible floor. Built and validated.

- **Base rates** (7,623 games): the lower team's score has a median of 19. Both sides top 13 in **70.9%**
  of games, top 17 in **52.2%**, top 20 in **44.3%**, top 24 in **27.6%**. "Both teams score 21+" is
  a minority event league-wide, not a default.
- **Best predictor is the MARKET's implied underdog team total** `(total − |spread|)/2`: correlation
  **0.408** with the realised minimum, vs **0.366** for this model's own weaker-side projection and
  0.317 for the posted total. Adding our model on top of the market number gained nothing on the 2025
  holdout (log-likelihood within ±0.4 at every threshold). So the card's floor is a read on the
  market's number, not an edge over it — say so.
- **Deployed as** per-threshold logistic fits on the implied dog total (X = 10…27), refit on all
  seasons; the board shows P(both > 13/17/20/24) and the "70% floor" (highest X both clear 70% of the
  time) and the 60% floor. Out-of-sample calibration on 2025 by tercile is tight — e.g. both > 20:
  predicted 26/43/58% vs actual 25/40/59%.
- **The honest ceiling:** even with an implied dog total above 28, both teams clear 20 only ~70.6% of
  the time and the 70% floor is 20 points. There is no common spot where "both will definitely get 21".
- **Board:** a `Floor N both clear, 70%` chip beside the Shootout chip, a `Both-teams floor` sort, a
  read line per card, and Backtest §9. Games where even 10 doesn't clear 70% display `<10`.
- **Why it earns its place:** it separates games the Shootout score conflates. 2026 wk 3 examples —
  WKU @ IU posted 60.5 with Shootout 75 but both-clear-20 at 10.9%; Kent State @ Ohio State posted
  59.5 with both-clear-20 at **6.3%** (spread −52.5); against UNT @ TXST, Shootout 92 **and** a 19-point
  floor. High total and high floor are different properties and only sometimes coincide.

## Deep alt-overs and the price ladder (added Sep 18 2026)

John's actual bet: a deep alternate over at a big price — e.g. a game posted 63.5, take over 47.5 at
**−600**. Two things decided by the data.

**1. That exact bet loses.** −600 needs **85.7%**. Buying 16 points on a 60–64.5 posted total clears
**82.1%** (n=1,095) — short by 3.6 points, **−4.2% ROI**, about −$2,530 per 100 bets of $600 risk.

**Points you must buy to break even, by posted-total bracket:**

| bracket | −300 | −400 | −500 | −600 | −800 |
|---|---|---|---|---|---|
| <45 | 9 | 11 | 13 | 15 | 16 |
| 45–49.5 | 10 | 12 | 15 | 16 | 18 |
| 50–54.5 | 11 | 14 | 16 | 17 | 19 |
| 55–59.5 | 12 | 14 | 16 | 17 | 19 |
| 60–64.5 | 13 | 15 | 17 | 19 | 21 |
| 65+ | 14 | 17 | 19 | 20 | 22 |

**2. The structural edge is the LEVEL of the total, not the matchup.** The same 16 points bought clears
89.6% on a sub-45 total but 80.0% on a 65+ total. If a book prices its alt ladder by points bought
rather than by the level of the number, the low-total games are underpriced and the high-total games
are where he overpays.

**3. A high both-teams floor does NOT help this bet — it slightly hurts.** Holding the posted total
constant and splitting by the market's implied underdog total, the most balanced third cleared −16
**1.9 points less often** than the most lopsided third, and the sign held in **all five** brackets
(−20: −0.9). Blowouts score points too; a deep alt-over needs a high total, not a shared one. The
both-teams floor stays on the board because it is the right metric for team totals and two-sided
scoring reads — it is the wrong one for this bet, and §8 now says so.

**Board:** every card prints a `What to take at each price` line — the deepest number that still beats
the juice at −200/−300/−400/−500/−600/−800, with the clear rate and the rate needed, computed from that
game's posted-total bracket. That is the line he should read at the betting window.

## Can the model separate two equally-priced alt-overs? (tested Sep 18 2026)

John's challenge: "two games at −600 aren't equal — the model is supposed to find the angle." Tested
properly, because it is the whole question for his bet.

**Method.** A deep alt-over is a bet on the LOWER TAIL, not the mean — so the target was P(total >
posted − k) and, separately, the size of the miss |total − posted| (the variance). Candidates:
combined turnover rate, spread size, the better defense's rating, pace, both teams' scoring
volatility and worst games so far, the weaker offense, success rate, the both-teams floor, the
market-vs-naive-total spike, week, games played. Fitted leave-one-season-out across all eleven
seasons; each season's games ranked; top fifth measured.

**Result: the matchup features SUBTRACT.** At a 16-point buy the top fifth by the full model cleared
**87.3%** vs **87.8%** for a ranking on the posted total alone; at 20 points, **92.3%** vs **93.3%**.
Combined turnover rate screened as significant single-feature (z = −2.17 at k=16, −2.82 at k=20) and
died in the ranking test — with 14 features screened that is the expected false positive. Variance
screen did find lopsidedness (|spread| t = −3.99, implied dog total t = +3.99, better defense
t = +2.49): **lopsided games have LOWER total variance**, the opposite of the "both teams can score"
intuition, and consistent with the earlier floor finding.

**What DOES separate them: the level of the number.** Lowest fifth of posted totals (avg 44.4):
a 20-point buy clears **93.3%** vs the 88.9% that −800 needs — above break-even in **10 of 11 seasons**.
Highest fifth: **86.2%**, below it. A 7-point spread in win rate between two bets a flat ladder prices
identically.

**EV at a flat-ladder book** (same price per points-bought regardless of the number's level):

| buy / price | lowest-fifth totals | all games | highest-fifth totals |
|---|---|---|---|
| 14 pts at −500 | 83.7% (+0.4% ROI) | 80.6% (−3.3%) | 76.1% (−8.7%) |
| 16 pts at −600 | 87.8% (**+2.4%**) | 84.4% (−1.5%) | 79.9% (−6.8%) |
| 20 pts at −800 | 93.3% (**+5.0%**) | 90.1% (+1.4%) | 86.2% (−3.0%) |

**The load-bearing assumption:** this only pays if his book charges the same for "20 points down"
on a 38.5 total as on a 66 total. A book that prices alt-totals off the real distribution kills it.
**Confirm his book's behaviour before sizing anything** — that is the open question, not the stats.

**Board:** the price-ladder line now tags each game `flat-ladder value zone` (posted ≤ 49.5),
neutral, or `overpriced zone` (posted > 59.5), with the 20-point clear rate against the −800 bar.

## The price checker (added Sep 18 2026) — supersedes the fixed-ladder framing

John's correction: **16 points is not always −600.** Books quote a price per game per number, so any
analysis assuming a fixed points-for-price ladder is answering the wrong question. The board now
carries a calculator instead.

**Tail model** (`data/tail_model.json`, fitted on 7,623 games): the miss `actual − posted` is modelled
as mean `4.838 − 0.0786 × posted` and SD `8.978 + 0.1322 × posted` (SD 14.9 at a posted 45, 17.6 at 65),
with the shape taken from the pooled standardised residual rather than a normal — that residual is
**right-skewed (+0.37)**, so the real lower tail is thinner than a normal curve: the 2nd percentile sits
at −1.87 SD against −2.05 for a normal. Deep buys are safer than a book using a normal assumption
would price them. Validation: the smooth model reproduces every empirical bracket clear rate to within
about one point at k = 11, 16 and 20.

**On each card:** `Price check — over [number] at [price]` → true clear rate, the rate the price needs,
ROI on risk, the fair price, points bought, and a take-it / pass verdict. Past 26 points bought it warns
that only 1.3% of games since 2015 missed by that much, so the estimate is extrapolated. (Worst miss in
eleven seasons: −49.5.)

**This replaces the earlier "buy N points at price P" tables as the operating tool.** Those tables are
still correct for a book with a flat ladder, but the calculator answers the real question game by game.

## Shootout tab (added Sep 18 2026 — `ncaaf_shootcal.py`, board tab 2)

Fourth tab on the board, built at John's request as the weekly sorting tool for over bets: one row
per game, matchup + Shootout percentile, sortable.

**How Shootout is computed.** Percentile rank of the game's expected total (the two model team
projections added) against the projected totals of every FBS-vs-FBS game in every *prior* season —
`np.searchsorted(sorted(prior_proj_totals), exp_total) / n * 100` in `build_ncaaf_board.py`. No
market input. Historical pool median 55.0, 90th pct 66.3, so UNT @ TXST at 67.3 expected lands at 92.

**What the percentile is worth to an over bettor** (`ncaaf_shootcal.py`, n=6,910 games with closing
totals, each game's percentile ranked against prior seasons only, so no season grades itself):

| Shootout | n | Avg posted | Avg actual | Over hits | Clears −12 | Clears −16 | Clears −20 | ROI, −16 at −600 |
|---|---|---|---|---|---|---|---|---|
| 0–20 | 1,655 | 46.0 | 47.4 | 51.8% | 80.8% | **88.0%** | 93.2% | **+2.6%** |
| 20–40 | 1,551 | 51.4 | 52.1 | 49.4% | 79.0% | 86.0% | 91.1% | +0.3% |
| 40–60 | 1,347 | 55.3 | 55.5 | 49.1% | 75.4% | 83.3% | 89.8% | −2.8% |
| 60–80 | 1,324 | 59.6 | 59.9 | 47.9% | 75.2% | 83.5% | 89.0% | −2.5% |
| 80–90 | 570 | 63.8 | 63.7 | 46.3% | 74.6% | 81.4% | 88.1% | −5.0% |
| 90–100 | 463 | 69.5 | 68.3 | 44.3% | 70.8% | **78.8%** | 84.7% | **−8.0%** |

Monotone in every column; Spearman(Shootout, clears −16) = −0.070. Break-even at −600 is 85.7%.
The top bucket's posted totals (69.5) essentially matched what landed (68.3) — the market already
prices shootouts, and those games carry the widest tails, which is exactly what kills a deep alt over.

**Consequence for the tab design.** Sorting this tab descending — which is how John says he will use
it — orders the week from worst over bet to best on eleven seasons of evidence. So the tab carries
the bucket record in two columns on every row (clears −16, ROI at −600) plus the full table beneath,
and a "Historical over-bet edge" sort that inverts the default. Do not remove those columns: without
them the tab is a scoring ranking wearing a betting label.

**Still load-bearing and still unverified:** the ROI column assumes a book charging a flat price per
points bought regardless of the total's level. John has not yet confirmed that at his book.

## 2026 forward test — running scorecard (updated Oct 22 2026, through week 7)

Run `python3 ncaaf_forward.py [through_week]` (now part of `refresh_all.sh`); it writes
`data/forward_test.json`. **Correction, Oct 22:** the ad-hoc forward test used on Sep 24, Oct 1 and Oct 8
projected every past game with the *current* week's league mean instead of that week's own as-of mean.
The as-of mean fell from 26.2 (week 1) to about 24, so week-1 totals were projected ~4.5 points too low
and the early weeks looked much worse than they were. The "39% → 66% climb" reported on Oct 8 was mostly
that error; the corrected series has no trend. Spreads were unaffected (the mean cancels in the margin).

388 completed FBS-vs-FBS games, all with closing totals:

| | Model | Market |
|---|---|---|
| Total MAE | 12.27 | **11.81** |
| Margin MAE | 12.69 | **11.36** |
| Winner straight-up | 71.1% | **75.3%** |
| Totals vs closer | 54.4% (SE 2.5) | — |
| Spreads vs closer | 46.2% | — |

Totals by week (corrected): 53 / 55 / 51 / 59 / 59 / 48 / 56%. Spreads: 41 / 51 / 46 / 44 / 40 / 55 / 46%.
League over rate 50.3%. 54.4% is under one standard error above the 52.4% break-even, and the backtest says
~51% (2015–24 50.9%, 2025 50.6%) — treat it as noise until it holds much longer. **When the model sits
7+ points off the closing total it has been right 40.0% (n=35)** — consistent with 46.8% on 729 such games
in 2015–24. Big total disagreements are the model failing, not an edge. Week-6 examples: UGA @ ALA model
+14 over the 51.5 posted, final 37; TEX @ OU model −10 under the 40.5, final 51.

**What is holding: the posted-total bracket effect on the buy-down.** Clears −16 by posted bracket:
<45 89% (n=38) · 45–49.5 91% (104) · 50–54.5 86% (145) · 55–59.5 **78%** (82) · 60–64.5 82% (17).
Posted under 50: **90.8%** (n=142); posted 55+: **79.2%** (n=101). Same direction as twelve seasons of
history, wider. Still the only piece with money in it, still gated on the unverified flat-pricing assumption.

**Board change (Oct 22):** the Shootout tab now hides games already played earlier in the week (they
were sorting to the top of both orderings); a caption counts them and they stay on the This week tab.

**New FBS members:** North Dakota State joined the Mountain West in 2026. Before 2026 it was pooled
into the "FCS" pseudo-team, so its 2026 ratings carry no prior-season FBS rating — read its games
with more caution early in the year.

## Big favourites bought down (added Sep 26 2026 — `ncaaf_bigfav.py`)

John's question: on a game like CMU at Miami posted −41.5, can we take the favourite at a bought-down
number (−27 instead of −41.5), and can we tell which big favourites blow the doors off from the ones
that nearly get upset? Dataset: 2,848 favourite-games posted −14 or bigger, 2014–2026, one row per
game with the favourite's and dog's as-of ratings.

### 1. There is no "level" edge on the spread, unlike the total

Regressing |margin error| on the posted spread gives a slope of **+0.032/pt (p=0.02)**; the same
regression for the total gives **+0.106/pt (p=2e-13)**. Margin error SD is ≈15.8 at every spread
level (15.59 under −7, 16.06 over −21) and the error is symmetric (skew −0.02) and near-normal. The
market's number is unbiased at every level: avg posted −30.7 → avg actual 30.0; −38.4 → 38.7.

So the totals finding (low posted totals underpriced on a flat ladder) has **no spread analogue**. A
flat price per point bought off the spread is roughly fairly priced. Files: `data/margin_tail_model.json`,
`data/margin_z.npy`.

### 2. Bigger favourites are slightly WORSE to buy down, not better

| Posted | n | Clears −12 | −15 | −18 | −21 |
|---|---|---|---|---|---|
| −14 to −21 | 1,132 | 79.4% | **86.0%** | 90.0% | 92.8% |
| −21 to −28 | 600 | 71.8% | 77.5% | 84.5% | 89.7% |
| −28 to −35 | 231 | 73.6% | 79.7% | 83.5% | 86.1% |
| −35 to −60 | 129 | 79.8% | 83.7% | 89.1% | 92.2% |

Chi-square says the level matters at K≥12 (p=0.002 to 0.011), and the direction is against the big
ones: the −21 to −35 band is the worst. Note the thin inventory — only 360 games at −28 or bigger in
twelve seasons, so read the two right-hand brackets as directional.

**As a FRACTION of the number it flips.** Share of favourites failing to deliver half the posted
number: −14 to −21 → 30.7%, −21 to −28 → 27.8%, −28+ → **16.1%**. So "take −27 instead of −45"
(0.60× the number) is safer than "take −12 instead of −20" (also 0.60×), but "buy 18 points" is
safer on the −20 game than on the −45 game. Which framing matters depends entirely on whether the
book prices the alt spread by points bought or by fraction of the line — **still unverified.**

### 3. Nothing predicts which big favourite blows it out — complete null

Sixteen univariate features tested against the favourite's error vs the posted spread, then partialled
on the posted spread: model edge, rating margin, favourite's defense, dog's offense, dog's floor
(dog offense vs favourite defense), pace, favourite explosiveness, dog turnover rate, favourite
offense/success/explosive, dog defense allowed, home favourite. Every partial |r| ≤ 0.045; only the
posted total reached p<0.05 (r=+0.045), which does not survive 16 comparisons.

L2 logistic on all eleven features, target = favourite clears posted−K: **LOSO log-likelihood gain is
negative in 8–9 of 11 seasons** at K=15/18/21. The 2025 holdout top quartile clears 94.3% vs 90.5%
overall (n=53) — inside noise, and LOSO says it does not generalize. Same verdict as the run/pass
matchup study: the market has already priced the mismatch.

### 4. What DOES work: bet the favourite's own team total, not the spread

Failure anatomy at 18 points bought (321 failures in 2,561): the favourite underscored its implied
total in 91.0% of them (avg 15.8 short), the dog overscored in 88.5% (avg 13.5 over), both in 79.4%.
The favourite's own variance is the bigger half — but the point is that betting the **spread** makes
you carry BOTH teams' variance, and betting the favourite's **team total** carries only one:

| Error measured against | Mean | SD |
|---|---|---|
| Posted spread (margin) | +0.05 | **15.84** |
| Favourite's implied team total | +0.03 | **12.07** |
| Dog's implied team total | −0.02 | 10.01 |
| Posted game total | — | 15.52 |

12.07² + 10.01² = 15.68² ≈ the observed 15.84 — the margin's variance is the quadrature sum, and
buying the team total drops the dog's half. Both implied totals are unbiased to within 0.03 points.

Clear rates, favourites −14 or bigger:

| Buy | Spread (train / 2025) | Fav team total (train / 2025) | Fair price, spread → team total |
|---|---|---|---|
| 12 | 76.4% / 81.5% | 83.2% / 81.0% | −323 → −494 |
| 15 | 82.3% / 86.7% | **88.8% / 89.6%** | −464 → **−795** |
| 18 | 87.5% / 90.5% | **92.7% / 91.9%** | −698 → **−1262** |
| 21 | 91.3% / 93.4% | 95.9% / 95.3% | −1054 → −2362 |

Per season, buying 15 off the favourite's team total cleared the −600 break-even (85.7%) in **13 of 13
seasons, 2014–2026** (range 86.5%–92.0%); the spread route cleared it in 3 of 13. At 18 down the team
total ran 91.0%–95.8% every season. It also holds across home vs road favourites (88.8% / 88.9%) and
low vs high posted totals (88.2% / 88.4%) — no subgroup dependence.

**The unresolved question is identical to the overs one:** these fair prices are deep (−795, −1262),
so the route only pays if the book quotes better than fair on an alternate team total. Books may not
offer team totals that deep at all. John needs to check what is actually posted.

## Honesty requirements — do not drop these

The caveat block at the top of "This week" must keep saying: the score predicts the game (numbers
above), does **not** beat the closing line (2025: totals 50%, spreads 49%), the closing line moves
toward it 58–67% of the time, and the opener record is positive but small-n. Never restyle the delta
column or the pills as an edge. Same standard as the NFL and MLB boards.

## Open items

1. **Forward test on 2026** — track weekly: line-move agreement, and hit rate vs the posted line at
   build time (Tuesday ≈ the opener). The board records the posted line in `ncaaf_board_data.json`
   each build; keep those files.
2. **GitHub Pages package** — John wants this once he's happy with the board: a repo with the
   scripts, a `.github/workflows/weekly.yml` (Tuesday 12:00 UTC: pull → build → render →
   copy the standalone file to `index.html` → commit), Pages on `main`. Keyless, no secrets.
   `claude/code/ncaaf_weekly.yml` is a ready-to-drop-in workflow.
3. CFBD advanced stats (EPA, SP+) as a v2 test — needs a free key; expect small gains at most, since
   success rate and explosiveness are already in.
4. Second-half / team-total markets remain untested here as in every other sport.
