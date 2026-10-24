#!/bin/bash
# rebuild_all.sh — the whole pipeline, from ESPN pull to the published page.
#   bash pipeline/rebuild_all.sh              # current season, next week with unplayed games
#   WEEK=9 bash pipeline/rebuild_all.sh       # force a week
#   SEASON=2027 bash pipeline/rebuild_all.sh  # force a season
# First run pulls ~11,000 games from ESPN (about 5 minutes, cached in pipeline/data/games/).
# Later runs only fetch new games; everything after the pull takes under a minute.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2
SEASON="${SEASON:-$(python3 -c 'import datetime as d; t=d.date.today(); print(t.year if t.month>=7 else t.year-1)')}"
export SEASON
echo "== season $SEASON"
python3 ncaaf_pull.py all 2014 "$SEASON" 2>&1 | tail -2
python3 ncaaf_features.py | tail -1
python3 ncaaf_ratings.py 2 0.75 0.9 features 2>&1 | tail -1
python3 ncaaf_backtest.py > backtest_report.txt 2>&1      # backtest_results.json, game_projections.csv
python3 ncaaf_overs.py    > overs_report.txt 2>&1         # overs_results.json, game_signals.csv
python3 ncaaf_shootcal.py | tail -7                       # shoot_calibration.json (Shootout tab caveat)
# research-only scripts, not needed for the board: ncaaf_minpts.py, ncaaf_floor.py, ncaaf_bothfloor.py, ncaaf_bigfav.py
python3 ncaaf_forward.py 2>&1 | grep -v -E "Warning|c /=" | tail -10 || true   # forward_test.json
python3 build_ncaaf_board.py "$SEASON" ${WEEK:-} --no-refresh 2>&1 | grep -v -E "Warning|c /=" | tail -3
python3 render_ncaaf_board.py
# publish: newest standalone page -> docs/index.html, plus a dated copy of the data
mkdir -p ../docs/history
latest=$(python3 -c 'import json; d=json.load(open("ncaaf_board_data.json")); print("NCAAF_Matchup_Scores_%s_wk%s.html" % (d["season"], d["week"]))')
cp "$latest" ../docs/index.html
cp ncaaf_board_data.json "../docs/history/board_$(date -u +%F).json"
echo "== published $latest -> docs/index.html"
