#!/usr/bin/env python3
"""render_ncaaf_board.py — bake ncaaf_board_data.json into the template.

Writes  artifact_body.html                      (for the Artifact tool: no doctype/html/head/body wrapper)
        NCAAF_Matchup_Scores_<season>_wk<week>.html   (standalone, for SendUserFile / GitHub Pages)
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
data_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "ncaaf_board_data.json")
tpl = open(os.path.join(BASE, "ncaaf_board_template.html")).read()
data = json.load(open(data_path))
payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
body = tpl.replace("/*DATA*/{}", payload)
open(os.path.join(BASE, "artifact_body.html"), "w").write(body)
standalone = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width,initial-scale=1">' + body + "</html>")
name = f"NCAAF_Matchup_Scores_{data['season']}_wk{data['week']}.html"
open(os.path.join(BASE, name), "w").write(standalone)
print("wrote artifact_body.html and", name, f"({len(standalone)//1024} KB)")
