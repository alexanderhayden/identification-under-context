#!/bin/zsh
cd /Users/alexanderhayden/Projects/identification-under-context
# Chain behind TASK 11 -- both hit the same Anthropic endpoint, and interleaving
# two call streams risks 429s that would surface as retries in the data.
while [ $(ls data/v2/grid/*.jsonl 2>/dev/null | wc -l) -lt 15 ] || \
      [ $(for f in data/v2/grid/*.jsonl; do wc -l < "$f"; done | awk '$1>=100' | wc -l) -lt 15 ]; do
  sleep 60
done
echo "TASK 11 complete -- starting extended dose-response"
python3 run_task10_extended.py
