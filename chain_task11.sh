#!/bin/zsh
cd /Users/alexanderhayden/Projects/identification-under-context
# Wait for TASK 10 to finish all four dose cells before starting TASK 11 --
# both hit the same Anthropic endpoint and interleaving them risks 429s that
# would show up as retries in the data.
while true; do
  done_cells=$(for f in data/v2/dose/*.jsonl; do [ -f "$f" ] && wc -l < "$f"; done | awk '$1>=100' | wc -l)
  [ "$done_cells" -ge 4 ] && break
  sleep 30
done
echo "TASK 10 complete -- starting TASK 11"
python3 run_task11.py
