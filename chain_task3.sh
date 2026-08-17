#!/bin/zsh
cd /Users/alexanderhayden/Projects/identification-under-context
# Wait for both matched instruct/base siblings to finish pulling, then run the
# TASK 3 base/instruct pairs. Times out after 4 hours rather than waiting forever.
deadline=$(( $(date +%s) + 14400 ))
while [ $(date +%s) -lt $deadline ]; do
  have_llama=$(ollama list | grep -c "llama3.1:8b-text-q4_K_M")
  have_mistral=$(ollama list | grep -c "mistral:7b-instruct-v0.2-q4_K_M")
  if [ "$have_llama" -ge 1 ] && [ "$have_mistral" -ge 1 ]; then
    echo "both matched siblings present -- starting TASK 3"
    python3 run_base_instruct.py
    exit 0
  fi
  sleep 60
done
echo "TIMEOUT: matched siblings never finished downloading"
ollama list
