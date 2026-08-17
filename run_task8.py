"""TASK 8: gpt-3.5-turbo-0613 bridge cell via OpenRouter (retired on OpenAI direct)."""
import json, time
from datetime import datetime, timezone
from pathlib import Path
from base_instruct_prompts import load_prompt_inputs, build_instruct_messages
from openrouter_client import call_openrouter, get_api_key
from score import parse_answer

ROOT=Path(__file__).resolve().parent
OUT=Path("data/v2/bridge/openrouter__gpt-3.5-turbo-0613_given_short.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
msgs=build_instruct_messages(load_prompt_inputs(),"given_short")
key=get_api_key()
run_id=datetime.now(timezone.utc).isoformat()
existing=sum(1 for _ in OUT.open()) if OUT.exists() else 0
print(f"starting at {existing}/100")
with OUT.open("a") as f:
    for i in range(existing,100):
        r=call_openrouter(key,"openai/gpt-3.5-turbo-0613",msgs,1.0,15,
            extra_payload={"provider":{"order":["Azure"],"allow_fallbacks":False},"usage":{"include":True}})
        rec={"run_id":run_id,"model":"gpt-3.5-turbo-0613","cell":"given_short","prompt_id":"given",
             "raw_response":r["raw_response"],"parsed_response":parse_answer(r["raw_response"]),
             "timestamp":time.time(),"temperature":r["temperature"],"finish_reason":r["finish_reason"],
             "failure":r["failure"],"attempts":r["attempts"],"reasoning_tokens":r["reasoning_tokens"],
             "id":r["id"],"provider":r["provider"],"logprobs":r["logprobs"],"client":"openrouter",
             "prefill_used":True,"sampling_params_sent":{},"usage":r.get("usage")}
        f.write(json.dumps(rec)+"\n"); f.flush()
        if (i+1)%25==0: print(f"  {i+1}/100 {r['raw_response']!r}")
print("done")
