"""TASK 9b cost probe. Uses the v2 main-grid prompt (no prefill) so this
reasoning-ON condition contrasts against the reasoning-OFF grid it will be
tabled beside. Reasoning cannot be disabled on these endpoints (verified live:
400 "Reasoning is mandatory for this endpoint and cannot be disabled")."""
import json, yaml
from pathlib import Path
from dotenv import dotenv_values
from run import build_messages, fill_template
from openrouter_client import call_openrouter, load_few_shot_messages
ROOT=Path('.')
cfg=yaml.safe_load(open('config.yaml'))
cfg['word1'],cfg['word2']=cfg['single_token_word1'],cfg['single_token_word2']
fs=load_few_shot_messages(ROOT/'prompts'/'few_shot_reference.txt')
gp=fill_template((ROOT/'prompts'/'given.txt').read_text(),cfg)
ngp=fill_template((ROOT/'prompts'/'not_given.txt').read_text(),cfg)
msgs=build_messages('given_short',fs,[],gp,ngp,assistant_prefill=None)
key=dotenv_values('.env')['OPENROUTER_API_KEY']

MODELS=["google/gemini-3.5-flash","google/gemini-3.6-flash","google/gemini-3.7-flash",
        "qwen/qwen3.8-max","qwen/qwen3.8-2.4t-a95b","meta/muse-spark-1.2"]
res={}
for m in MODELS:
    costs=[];rts=[];cts=[];ok=0;err=None;last=None
    for _ in range(2):
        r=call_openrouter(key,m,msgs,1.0,4000,
            extra_payload={"usage":{"include":True},"reasoning":{"enabled":True}})
        if r["failure"]: err=str(r["failure"])[:110]; continue
        ok+=1; last=r; u=r.get("usage") or {}
        costs.append(u.get("cost",0) or 0); cts.append(u.get("completion_tokens",0))
        rts.append((u.get("completion_tokens_details") or {}).get("reasoning_tokens",0))
    if ok:
        res[m]=dict(cost=sum(costs)/ok,ctok=sum(cts)/ok,rtok=sum(rts)/ok)
        print(f"{m:30} ${sum(costs)/ok:.5f}/call ctok={sum(cts)/ok:5.0f} rtok={sum(rts)/ok:5.0f} "
              f"resp={str(last['raw_response'])[:45]!r}",flush=True)
    else:
        print(f"{m:30} FAILED: {err}",flush=True)
print("\n--- PROJECTED: 100 calls per model, given_short only ---")
tot=0
for m,d in res.items():
    c=d['cost']*100; tot+=c; print(f"  {m:30} ${c:8.2f}")
print(f"  TOTAL ({len(res)} models) ${tot:.2f}")
json.dump(res,open('task9b_costs.json','w'))
