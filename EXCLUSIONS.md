# Model exclusions

Evidence for why each model below was excluded from the study. This is a record of what
happened, for the report — not just bookkeeping.

All test calls used the study's actual invariants: `temperature: 1.0`, `max_tokens: 15`,
`reasoning: {"enabled": false}` sent as a fixed constant on every request (see
`openrouter_client.py` — this is never made conditional per-model, since mixing
reasoning-on and reasoning-off models across cells would make them non-comparable).

## OpenAI gpt-5.x line (no `temperature` parameter)

**Not an API error — a silent no-op.** `supported_parameters` for every current OpenAI
proprietary model (`gpt-5.6-*`, `gpt-5.5*`, `gpt-5.4*`, `gpt-chat-latest`, and back through the
whole gpt-5.x line) omits `temperature` entirely, replaced by `reasoning_effort`. Sending
`temperature: 1.0` anyway does **not** produce an error — OpenRouter returns `200` and the call
succeeds normally:

```
POST openai/gpt-5.6-luna, temperature: 1.0, reasoning: {"enabled": false}
-> 200 OK
{"choices":[{"finish_reason":"stop","message":{"content":"banana", ...}}],
 "usage":{...,"completion_tokens_details":{"reasoning_tokens":0,...}}}
```

This is excluded on capability grounds, not a runtime failure: since `temperature` isn't in
`supported_parameters`, there's no evidence the value is actually honored by the sampler —
OpenRouter silently drops parameters a model doesn't declare support for rather than rejecting
the request. Including these models would silently violate the study's `temperature: 1.0`
design constraint with no error to catch it.

The last OpenAI proprietary model confirmed to still declare `temperature` support is
`openai/gpt-4.1` (used in the study). The `gpt-oss-*` open-weight line also still declares it.

## google/gemini-3.7-flash (mandatory reasoning)

```
POST google/gemini-3.7-flash, reasoning: {"enabled": false}
-> 400 Bad Request
{"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.",
 "code":400,"metadata":{"provider_name":null,
 "previous_errors":[{"code":400,"message":"Reasoning is mandatory for this endpoint and cannot be disabled."}]}}}
```

## google/gemini-3.6-flash (mandatory reasoning)

```
POST google/gemini-3.6-flash, reasoning: {"enabled": false}
-> 400 Bad Request
{"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.",
 "code":400,"metadata":{"provider_name":null,
 "previous_errors":[{"code":400,"message":"Reasoning is mandatory for this endpoint and cannot be disabled."}]}}}
```

## google/gemini-3.5-flash-lite (mandatory reasoning)

```
POST google/gemini-3.5-flash-lite, reasoning: {"enabled": false}
-> 400 Bad Request
{"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.",
 "code":400,"metadata":{"provider_name":null,
 "previous_errors":[{"code":400,"message":"Reasoning is mandatory for this endpoint and cannot be disabled."}]}}}
```

## qwen/qwen3.8-max (mandatory reasoning)

```
POST qwen/qwen3.8-max, reasoning: {"enabled": false}
-> 400 Bad Request
{"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.",
 "code":400,"metadata":{"provider_name":null}}}
```

Discovered in the full run before this was caught as a systematic issue: 100/100 calls failed
identically across all completed cells (`given_short`, `not_given_short`, `given_long`), each
burning 4 retry attempts with exponential backoff before giving up, since the error is
deterministic and not transient. The failure-only data files were deleted after confirming
every record in them had `failure` set and `raw_response: null`.

## qwen/qwen3.8-2.4t-a95b (mandatory reasoning)

```
POST qwen/qwen3.8-2.4t-a95b, reasoning: {"enabled": false}
-> 400 Bad Request
{"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.",
 "code":400,"metadata":{"provider_name":null,
 "previous_errors":[... same message, repeated across retries]}}}
```

## meta/muse-spark-1.2 (mandatory reasoning)

Two separate issues were hit for this model, in sequence:

1. **Account-level gate (resolved, not why it's excluded):** the first full run attempt
   returned `403` on every call:
   ```
   {"error":{"message":"This model requires you to complete the following before use: 18+ age confirmation. Confirm at https://openrouter.ai/settings/preferences.",
    "code":403,"metadata":{"missing_attestation_types":["age_18plus"],"provider_name":null}}}
   ```
   The age attestation was subsequently completed in OpenRouter account settings, and a
   follow-up call confirmed the 403 no longer occurs.

2. **Mandatory reasoning (actual exclusion reason):** with the age gate cleared, the model
   still rejects `reasoning: {"enabled": false}`:
   ```
   POST meta/muse-spark-1.2, reasoning: {"enabled": false}
   -> 400 Bad Request
   {"error":{"message":"Reasoning is mandatory for this endpoint and cannot be disabled.",
    "code":400,"metadata":{"provider_name":null}}}
   ```

`meta/muse-spark-1.2` is excluded on the same mandatory-reasoning grounds as the other five
models above, independent of the (resolved) account gate.

## Substitutes used instead

- `google/gemini-3.1-flash-lite-image` — confirmed `200` with `reasoning: {"enabled": false}`.
- `qwen/qwen3.7-plus` — confirmed `200` with `reasoning: {"enabled": false}`.

Both are older/smaller than the flagship models they replace; see `PROJECT.md` /
`config.yaml` for the final model roster actually used in the study.
