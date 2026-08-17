"""Per-model parameter capability table (TASK 9a).

Models declare which controls they accept instead of the harness assuming a
single uniform request shape. This exists because the gpt-5.x line was excluded
from the study on the stated grounds that it "doesn't support temperature"
(EXCLUSIONS.md) -- which a live probe on 2026-08-16 showed to be **false** for
gpt-5.4-nano. Every entry below is from a live 400-or-200 probe, not from docs.

Probe method: one real chat call per parameter, `max_completion_tokens=10`,
`reasoning_effort="none"`, checking whether the API returns 200 or a 400
`unsupported_parameter` / `unsupported_value`.
"""

# name -> {param: True/False}, plus notes. True = accepted live.
CAPABILITIES = {
    "gpt-5.4-nano": {
        "temperature": True,          # accepted at 0.0, 0.7 and 1.0 -- EXCLUSIONS.md is wrong
        "top_p": True,
        "frequency_penalty": True,
        "presence_penalty": True,
        "logprobs": True,             # the only 2026-gen model in this study that gives real logprobs
        "max_tokens": False,          # rejected: must use max_completion_tokens
        "max_completion_tokens": True,
        "stop": False,                # rejected outright
        "reasoning_effort": ["none", "low", "medium", "high", "xhigh"],
        "notes": "Real blockers are the max_tokens->max_completion_tokens rename and the absence of "
                 "`stop` -- NOT temperature. At reasoning_effort='none' with "
                 "max_completion_tokens=15 it answers in 4 tokens with reasoning_tokens=0, i.e. it "
                 "is directly comparable to the non-reasoning main grid.",
    },
    "gpt-5-nano": {
        "temperature": True,
        "max_tokens": False,
        "max_completion_tokens": True,
        "reasoning_effort": ["minimal", "low", "medium", "high"],
        "notes": "At max_completion_tokens=15 the entire budget is consumed by reasoning tokens and "
                 "content comes back empty -- the same failure mode claude-opus-5 showed with "
                 "extended thinking left on (run_v2.py's ROUTES comment). Needs a much larger "
                 "budget, which makes it NOT directly comparable to the max_tokens=15 main grid.",
    },
}


def supports(model: str, param: str) -> bool:
    entry = CAPABILITIES.get(model, {})
    val = entry.get(param)
    return bool(val) if not isinstance(val, list) else True


def request_kwargs(model: str, max_tokens: int, temperature: float,
                   reasoning_effort: str | None = None) -> dict:
    """Builds the kwargs this specific model actually accepts."""
    kw: dict = {}
    if supports(model, "max_completion_tokens"):
        kw["max_completion_tokens"] = max_tokens
    else:
        kw["max_tokens"] = max_tokens
    if supports(model, "temperature"):
        kw["temperature"] = temperature
    if reasoning_effort is not None:
        kw["reasoning_effort"] = reasoning_effort
    return kw
