"""Shared OpenRouter call logic used by smoke_test.py and diagnostic scripts."""
import json
import os
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_RETRIES = 3  # retries after the initial attempt -- up to 4 attempts total
BACKOFF_BASE_SECONDS = 1.0  # sleeps 1s, 2s, 4s between failed attempts


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def get_api_key() -> str:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set (check .env)")
    return api_key


def load_few_shot_messages(path: Path) -> list[dict]:
    messages = []
    for line in path.read_text().splitlines():
        if line.startswith("User: "):
            messages.append({"role": "user", "content": line[len("User: "):]})
        elif line.startswith("Assistant: "):
            messages.append({"role": "assistant", "content": line[len("Assistant: "):]})
    return messages


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    verbose: bool = False,
    max_retries: int = MAX_RETRIES,
    extra_payload: dict | None = None,
) -> dict:
    """Makes one logical call to OpenRouter, retrying transient failures with
    exponential backoff. Never raises for a request-level failure -- a bad
    call must not halt the caller's loop, since a dropped call would corrupt
    the denominator of every proportion derived from it.

    Returns a dict:
      raw_response:  str | None   -- None only if every attempt failed
      finish_reason: str | None   -- None if every attempt failed
      temperature:   float        -- what was actually in the request body on
                                      the attempt that produced this result
                                      (read back from resp.request.body, not
                                      the `temperature` argument); falls back
                                      to the argument only if every attempt
                                      failed before a request was ever sent
      failure:       str | None   -- None on success, else the last error
                                      after all retries were exhausted
      attempts:      int          -- how many HTTP attempts were made
      reasoning_tokens: int | None -- from body["usage"], if the API reports
                                      it; None if absent or every attempt
                                      failed. Reasoning is requested disabled
                                      (see payload below), so this is a check
                                      that the model actually honored that.
      id:            str | None   -- OpenRouter's generation id (body["id"]),
                                      None if every attempt failed. Log this
                                      so provider routing can be checked after
                                      the fact via GET /api/v1/generation.
      provider:      str | None   -- body["provider"], the upstream provider
                                      that actually served this call (e.g.
                                      "Amazon Bedrock", "GMICloud"). Some
                                      models load-balance across multiple
                                      providers per model id -- log this per
                                      call, don't assume it's constant within
                                      a cell.
      logprobs:      dict | None  -- choice["logprobs"] verbatim, if the API
                                      returned one (only present when the
                                      request asked for it via extra_payload);
                                      None otherwise or if every attempt
                                      failed.

    extra_payload, if given, is merged into the request body on top of the
    standard fields above (e.g. {"top_p": 1.0}, or {"logprobs": true,
    "top_logprobs": 20}, or {"provider": {"order": ["Novita"],
    "allow_fallbacks": False}}). Lets one-off diagnostic cells set extra
    sampling/routing params without a new function per param.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False},
    }
    if extra_payload:
        payload.update(extra_payload)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    total_attempts = 1 + max_retries
    last_error = None

    for attempt in range(1, total_attempts + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)

            if resp.request.body is None:
                raise RuntimeError("requests did not record a request body")
            sent_body = json.loads(resp.request.body)

            if verbose and attempt == 1:
                redacted_headers = {
                    k: ("Bearer ***" if k == "Authorization" else v) for k, v in resp.request.headers.items()
                }
                print("--- exact request sent to OpenRouter ---")
                print(f"POST {OPENROUTER_URL}")
                print("headers:", json.dumps(redacted_headers, indent=2))
                print("body:", json.dumps(sent_body, indent=2))
                print("-----------------------------------------")

            resp.raise_for_status()
            body = resp.json()
            choice = body["choices"][0]
            usage = body.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            reasoning_tokens = details.get("reasoning_tokens", usage.get("reasoning_tokens"))
            return {
                "raw_response": choice["message"]["content"],
                "finish_reason": choice.get("finish_reason"),
                "temperature": sent_body["temperature"],
                "failure": None,
                "attempts": attempt,
                "reasoning_tokens": reasoning_tokens,
                "id": body.get("id"),
                "provider": body.get("provider"),
                "logprobs": choice.get("logprobs"),
                "usage": usage or None,
            }
        except (requests.exceptions.RequestException, RuntimeError, KeyError, IndexError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < total_attempts:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

    return {
        "raw_response": None,
        "finish_reason": None,
        "temperature": temperature,
        "failure": last_error,
        "attempts": total_attempts,
        "reasoning_tokens": None,
        "id": None,
        "provider": None,
        "logprobs": None,
        "usage": None,
    }
