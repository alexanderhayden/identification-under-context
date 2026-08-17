"""Direct-provider call logic (Anthropic, OpenAI) for comparison runs that
bypass OpenRouter entirely. Mirrors openrouter_client.py's call_openrouter:
same return schema, same retry/backoff behavior, plus a "client" field
("anthropic" or "openai") recording which direct API served the call. Does
not touch openrouter_client.py or the OpenRouter path in run.py.

"provider" in the return dict keeps its existing meaning from the
OpenRouter path (upstream backend that served the call, e.g. "Amazon
Bedrock"). There is no upstream routing on a direct call, so it's always the
literal provider name ("Anthropic" / "OpenAI") on success, None on failure --
"client" is the field that actually distinguishes which API this record came
from, since a downstream consumer joining OpenRouter and direct-provider
records together needs a stable discriminator that isn't overloaded.
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

MAX_RETRIES = 3  # retries after the initial attempt -- up to 4 attempts total
BACKOFF_BASE_SECONDS = 1.0  # sleeps 1s, 2s, 4s between failed attempts


def get_anthropic_api_key() -> str:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set (check .env or shell environment)")
    return api_key


def get_openai_api_key() -> str:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set (check .env or shell environment)")
    return api_key


def _failure_result(client: str, temperature: float, last_error: str | None, total_attempts: int) -> dict:
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
        "client": client,
        "usage": None,
    }


def call_anthropic(
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    max_retries: int = MAX_RETRIES,
    extra_payload: dict | None = None,
) -> dict:
    """One logical call to Anthropic's Messages API directly (no OpenRouter).
    `model` must be a provider-native model id (e.g. "claude-opus-5"), not an
    OpenRouter-style "anthropic/claude-opus-5" slug. `extra_payload` merges
    into the request kwargs, e.g. {"thinking": {"type": "disabled"}}.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if extra_payload:
        kwargs.update(extra_payload)

    total_attempts = 1 + max_retries
    last_error = None
    for attempt in range(1, total_attempts + 1):
        try:
            resp = client.messages.create(**kwargs)
            text = "".join(block.text for block in resp.content if block.type == "text")
            return {
                "raw_response": text,
                "finish_reason": resp.stop_reason,
                "temperature": temperature,
                "failure": None,
                "attempts": attempt,
                "reasoning_tokens": None,
                "id": resp.id,
                "provider": "Anthropic",
                "logprobs": None,
                "client": "anthropic",
                "usage": resp.usage.model_dump() if resp.usage else None,
            }
        except anthropic.APIError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < total_attempts:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

    return _failure_result("anthropic", temperature, last_error, total_attempts)


def call_openai(
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    max_retries: int = MAX_RETRIES,
    extra_payload: dict | None = None,
) -> dict:
    """One logical call to OpenAI's chat completions API directly (no
    OpenRouter). `model` must be a provider-native model id (e.g. "gpt-4.1"),
    not an OpenRouter-style "openai/gpt-4.1" slug. `extra_payload` merges
    into the request kwargs, e.g. {"logprobs": True, "top_logprobs": 20}.
    """
    import openai

    client = openai.OpenAI(api_key=api_key)
    kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if extra_payload:
        kwargs.update(extra_payload)

    total_attempts = 1 + max_retries
    last_error = None
    for attempt in range(1, total_attempts + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            usage = resp.usage
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
            logprobs = choice.logprobs.model_dump() if choice.logprobs else None
            return {
                "raw_response": choice.message.content,
                "finish_reason": choice.finish_reason,
                "temperature": temperature,
                "failure": None,
                "attempts": attempt,
                "reasoning_tokens": reasoning_tokens,
                "id": resp.id,
                "provider": "OpenAI",
                "logprobs": logprobs,
                "client": "openai",
                "usage": usage.model_dump() if usage else None,
            }
        except openai.APIError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < total_attempts:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

    return _failure_result("openai", temperature, last_error, total_attempts)


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def _ollama_call(endpoint: str, payload: dict, temperature: float, client_name: str,
                 max_retries: int = MAX_RETRIES) -> dict:
    """Shared transport for both Ollama endpoints. Local, no API key, no
    upstream routing -- `provider` is always the literal "Ollama" on success so
    the field keeps the same meaning it has on every other path.
    """
    import requests

    url = f"{OLLAMA_BASE_URL}/{endpoint}"
    total_attempts = 1 + max_retries
    last_error = None
    for attempt in range(1, total_attempts + 1):
        try:
            resp = requests.post(url, json=payload, timeout=600)
            resp.raise_for_status()
            body = resp.json()
            choice = body["choices"][0]
            text = choice.get("text") if endpoint == "completions" else choice.get("message", {}).get("content")
            return {
                "raw_response": text,
                "finish_reason": choice.get("finish_reason"),
                "temperature": temperature,
                "failure": None,
                "attempts": attempt,
                "reasoning_tokens": None,
                "id": body.get("id"),
                "provider": "Ollama",
                "logprobs": choice.get("logprobs"),
                "client": client_name,
                "usage": body.get("usage"),
            }
        except Exception as exc:  # requests raises a family of exceptions, not one base APIError
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < total_attempts:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

    return _failure_result(client_name, temperature, last_error, total_attempts)


def call_ollama_chat(model: str, messages: list[dict], temperature: float, max_tokens: int,
                     max_retries: int = MAX_RETRIES, extra_payload: dict | None = None) -> dict:
    """Instruction-tuned local models through Ollama's OpenAI-compatible chat
    endpoint -- the model's own chat template is applied server-side.
    """
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if extra_payload:
        payload.update(extra_payload)
    return _ollama_call("chat/completions", payload, temperature, "ollama_chat", max_retries)


def call_ollama_completion(model: str, prompt: str, temperature: float, max_tokens: int,
                           max_retries: int = MAX_RETRIES, extra_payload: dict | None = None) -> dict:
    """Base (non-instruction-tuned) local models through Ollama's legacy
    completions endpoint: raw text in, raw continuation out, no chat template
    applied at any layer.
    """
    payload = {"model": model, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
    if extra_payload:
        payload.update(extra_payload)
    return _ollama_call("completions", payload, temperature, "ollama_completion", max_retries)


def call_openai_completion(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = MAX_RETRIES,
    extra_payload: dict | None = None,
) -> dict:
    """One logical call to OpenAI's legacy /v1/completions endpoint (base
    models: davinci-002, babbage-002 -- no chat template, no instruction
    tuning). `prompt` is a single plain-text string, not a messages array.
    `extra_payload` merges into the request kwargs, e.g. {"logprobs": 5}.
    Same return schema as call_openai/call_anthropic; "finish_reason" and
    "id" map onto this endpoint's own fields of the same name.
    """
    import openai

    client = openai.OpenAI(api_key=api_key)
    kwargs = {"model": model, "prompt": prompt, "temperature": temperature, "max_tokens": max_tokens}
    if extra_payload:
        kwargs.update(extra_payload)

    total_attempts = 1 + max_retries
    last_error = None
    for attempt in range(1, total_attempts + 1):
        try:
            resp = client.completions.create(**kwargs)
            choice = resp.choices[0]
            usage = resp.usage
            details = getattr(usage, "completion_tokens_details", None) if usage else None
            reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
            logprobs = choice.logprobs.model_dump() if choice.logprobs else None
            return {
                "raw_response": choice.text,
                "finish_reason": choice.finish_reason,
                "temperature": temperature,
                "failure": None,
                "attempts": attempt,
                "reasoning_tokens": reasoning_tokens,
                "id": resp.id,
                "provider": "OpenAI",
                "logprobs": logprobs,
                "client": "openai_completion",
                "usage": usage.model_dump() if usage else None,
            }
        except openai.APIError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < total_attempts:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

    return _failure_result("openai_completion", temperature, last_error, total_attempts)
