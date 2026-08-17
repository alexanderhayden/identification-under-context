"""Fetches https://openrouter.ai/api/v1/models and prints a table of
candidate models for this study. Does not pick models or touch config.yaml.

Filters to models whose supported_parameters includes "temperature" (required
for this study's temperature=1.0 sampling), groups by provider prefix, sorts
each group newest-first by the API's `created` timestamp, and shows the top 5
per group.
"""
import argparse
from datetime import datetime, timezone

import requests

MODELS_URL = "https://openrouter.ai/api/v1/models"
PROVIDER_PREFIXES = ["anthropic/", "openai/", "google/", "meta-llama/", "qwen/", "deepseek/", "mistralai/"]
TOP_N = 5


def fetch_models() -> list[dict]:
    resp = requests.get(MODELS_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]


def price_per_million(price_str) -> str:
    if price_str is None:
        return "n/a"
    return f"${float(price_str) * 1_000_000:.2f}/M"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()

    models = fetch_models()
    candidates = [m for m in models if "temperature" in m.get("supported_parameters", [])]

    for prefix in PROVIDER_PREFIXES:
        group = [m for m in candidates if m["id"].startswith(prefix)]
        group.sort(key=lambda m: m.get("created", 0), reverse=True)
        top = group[: args.top_n]

        print(f"\n=== {prefix} ({len(group)} candidates, showing top {len(top)}) ===")
        header = f"{'id':<45} {'reasoning':<10} {'prompt $/M':<12} {'completion $/M':<15} {'context':<10} {'created'}"
        print(header)
        print("-" * len(header))
        for m in top:
            supported = m.get("supported_parameters", [])
            has_reasoning = "reasoning" in supported or "include_reasoning" in supported
            pricing = m.get("pricing", {})
            created_ts = m.get("created")
            created_str = (
                datetime.fromtimestamp(created_ts, tz=timezone.utc).strftime("%Y-%m-%d") if created_ts else "n/a"
            )
            print(
                f"{m['id']:<45} {str(has_reasoning):<10} "
                f"{price_per_million(pricing.get('prompt')):<12} "
                f"{price_per_million(pricing.get('completion')):<15} "
                f"{m.get('context_length', 'n/a'):<10} {created_str}"
            )


if __name__ == "__main__":
    main()
