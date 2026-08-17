"""Single source of truth for the base-vs-instruct prompt pair (TASK 3).

The whole point of a matched base/instruct comparison is that the two arms
differ ONLY in chat templating. So both arms are built here, from the same
inputs, by one function each, and `dump_pair()` prints them side by side for
audit (TASK 0a). Nothing else in this session constructs a base or instruct
prompt.

What is identical across the two arms:
  - the 11 reference few-shot pairs, same order, same text
  - the task prompt (given_short / not_given_short), same {WORD1}/{WORD2}/{P}/
    {Q}/{R}/{SEED} substitutions from config.yaml
  - the trailing "My answer is:" assistant prefill
  - temperature, and the word pair / seed

What differs, and only this:
  - BASE receives one flat string with literal "User: " / "Assistant: " turn
    markers, sent to a completions endpoint. No chat template is applied at any
    layer.
  - INSTRUCT receives the same content as role-tagged message dicts, sent to a
    chat endpoint, where the model's own chat template renders the turn markers
    server-side.

Note on the prefill: the reference appends
`Message(role="assistant", content="My answer is:")` to EVERY sample, both
branches (AUDIT.md item 3, sad/.../utils.py:43,63). Giving it to both arms here
is therefore both reference-faithful and the thing that makes the two arms
semantically identical -- omitting it on the instruct arm only would leave the
base arm forced to answer while the instruct arm was free to preamble or
refuse, which is an asymmetry in the measurement, not in the models.
"""
from pathlib import Path

import yaml

from run import build_messages, fill_template
from openrouter_client import load_few_shot_messages

ROOT = Path(__file__).resolve().parent

PREFILL = "My answer is:"


def load_prompt_inputs(word_pair: tuple[str, str] | None = None) -> dict:
    """Loads config.yaml and the prompt/few-shot files, applies the v2 faithful
    settings (single-token word pair unless overridden), and returns everything
    both arms need.
    """
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    w1, w2 = word_pair if word_pair else (cfg["single_token_word1"], cfg["single_token_word2"])
    cfg["word1"], cfg["word2"] = w1, w2

    few_shot_messages = load_few_shot_messages(ROOT / "prompts" / "few_shot_reference.txt")
    few_shot_text = (ROOT / "prompts" / "few_shot_reference.txt").read_text().rstrip("\n")

    return {
        "cfg": cfg,
        "few_shot_messages": few_shot_messages,
        "few_shot_text": few_shot_text,
        "given_prompt": fill_template((ROOT / "prompts" / "given.txt").read_text(), cfg),
        "not_given_prompt": fill_template((ROOT / "prompts" / "not_given.txt").read_text(), cfg),
    }


def build_base_prompt(inputs: dict, cell: str) -> str:
    """Flat text for a base model. Ends in the prefill, no trailing newline --
    the model continues directly after the colon.
    """
    task_prompt = inputs["given_prompt"] if cell.startswith("given") else inputs["not_given_prompt"]
    return f"{inputs['few_shot_text']}\nUser: {task_prompt}\nAssistant: {PREFILL}"


def build_instruct_messages(inputs: dict, cell: str) -> list[dict]:
    """Role dicts for an instruction-tuned model. Same content as
    build_base_prompt, with the turn markers left to the chat template.
    """
    return build_messages(
        cell,
        inputs["few_shot_messages"],
        [],  # no filler: every cell in TASK 3 is a "_short" cell
        inputs["given_prompt"],
        inputs["not_given_prompt"],
        assistant_prefill=PREFILL,
    )


def flatten_instruct(messages: list[dict]) -> str:
    """Renders the instruct message array back into the same flat form the base
    arm gets, so the two can be diffed for semantic equality (TASK 0a).
    """
    role_marker = {"user": "User: ", "assistant": "Assistant: "}
    return "\n".join(f"{role_marker[m['role']]}{m['content']}" for m in messages)


def dump_pair(cell: str, inputs: dict | None = None) -> tuple[str, list[dict]]:
    inputs = inputs or load_prompt_inputs()
    base = build_base_prompt(inputs, cell)
    instruct = build_instruct_messages(inputs, cell)
    return base, instruct
