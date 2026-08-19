"""Score every caller-provided option in one token-efficient LLM assignment."""

# 0) Imports
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, ConfigDict, ValidationError
from ._adaptive_openai import (
    _AdaptiveProviderCaller,
    _batch_config,
    _batch_inputs,
    _run_adaptive_operations,
)
from ._openai_support import _OpenAIClient, resolve_openai_client, response_text
from ._prompt_support import build_messages, require_text

__all__ = ["OptionScoringError", "score_options"]

# 1) Sub functions
class _StructuredOptionScores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scores: list[Literal[1, 2, 3, 4, 5]]


@dataclass(frozen=True)
class _ScoringAttempt:
    protocol: str
    raw_response: str
    errors: tuple[str, ...]


class OptionScoringError(ValueError):
    """The model did not return one legal score for every supplied option."""

    def __init__(self, attempts: Sequence[_ScoringAttempt]) -> None:
        self.attempts = tuple(attempts)
        final_errors = self.attempts[-1].errors if self.attempts else ("No response",)
        super().__init__("; ".join(final_errors))


def _validate_options(options: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(options, Mapping):
        raise TypeError("options must be a mapping of stable IDs to descriptions")
    if not options:
        raise ValueError("options must not be empty")

    normalized: dict[str, str] = {}
    for option_id, description in options.items():
        if not isinstance(option_id, str) or not option_id.strip():
            raise ValueError("option IDs must be nonblank strings")
        if option_id != option_id.strip():
            raise ValueError("option IDs must not have surrounding whitespace")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("option descriptions must be nonblank strings")
        normalized[option_id] = description.strip()
    return normalized


def _compact_prompt(
    input_text: str,
    options: Mapping[str, str],
    rubric: str,
    aliases: Sequence[str],
) -> str:
    option_lines = "\n\n".join(
        f"{alias}:\n  {description}"
        for alias, description in zip(aliases, options.values(), strict=True)
    )
    example = json.dumps({alias: 5 for alias in aliases}, separators=(",", ":"))
    return f"""Independently score how well every option fits the input.

Input:
{input_text}

Options:
{option_lines}

Scoring rubric:
{rubric}

Score every option from 1 to 5. Multiple options may receive the same score.
Option order is not evidence of quality.

Return only one compact JSON object with exactly these option aliases and integer scores.
Example shape:
{example}
"""


def _structured_prompt(
    input_text: str,
    options: Mapping[str, str],
    rubric: str,
) -> str:
    option_lines = "\n\n".join(
        f"Position {index}:\n  {description}"
        for index, description in enumerate(options.values(), start=1)
    )
    return f"""Independently score how well every option fits the input.

Input:
{input_text}

Options:
{option_lines}

Scoring rubric:
{rubric}

Score every option from 1 to 5. Multiple options may receive the same score.
Option order is not evidence of quality. Return one integer score for every
position in the exact order shown.
"""


def _parse_compact_scores(
    raw_response: str,
    aliases: Sequence[str],
) -> tuple[list[int] | None, tuple[str, ...]]:
    score_values = {1, 2, 3, 4, 5}
    normalized = raw_response.strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?\s*", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s*```$", "", normalized)
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start >= 0 and end > start:
        normalized = normalized[start : end + 1]

    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as error:
        return None, (f"Compact response was not valid JSON: {error.msg}",)
    if not isinstance(payload, dict):
        return None, ("Compact response must be a JSON object",)
    if set(payload) != set(aliases):
        return None, ("Compact response did not score every option exactly once",)

    scores: list[int] = []
    for alias in aliases:
        raw_score = payload[alias]
        if isinstance(raw_score, bool):
            return None, (f"{alias} must be an integer from 1 to 5",)
        if isinstance(raw_score, int):
            score = raw_score
        elif isinstance(raw_score, str) and raw_score.strip().isdigit():
            score = int(raw_score.strip())
        else:
            return None, (f"{alias} must be an integer from 1 to 5",)
        if score not in score_values:
            return None, (f"{alias} must be an integer from 1 to 5",)
        scores.append(score)
    return scores, ()


def _parse_structured_scores(
    raw_response: str,
    option_count: int,
) -> tuple[list[int] | None, tuple[str, ...]]:
    try:
        payload = _StructuredOptionScores.model_validate_json(raw_response)
    except ValidationError as error:
        errors = tuple(
            f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
            for item in error.errors()
        )
        return None, errors
    if len(payload.scores) != option_count:
        return None, (
            (
                f"Structured response returned {len(payload.scores)} scores for "
                f"{option_count} options"
            ),
        )
    return list(payload.scores), ()


def _stable_scores(options: Mapping[str, str], scores: Sequence[int]) -> dict[str, int]:
    return {option_id: score for option_id, score in zip(options, scores, strict=True)}


# 2) Single-input wrapper
def _score_one(
    input_text: str,
    options: Mapping[str, str],
    rubric: str,
    client: _OpenAIClient,
    caller: _AdaptiveProviderCaller | None = None,
) -> dict[str, int]:
    system_prompt = """You score every supplied option independently.
Follow the caller's rubric and return only the requested response format.
"""
    max_structured_attempts = 2
    aliases = [f"c{index}" for index in range(1, len(options) + 1)]
    maximum_output_tokens = max(128, 22 * len(options))
    attempts: list[_ScoringAttempt] = []

    compact_request = {
        "input": build_messages(
            _compact_prompt(input_text, options, rubric, aliases),
            system_prompt,
        ),
        "reasoning": {"effort": "none"},
        "temperature": 0,
        "max_output_tokens": maximum_output_tokens,
    }
    compact_response = (
        client.create_response(**compact_request)
        if caller is None
        else caller.call(client.create_response, **compact_request)
    )
    compact_text = response_text(compact_response)
    compact_scores, compact_errors = _parse_compact_scores(compact_text, aliases)
    attempts.append(
        _ScoringAttempt(
            protocol="compact-json",
            raw_response=compact_text,
            errors=compact_errors,
        )
    )
    if compact_scores is not None:
        scores_by_id = _stable_scores(options, compact_scores)
        return scores_by_id

    structured_prompt = _structured_prompt(input_text, options, rubric)
    structured_messages = build_messages(structured_prompt, system_prompt)
    for attempt_number in range(1, max_structured_attempts + 1):
        structured_request = {
            "input": structured_messages,
            "text_format": _StructuredOptionScores,
            "reasoning": {"effort": "none"},
            "temperature": 0,
            "max_output_tokens": maximum_output_tokens,
        }
        structured_response = (
            client.parse_response(**structured_request)
            if caller is None
            else caller.call(client.parse_response, **structured_request)
        )
        structured_text = response_text(structured_response)
        structured_scores, structured_errors = _parse_structured_scores(
            structured_text,
            option_count=len(options),
        )
        attempts.append(
            _ScoringAttempt(
                protocol=f"structured-{attempt_number}",
                raw_response=structured_text,
                errors=structured_errors,
            )
        )
        if structured_scores is not None:
            scores_by_id = _stable_scores(options, structured_scores)
            return scores_by_id

    raise OptionScoringError(attempts)


# 3) Single or batch wrapper
def score_options(
    input_text: str | Mapping[str, str],
    options: Mapping[str, str],
    rubric: str,
    llm: Mapping[str, object],
    batch: bool | Mapping[str, object] = False,
) -> dict[str, int] | dict[str, dict[str, int]]:
    """Return one score from 1 to 5 for every supplied option.

    The first call requests compact JSON. If it does not cover the option set
    exactly, the helper makes up to two structured recovery calls. Batch mode
    applies the same options and rubric to every input.
    """
    batch_config = _batch_config(batch)
    options = _validate_options(options)
    rubric = require_text(rubric, "rubric")
    client = resolve_openai_client(llm, disable_sdk_retries=batch_config is not None)
    if batch_config is None:
        input_text = require_text(input_text, "input_text")
        scores = _score_one(input_text, options, rubric, client)
        return scores

    inputs = _batch_inputs(input_text)

    def score_one(value: str, caller: _AdaptiveProviderCaller) -> dict[str, int]:
        return _score_one(value, options, rubric, client, caller)

    scores = _run_adaptive_operations(
        list(inputs.values()),
        score_one,
        model=client.model,
        scope="score-options",
        config=batch_config,
    )
    scores_by_id = dict(zip(inputs, scores, strict=True))
    return scores_by_id
