"""Choose exactly one caller-provided option with an LLM."""

# 0) Imports
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, StrictInt, ValidationError
from ._adaptive_openai import (
    _AdaptiveProviderCaller,
    _batch_config,
    _batch_inputs,
    _run_adaptive_operations,
)
from ._openai_support import _OpenAIClient, resolve_openai_client, response_text
from ._prompt_support import build_messages, require_text

__all__ = ["OptionChoiceError", "choose_from_options"]

# 1) Sub functions
class _StructuredOptionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    choice: StrictInt


@dataclass(frozen=True)
class _ChoiceAttempt:
    protocol: str
    raw_response: str
    errors: tuple[str, ...]


class OptionChoiceError(ValueError):
    """The model did not identify exactly one supplied option."""

    def __init__(self, attempts: Sequence[_ChoiceAttempt]) -> None:
        self.attempts = tuple(attempts)
        final_errors = self.attempts[-1].errors if self.attempts else ("No response",)
        super().__init__("; ".join(final_errors))


def _validate_options(options: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(options, Mapping):
        raise TypeError("options must be a mapping of stable IDs to option text")
    if not options:
        raise ValueError("options must not be empty")

    normalized: dict[str, str] = {}
    seen_text: set[str] = set()
    for option_id, option_text in options.items():
        if not isinstance(option_id, str) or not option_id.strip():
            raise ValueError("option IDs must be nonblank strings")
        if option_id != option_id.strip():
            raise ValueError("option IDs must not have surrounding whitespace")
        if not isinstance(option_text, str) or not option_text.strip():
            raise ValueError("option text must be a nonblank string")
        cleaned_text = option_text.strip()
        text_key = cleaned_text.casefold()
        if text_key in seen_text:
            raise ValueError("option text must be unique ignoring case")
        normalized[option_id] = cleaned_text
        seen_text.add(text_key)
    return normalized


def _choice_prompt(
    input_text: str,
    options: Mapping[str, str],
    criteria: str,
    structured: bool,
) -> str:
    option_lines = "\n\n".join(
        f"{number}:\n  {option_text}"
        for number, option_text in enumerate(options.values(), start=1)
    )
    output_instruction = (
        "Return the selected option number in the requested structured format."
        if structured
        else f"Return only one integer from 1 to {len(options)}."
    )
    return f"""Choose the single best option for the input.

Input:
{input_text}

Options:
{option_lines}

Selection criteria:
{criteria}

Compare the complete option set. Option order is not evidence of quality.
{output_instruction}
"""


def _parse_compact_choice(
    raw_response: str,
    option_texts: Sequence[str],
) -> tuple[int | None, tuple[str, ...]]:
    response = raw_response.strip()
    if response.startswith("```"):
        response = re.sub(r"^```(?:text)?\s*", "", response, flags=re.IGNORECASE)
        response = re.sub(r"\s*```$", "", response).strip()

    if response.isdigit():
        return _validate_choice_number(int(response), len(option_texts))

    numbered = re.fullmatch(r"(\d+)\s*[.)-]\s*(.+)", response, flags=re.DOTALL)
    if numbered is not None:
        number = int(numbered.group(1))
        choice, errors = _validate_choice_number(number, len(option_texts))
        if errors:
            return None, errors
        echoed_text = numbered.group(2).strip()
        if echoed_text.casefold() == option_texts[number - 1].casefold():
            return choice, ()
        return None, (
            f"Choice {number} did not repeat the supplied option text exactly",
        )

    by_text = {
        option_text.casefold(): number
        for number, option_text in enumerate(option_texts, start=1)
    }
    exact_choice = by_text.get(response.casefold())
    if exact_choice is not None:
        return exact_choice, ()
    return None, ("Compact response did not identify exactly one supplied option",)


def _parse_structured_choice(
    raw_response: str,
    option_count: int,
) -> tuple[int | None, tuple[str, ...]]:
    try:
        payload = _StructuredOptionChoice.model_validate_json(raw_response)
    except ValidationError as error:
        errors = tuple(
            f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
            for item in error.errors()
        )
        return None, errors
    return _validate_choice_number(payload.choice, option_count)


def _validate_choice_number(
    choice: int,
    option_count: int,
) -> tuple[int | None, tuple[str, ...]]:
    if not 1 <= choice <= option_count:
        return None, (f"Choice {choice} is outside 1 to {option_count}",)
    return choice, ()


# 2) Single-input wrapper
def _choose_one(
    input_text: str,
    options: Mapping[str, str],
    criteria: str,
    client: _OpenAIClient,
    caller: _AdaptiveProviderCaller | None = None,
) -> str:
    system_prompt = """You compare supplied options and choose exactly one.
Follow the caller's selection criteria and return only the requested format.
"""
    max_structured_attempts = 2
    option_ids = list(options)
    attempts: list[_ChoiceAttempt] = []

    compact_request = {
        "input": build_messages(
            _choice_prompt(input_text, options, criteria, structured=False),
            system_prompt,
        ),
        "reasoning": {"effort": "none"},
        "temperature": 0,
        "max_output_tokens": 32,
    }
    compact_response = (
        client.create_response(**compact_request)
        if caller is None
        else caller.call(client.create_response, **compact_request)
    )
    compact_text = response_text(compact_response)
    compact_choice, compact_errors = _parse_compact_choice(
        compact_text,
        list(options.values()),
    )
    attempts.append(
        _ChoiceAttempt(
            protocol="compact-number",
            raw_response=compact_text,
            errors=compact_errors,
        )
    )
    if compact_choice is not None:
        selected_option_id = option_ids[compact_choice - 1]
        return selected_option_id

    structured_messages = build_messages(
        _choice_prompt(input_text, options, criteria, structured=True),
        system_prompt,
    )
    for attempt_number in range(1, max_structured_attempts + 1):
        structured_request = {
            "input": structured_messages,
            "text_format": _StructuredOptionChoice,
            "reasoning": {"effort": "none"},
            "temperature": 0,
            "max_output_tokens": 64,
        }
        structured_response = (
            client.parse_response(**structured_request)
            if caller is None
            else caller.call(client.parse_response, **structured_request)
        )
        structured_text = response_text(structured_response)
        structured_choice, structured_errors = _parse_structured_choice(
            structured_text,
            option_count=len(options),
        )
        attempts.append(
            _ChoiceAttempt(
                protocol=f"structured-{attempt_number}",
                raw_response=structured_text,
                errors=structured_errors,
            )
        )
        if structured_choice is not None:
            selected_option_id = option_ids[structured_choice - 1]
            return selected_option_id

    raise OptionChoiceError(attempts)


# 3) Single or batch wrapper
def choose_from_options(
    input_text: str | Mapping[str, str],
    options: Mapping[str, str],
    criteria: str,
    llm: Mapping[str, object],
    batch: bool | Mapping[str, object] = False,
) -> str | dict[str, str]:
    """Return one supplied option ID for one input or a stable-ID batch.

    The first call requests a compact answer. If it cannot be resolved exactly,
    the helper makes up to two structured recovery calls. Batch mode applies the
    same options and criteria to every input.
    """
    batch_config = _batch_config(batch)
    options = _validate_options(options)
    criteria = require_text(criteria, "criteria")
    client = resolve_openai_client(llm, disable_sdk_retries=batch_config is not None)
    if batch_config is None:
        input_text = require_text(input_text, "input_text")
        choice = _choose_one(input_text, options, criteria, client)
        return choice

    inputs = _batch_inputs(input_text)

    def choose_one(value: str, caller: _AdaptiveProviderCaller) -> str:
        return _choose_one(value, options, criteria, client, caller)

    choices = _run_adaptive_operations(
        list(inputs.values()),
        choose_one,
        model=client.model,
        scope="choose-from-options",
        config=batch_config,
    )
    choices_by_id = dict(zip(inputs, choices, strict=True))
    return choices_by_id
