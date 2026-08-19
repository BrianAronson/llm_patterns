"""Map user-provided table columns to caller-defined target fields."""

# 0) Imports
from __future__ import annotations
import json
import keyword
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING
from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from ._openai_support import resolve_openai_client, response_text
from ._prompt_support import build_messages
if TYPE_CHECKING:
    from pandas import DataFrame

__all__ = ["TableMappingError", "map_table_columns"]


# 1) Sub functions
@dataclass(frozen=True)
class _MappingAttempt:
    raw_response: str
    errors: tuple[str, ...]


class TableMappingError(ValueError):
    """The model did not produce a legal table mapping."""

    def __init__(self, attempts: Sequence[_MappingAttempt]) -> None:
        self.attempts = tuple(attempts)
        final_errors = self.attempts[-1].errors if self.attempts else ("No response",)
        super().__init__("; ".join(final_errors))


def _build_source_profile(
    source_columns: Sequence[str] | Mapping[str, Iterable[object]] | DataFrame,
) -> dict[str, list[str]]:
    profile: dict[str, list[str]] = {}
    items = getattr(source_columns, "items", None)
    if callable(items):
        for name, values in items():
            _add_source_column(profile, name, values)
    else:
        if isinstance(source_columns, (str, bytes)):
            raise TypeError("source_columns must not be a string")
        try:
            names = iter(source_columns)
        except TypeError as error:
            raise TypeError(
                "source_columns must be a sequence of names, mapping, or DataFrame"
            ) from error
        for name in names:
            _add_source_column(profile, name, ())
    if not profile:
        raise ValueError("source_columns must not be empty")
    return profile


def _add_source_column(
    profile: dict[str, list[str]],
    name: object,
    values: Iterable[object],
) -> None:
    max_sample_values = 3
    if not isinstance(name, str) or not name.strip():
        raise ValueError("source column names must be nonblank strings")
    if name in profile:
        raise ValueError("source column names must be unique")
    if isinstance(values, (str, bytes)):
        samples: Iterable[object] = (values,)
    else:
        try:
            samples = iter(values)
        except TypeError as error:
            raise TypeError(
                f"sample values for source column {name!r} must be iterable"
            ) from error
    profile[name] = [
        _bounded_sample(value) for value in islice(samples, max_sample_values)
    ]


def _validate_target_fields(target_fields: Mapping[str, str]) -> dict[str, str]:
    if not target_fields:
        raise ValueError("target_fields must not be empty")

    normalized: dict[str, str] = {}
    for name, description in target_fields.items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or keyword.iskeyword(name)
        ):
            raise ValueError(
                "target field names must be valid non-keyword Python identifiers"
            )
        if not isinstance(description, str) or not description.strip():
            raise ValueError("target field descriptions must be nonblank strings")
        normalized[name] = description.strip()
    return normalized


def _bounded_sample(value: object) -> str:
    max_sample_characters = 120
    text = str(value)
    if len(text) <= max_sample_characters:
        return text
    return text[: max_sample_characters - 1] + "…"


def _mapping_prompt(
    source_profile: Mapping[str, Sequence[str]],
    target_fields: Mapping[str, str],
) -> str:
    return f"""Map these source table columns to the required target fields.

Target fields and their meanings:
{json.dumps(target_fields, ensure_ascii=False, indent=2)}

Source columns and representative values:
{json.dumps(source_profile, ensure_ascii=False, indent=2)}

Return one JSON object whose keys are exactly the target field names and whose
values are exact source column names. Map every target field once. Do not use
one source column for multiple target fields.
"""


def _validate_mapping_response(
    raw_response: str,
    output_model: type[BaseModel],
    source_names: set[str],
) -> tuple[dict[str, str] | None, tuple[str, ...]]:
    try:
        value = output_model.model_validate_json(raw_response)
    except ValidationError as error:
        errors = tuple(
            f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
            for item in error.errors()
        )
        return None, errors

    mapping = value.model_dump()
    unknown = sorted(set(mapping.values()) - source_names)
    reused = sorted(
        source for source, count in Counter(mapping.values()).items() if count > 1
    )
    errors: list[str] = []
    if unknown:
        errors.append(f"Unknown source columns: {unknown}")
    if reused:
        errors.append(f"Source columns mapped more than once: {reused}")
    return (None, tuple(errors)) if errors else (mapping, ())


# 2) Wrapper function
def map_table_columns(
    source_columns: Sequence[str] | Mapping[str, Iterable[object]] | DataFrame,
    target_fields: Mapping[str, str],
    llm: Mapping[str, object],
) -> dict[str, str]:
    """Return one real, distinct source column for every target field.

    ``source_columns`` may contain names, names with representative values, or
    a DataFrame-like object. At most three short values per source column are
    sent to the model. One correction is allowed after an invalid mapping.
    """
    system_prompt = """You map source table columns to caller-defined target fields.
Use only the exact source column names supplied by the caller.
"""

    source_profile = _build_source_profile(source_columns)
    target_fields = _validate_target_fields(target_fields)
    client = resolve_openai_client(llm, required_response_methods=("parse",))
    if len(source_profile) < len(target_fields):
        raise ValueError(
            "source_columns must contain at least as many columns as target_fields"
        )

    output_model = create_model(
        "TableColumnMapping",
        __config__=ConfigDict(extra="forbid"),
        **{name: (str, ...) for name in target_fields},
    )
    prompt = _mapping_prompt(source_profile, target_fields)
    original_messages = build_messages(prompt, system_prompt)
    messages = original_messages
    attempts: list[_MappingAttempt] = []

    for attempt_number in (1, 2):
        response = client.parse_response(
            input=messages,
            text_format=output_model,
            reasoning={"effort": "none"},
        )
        raw_response = response_text(response)
        mapping, errors = _validate_mapping_response(
            raw_response,
            output_model,
            source_names=set(source_profile),
        )
        attempts.append(_MappingAttempt(raw_response=raw_response, errors=errors))
        if mapping is not None:
            validated_mapping = {target: mapping[target] for target in target_fields}
            return validated_mapping
        if attempt_number == 2:
            break

        correction = (
            "The proposed mapping failed deterministic validation. Correct the "
            "mapping using only the supplied target fields and exact source column "
            "names.\n\nValidation errors:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
        messages = [
            *original_messages,
            {"role": "assistant", "content": raw_response},
            {"role": "user", "content": correction},
        ]

    raise TableMappingError(attempts)
