"""Apply patch-sized LLM edits without asking it to regenerate a document."""

# 0) Imports
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError, create_model
from ._openai_support import resolve_openai_client, response_text
from ._prompt_support import build_messages, require_text

__all__ = ["SelectedLineEditError", "edit_selected_lines"]

# 1) Sub functions
@dataclass(frozen=True)
class _EditAttempt:
    protocol: str
    raw_response: str
    errors: tuple[str, ...]


class SelectedLineEditError(ValueError):
    """The model did not return legal replacements for the selected lines."""

    def __init__(self, attempts: Sequence[_EditAttempt]) -> None:
        self.attempts = tuple(attempts)
        final_errors = self.attempts[-1].errors if self.attempts else ("No response",)
        super().__init__("; ".join(final_errors))


def _validate_selected_lines(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("selected_lines must be a sequence of strings")
    selections = tuple(values)
    if any(not isinstance(value, str) for value in selections):
        raise TypeError("selected_lines must contain only strings")
    if any(not value.strip() for value in selections):
        raise ValueError("selected_lines must not contain blank text")
    if any("\n" in value or "\r" in value for value in selections):
        raise ValueError("selected_lines entries must each contain exactly one line")
    if len(set(selections)) != len(selections):
        raise ValueError("selected_lines must not contain duplicates")
    return selections


def _resolve_line_indexes(
    document_lines: Sequence[str],
    selected_lines: Sequence[str],
) -> dict[str, int]:
    line_texts = [_line_text(line) for line in document_lines]
    resolved: dict[str, int] = {}
    for selection in selected_lines:
        matches = [
            index
            for index, line_text in enumerate(line_texts)
            if line_text == selection
        ]
        if not matches:
            raise ValueError(
                f"selected line is not present in the document: {selection!r}"
            )
        if len(matches) > 1:
            raise ValueError(
                "selected line occurs more than once in the document and is ambiguous: "
                f"{selection!r}"
            )
        resolved[selection] = matches[0]
    return resolved


def _edit_prompt(
    document: str,
    selected_lines: Mapping[str, str],
    instruction: str,
    context: str | None,
) -> str:
    context_section = f"\n\nRelevant context:\n{context}" if context is not None else ""
    return f"""Instruction:
{instruction}{context_section}

Document:
{document}

Selected lines:
{json.dumps(selected_lines, ensure_ascii=False, indent=2)}

Return compact JSON with exactly the same keys and one complete single-line replacement per key. Preserve each Markdown prefix. Return unchanged text when no edit is needed.
"""


def _recovery_prompt(original_prompt: str, errors: Sequence[str]) -> str:
    error_list = "\n".join(f"- {error}" for error in errors)
    return f"""{original_prompt}

The previous response failed deterministic validation. Return a corrected
replacement mapping in the requested structured format.

Validation errors:
{error_list}
"""


def _validate_replacements(
    raw_response: str,
    output_model: type[BaseModel],
    selected_lines: Mapping[str, str],
) -> tuple[dict[str, str] | None, tuple[str, ...]]:
    try:
        value = output_model.model_validate_json(raw_response)
    except ValidationError as error:
        errors = tuple(
            f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
            for item in error.errors()
        )
        return None, errors

    replacements = value.model_dump()
    errors: list[str] = []
    for alias, replacement in replacements.items():
        if not replacement.strip():
            errors.append(f"{alias}: replacement must not be blank")
        if "\n" in replacement or "\r" in replacement:
            errors.append(f"{alias}: replacement must remain one line")
        if _markdown_prefix(replacement) != _markdown_prefix(selected_lines[alias]):
            errors.append(f"{alias}: replacement changes the Markdown prefix")
    return (None, tuple(errors)) if errors else (replacements, ())


def _apply_replacements(
    lines: Sequence[str],
    replacements: Mapping[str, str],
    line_indexes: Mapping[str, int],
) -> str:
    edited = list(lines)
    for alias, replacement in replacements.items():
        line_index = line_indexes[alias]
        edited[line_index] = replacement + _line_ending(edited[line_index])
    edited_document = "".join(edited)
    return edited_document


def _line_text(line: str) -> str:
    ending = _line_ending(line)
    return line[: -len(ending)] if ending else line


def _line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    if line.endswith("\r"):
        return "\r"
    return ""


def _markdown_prefix(line: str) -> tuple[str, str]:
    patterns = (
        ("heading", r"^(\s*#{1,6}\s+)"),
        ("bullet", r"^(\s*[-*+]\s+)"),
        ("numbered", r"^(\s*\d+[.)]\s+)"),
        ("quote", r"^(\s*>\s*)"),
    )
    for kind, pattern in patterns:
        match = re.match(pattern, line)
        if match:
            return kind, match.group(1)
    indentation = line[: len(line) - len(line.lstrip())]
    return "text", indentation


def _max_output_tokens(lines: Iterable[str]) -> int:
    characters = sum(len(line) for line in lines)
    return max(128, min(4096, 64 + characters * 2))


# 2) Wrapper function
def edit_selected_lines(
    document: str,
    selected_lines: Sequence[str],
    instruction: str,
    llm: Mapping[str, object],
    context: str | None = None,
) -> str:
    """Use the whole document as context but request only selected replacements.

    The caller identifies editable lines by their exact text. Each selected line
    must occur exactly once in the document. The model sees the complete document
    but must return only one single-line replacement per selection. Replacements
    that change a line's Markdown prefix are rejected, and valid replacements are
    applied locally without reconstructing any unselected text.
    """
    system_prompt = "Use the document only as context. Edit only the selected lines and return only their replacement mapping."
    max_structured_attempts = 2

    if not isinstance(document, str):
        raise TypeError("document must be a string")
    instruction = require_text(instruction, "instruction")
    context = require_text(context, "context") if context is not None else None
    selections = _validate_selected_lines(selected_lines)
    if not selections:
        return document

    lines = document.splitlines(keepends=True)
    line_indexes = _resolve_line_indexes(lines, selections)
    selected_by_alias = {
        f"e{selection_number}": selection
        for selection_number, selection in enumerate(selections, start=1)
    }
    index_by_alias = {
        alias: line_indexes[selection] for alias, selection in selected_by_alias.items()
    }

    client = resolve_openai_client(llm)
    output_model = create_model(
        "SelectedLineReplacements",
        __config__=ConfigDict(extra="forbid"),
        **{alias: (StrictStr, ...) for alias in selected_by_alias},
    )
    prompt = _edit_prompt(
        document,
        selected_by_alias,
        instruction=instruction,
        context=context,
    )
    messages = build_messages(prompt, system_prompt)
    max_output_tokens = _max_output_tokens(selected_by_alias.values())
    attempts: list[_EditAttempt] = []

    compact_response = client.create_response(
        input=messages,
        reasoning={"effort": "none"},
        temperature=0,
        max_output_tokens=max_output_tokens,
    )
    compact_text = response_text(compact_response)
    replacements, errors = _validate_replacements(
        compact_text,
        output_model,
        selected_lines=selected_by_alias,
    )
    attempts.append(
        _EditAttempt(
            protocol="compact-json",
            raw_response=compact_text,
            errors=errors,
        )
    )
    if replacements is not None:
        edited_document = _apply_replacements(lines, replacements, index_by_alias)
        return edited_document

    recovery_errors = errors
    for attempt_number in range(1, max_structured_attempts + 1):
        recovery_prompt = _recovery_prompt(prompt, recovery_errors)
        structured_response = client.parse_response(
            input=build_messages(recovery_prompt, system_prompt),
            text_format=output_model,
            reasoning={"effort": "none"},
            temperature=0,
            max_output_tokens=max_output_tokens,
        )
        structured_text = response_text(structured_response)
        replacements, recovery_errors = _validate_replacements(
            structured_text,
            output_model,
            selected_lines=selected_by_alias,
        )
        attempts.append(
            _EditAttempt(
                protocol=f"structured-{attempt_number}",
                raw_response=structured_text,
                errors=recovery_errors,
            )
        )
        if replacements is not None:
            edited_document = _apply_replacements(
                lines, replacements, index_by_alias
            )
            return edited_document

    raise SelectedLineEditError(attempts)
