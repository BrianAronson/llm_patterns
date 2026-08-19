"""Generate validated structured records from caller-provided subjects."""

# 0) Imports
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    create_model,
)
from ._adaptive_openai import (
    _AdaptiveProviderCaller,
    _batch_config,
    _run_adaptive_operations,
)
from ._openai_support import _OpenAIClient, resolve_openai_client, response_text
from ._prompt_support import build_messages, require_text

__all__ = ["StructuredRecordError", "generate_structured_records"]

# 1) Sub functions
_ValueType = Literal["string", "integer", "number", "boolean", "string_list"]


class _FieldPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_name: StrictStr = Field(min_length=1)
    value_type: _ValueType
    nullable: StrictBool
    format_instruction: StrictStr = Field(min_length=1)


class _SchemaPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fields: list[_FieldPlan] = Field(min_length=1)


@dataclass(frozen=True)
class _RecordAttempt:
    raw_response: str
    errors: tuple[str, ...]


class StructuredRecordError(ValueError):
    """The model did not produce a valid record for one stable input ID."""

    def __init__(
        self,
        input_id: str,
        attempts: Sequence[_RecordAttempt],
    ) -> None:
        self.input_id = input_id
        self.attempts = tuple(attempts)
        final_errors = self.attempts[-1].errors if self.attempts else ("No response",)
        super().__init__(f"{input_id}: {'; '.join(final_errors)}")


def _validate_context(context: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(context, Mapping):
        raise TypeError("context must be a mapping of stable IDs to subject text")
    if not context:
        raise ValueError("context must not be empty")

    validated: dict[str, str] = {}
    for input_id, context_text in context.items():
        if not isinstance(input_id, str) or not input_id.strip():
            raise ValueError("input IDs must be nonblank strings")
        if input_id != input_id.strip():
            raise ValueError("input IDs must not have surrounding whitespace")
        validated[input_id] = require_text(context_text, "every context value")
    return validated


def _validate_questions(questions: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(questions, Mapping):
        raise TypeError("questions must be a mapping of field names to questions")
    if not questions:
        raise ValueError("questions must not be empty")

    validated: dict[str, str] = {}
    for field_name, question in questions.items():
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError("question field names must be nonblank strings")
        if field_name != field_name.strip():
            raise ValueError(
                "question field names must not have surrounding whitespace"
            )
        validated[field_name] = require_text(question, "every question")
    return validated


def _schema_plan_errors(
    plan: _SchemaPlan,
    questions: Mapping[str, str],
) -> tuple[str, ...]:
    planned_names = [field.field_name for field in plan.fields]
    expected_names = list(questions)
    errors: list[str] = []
    duplicates = sorted(
        {name for name in planned_names if planned_names.count(name) > 1}
    )
    missing = [name for name in expected_names if name not in planned_names]
    unexpected = [name for name in planned_names if name not in questions]
    whitespace = [name for name in planned_names if name != name.strip()]
    if duplicates:
        errors.append("duplicate fields: " + ", ".join(duplicates))
    if missing:
        errors.append("missing fields: " + ", ".join(missing))
    if unexpected:
        errors.append("unexpected fields: " + ", ".join(unexpected))
    if whitespace:
        errors.append("field names contain surrounding whitespace")
    return tuple(errors)


def _output_model(
    questions: Mapping[str, str],
    plan: _SchemaPlan,
) -> type[BaseModel]:
    planned_fields = {field.field_name: field for field in plan.fields}
    fields = {
        f"answer_{index}": (
            _field_annotation(planned_fields[field_name]),
            Field(
                alias=field_name,
                description=(
                    f"{question} Output format: "
                    f"{planned_fields[field_name].format_instruction}"
                ),
            ),
        )
        for index, (field_name, question) in enumerate(questions.items(), start=1)
    }
    return create_model(
        "StructuredRecord",
        __config__=ConfigDict(
            extra="forbid",
            loc_by_alias=True,
            populate_by_name=True,
        ),
        **fields,
    )


def _field_annotation(field: _FieldPlan) -> object:
    annotation: object
    if field.value_type == "string":
        annotation = StrictStr
    elif field.value_type == "integer":
        annotation = StrictInt
    elif field.value_type == "number":
        annotation = StrictFloat
    elif field.value_type == "boolean":
        annotation = StrictBool
    else:
        annotation = list[StrictStr]
    return annotation | None if field.nullable else annotation


def _record_prompt(context_text: str, instructions: str) -> str:
    return f"""Complete one structured record from this context.

Assignment:
{instructions}

Context:
{context_text}

Answer every requested question through its corresponding record field. All
answers must describe the same row and agree with each other.
"""


def _schema_prompt(questions: Mapping[str, str], instructions: str) -> str:
    return f"""Design one shared row schema for this assignment.

Assignment:
{instructions}

Questions and fixed output field names:
{json.dumps(questions, ensure_ascii=False, indent=2)}

Return every supplied field name exactly once. Choose from string, integer,
number, boolean, or string_list. Mark a value nullable only when the assignment
allows a missing or unknown answer. The format instruction should describe
representation only and must not add new task requirements.
"""


def _validation_errors(error: ValidationError) -> tuple[str, ...]:
    return tuple(
        f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
        for item in error.errors()
    )


# 2) Schema and record wrappers
def _infer_schema_plan(
    questions: Mapping[str, str],
    instructions: str,
    client: _OpenAIClient,
) -> _SchemaPlan:
    system_prompt = """You design one reusable flat data-row schema.
The caller owns the field names, questions, and assignment. Keep every field
name exactly as supplied. Choose the simplest useful JSON value type and a
concise output-format instruction for each field without changing its meaning.
"""
    max_attempts = 2
    original_messages = build_messages(
        _schema_prompt(questions, instructions),
        system_prompt,
    )
    messages = original_messages
    final_errors: tuple[str, ...] = ()

    for attempt_number in range(1, max_attempts + 1):
        raw_response = ""
        try:
            response = client.parse_response(
                input=messages,
                text_format=_SchemaPlan,
                reasoning={"effort": "none"},
            )
            raw_response = response_text(response)
            plan = _SchemaPlan.model_validate_json(raw_response)
        except ValidationError as error:
            final_errors = _validation_errors(error)
        else:
            final_errors = _schema_plan_errors(plan, questions)
            if not final_errors:
                return plan

        if attempt_number == max_attempts:
            break
        correction = (
            "The schema plan failed local validation. Return a corrected plan "
            "with each supplied field name exactly once.\n\nValidation errors:\n"
            + "\n".join(f"- {error}" for error in final_errors)
        )
        messages = [*original_messages]
        if raw_response:
            messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": correction})

    raise ValueError(
        "could not infer a valid shared record schema: " + "; ".join(final_errors)
    )


def _generate_one(
    input_id: str,
    context_text: str,
    output_model: type[BaseModel],
    instructions: str,
    client: _OpenAIClient,
    caller: _AdaptiveProviderCaller | None = None,
) -> dict[str, object]:
    system_prompt = """You complete one typed record for a caller-provided subject.
Follow the caller's assignment and make every field describe the same subject.
Make related fields mutually consistent and return only the requested format.
"""
    max_attempts = 2
    original_messages = build_messages(
        _record_prompt(context_text, instructions),
        system_prompt,
    )
    messages = original_messages
    attempts: list[_RecordAttempt] = []

    for attempt_number in range(1, max_attempts + 1):
        raw_response = ""
        try:
            request = {
                "input": messages,
                "text_format": output_model,
                "reasoning": {"effort": "none"},
            }
            response = (
                client.parse_response(**request)
                if caller is None
                else caller.call(client.parse_response, **request)
            )
            raw_response = response_text(response)
            record = output_model.model_validate_json(raw_response)
        except ValidationError as error:
            errors = _validation_errors(error)
        else:
            attempts.append(_RecordAttempt(raw_response=raw_response, errors=()))
            record_data = record.model_dump(mode="json", by_alias=True)
            return record_data

        attempts.append(_RecordAttempt(raw_response=raw_response, errors=errors))
        if attempt_number == max_attempts:
            break

        correction = (
            "The record failed local validation. Return a corrected record "
            "for the same subject. Keep all fields mutually consistent and satisfy "
            "the exact requested question fields.\n\nValidation errors:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
        messages = [*original_messages]
        if raw_response:
            messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": correction})

    raise StructuredRecordError(input_id, attempts)


# 3) Main wrapper function
def generate_structured_records(
    context: Mapping[str, str],
    questions: Mapping[str, str],
    instructions: str,
    llm: Mapping[str, object],
    batch: bool | Mapping[str, object] = False,
) -> dict[str, dict[str, object]]:
    """Answer one shared question set for every stable context ID.

    ``context`` maps each record's stable ID to the text the model should use.
    One setup call infers a flat schema from ``questions`` and ``instructions``.
    The resulting Pydantic model is reused for every record, with one correction
    allowed for an invalid schema plan and one for each invalid result.
    """
    context = _validate_context(context)
    questions = _validate_questions(questions)
    instructions = require_text(instructions, "instructions")
    batch_config = _batch_config(batch)

    schema_client = resolve_openai_client(llm, required_response_methods=("parse",))
    schema_plan = _infer_schema_plan(questions, instructions, schema_client)
    output_model = _output_model(questions, schema_plan)
    record_client = schema_client
    if batch_config is not None:
        record_client = resolve_openai_client(
            llm,
            disable_sdk_retries=True,
            required_response_methods=("parse",),
        )
    jobs = list(context.items())

    if batch_config is None:
        generated = [
            _generate_one(
                input_id,
                context_text,
                output_model,
                instructions,
                record_client,
            )
            for input_id, context_text in jobs
        ]
    else:
        def generate_one(
            job: tuple[str, str],
            caller: _AdaptiveProviderCaller,
        ) -> dict[str, object]:
            input_id, context_text = job
            generated_record = _generate_one(
                input_id,
                context_text,
                output_model,
                instructions,
                record_client,
                caller,
            )
            return generated_record

        generated = _run_adaptive_operations(
            jobs,
            generate_one,
            model=record_client.model,
            scope="generate-structured-records",
            config=batch_config,
        )

    records_by_id = dict(zip(context, generated, strict=True))
    return records_by_id
