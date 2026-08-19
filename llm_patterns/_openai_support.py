"""Small internal boundary around the OpenAI client used by the helpers."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class _OpenAIClient:
    service: Any
    model: str

    def create_response(self, **request: object) -> object:
        return self.service.responses.create(model=self.model, **request)

    def parse_response(self, **request: object) -> object:
        return self.service.responses.parse(model=self.model, **request)


def resolve_openai_client(
    llm: Mapping[str, object],
    disable_sdk_retries: bool = False,
    required_response_methods: Sequence[str] = ("create", "parse"),
) -> _OpenAIClient:
    if not isinstance(llm, Mapping):
        raise TypeError("llm must be a mapping with 'service' and 'model'")
    if set(llm) != {"service", "model"}:
        raise ValueError("llm must contain exactly 'service' and 'model'")

    service = llm["service"]
    model = llm["model"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError("llm['model'] must be a nonblank string")
    if disable_sdk_retries:
        with_options = getattr(service, "with_options", None)
        if callable(with_options):
            service = with_options(max_retries=0)
    responses = getattr(service, "responses", None)
    missing_methods = [
        method
        for method in required_response_methods
        if not callable(getattr(responses, method, None))
    ]
    if missing_methods:
        required = " and ".join(
            f"responses.{method}(...)" for method in missing_methods
        )
        raise TypeError(f"llm['service'] must provide {required}")
    return _OpenAIClient(service=service, model=model.strip())


def response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    output = getattr(response, "output", None)
    chunks: list[str] = []
    if isinstance(output, list):
        for item in output:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
    if chunks:
        return "".join(chunks)
    raise TypeError("llm service responses must return response text")
