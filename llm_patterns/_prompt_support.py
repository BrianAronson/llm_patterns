"""Internal role/content message construction."""

from collections.abc import Sequence
from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


def require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value.strip()


def build_messages(
    user_prompt: str,
    system_prompt: str | None = None,
    prior_messages: Sequence[Message] = (),
) -> list[Message]:
    """Build provider-neutral role/content messages around a caller prompt."""
    if not user_prompt.strip():
        raise ValueError("user_prompt must not be blank")

    messages: list[Message] = []
    if system_prompt is not None:
        if not system_prompt.strip():
            raise ValueError("system_prompt must be nonblank when supplied")
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(prior_messages)
    messages.append({"role": "user", "content": user_prompt})
    return messages
