"""Run complete independent prompts with adaptive local concurrency."""

# 0) Imports
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from ._adaptive_openai import (
    _AdaptiveProviderCaller,
    _BatchConfig,
    _run_adaptive_operations,
    _validated_batch_config,
)
from ._openai_support import _OpenAIClient, resolve_openai_client, response_text

__all__ = ["run_prompt_batch"]

_DURABLE_SCHEMA = "local-prompt-batch-v1"

# 1) Sub functions
def _request_prompt(
    prompt: str,
    client: _OpenAIClient,
    caller: _AdaptiveProviderCaller,
) -> str:
    response = caller.call(
        client.create_response,
        input=prompt,
        reasoning={"effort": "none"},
    )
    completed_text = response_text(response)
    return completed_text


def _validate_prompts(prompts: Sequence[str]) -> list[str]:
    if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a sequence of complete prompt strings")
    normalized: list[str] = []
    for prompt in prompts:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("every prompt must be a nonblank string")
        normalized.append(prompt)
    return normalized


def _validate_durable_prompts(
    prompts: Sequence[str] | Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(prompts, Mapping):
        raise TypeError(
            "run_directory requires prompts as a mapping of stable input IDs to text"
        )
    normalized: dict[str, str] = {}
    for input_id, prompt in prompts.items():
        if not isinstance(input_id, str) or not input_id.strip():
            raise ValueError("durable input IDs must be nonblank strings")
        if input_id != input_id.strip():
            raise ValueError("durable input IDs must not have surrounding whitespace")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("every durable prompt must be a nonblank string")
        normalized[input_id] = prompt
    return normalized


def _prepare_run_directory(run_directory: str | Path) -> Path:
    if isinstance(run_directory, str):
        if not run_directory.strip():
            raise ValueError("run_directory must not be blank")
        root = Path(run_directory)
    elif isinstance(run_directory, Path):
        root = run_directory
    else:
        raise TypeError("run_directory must be a string or Path")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _durable_manifest(
    prompts: Mapping[str, str],
    model: str,
    max_retries: int,
) -> dict[str, object]:
    return {
        "schema": _DURABLE_SCHEMA,
        "model": model,
        "reasoning": {"effort": "none"},
        "max_retries": max_retries,
        "prompts": [
            {
                "input_id": input_id,
                "prompt_sha256": _sha256(prompt),
                "result_path": f"results/{index:06d}.json",
            }
            for index, (input_id, prompt) in enumerate(prompts.items())
        ],
    }


def _load_durable_results(
    root: Path,
    manifest: Mapping[str, object],
) -> dict[int, str]:
    records = manifest["prompts"]
    if not isinstance(records, list):
        raise ValueError("run_manifest.json has an invalid prompts list")
    completed: dict[int, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("run_manifest.json has an invalid prompt record")
        result_path = record.get("result_path")
        if not isinstance(result_path, str):
            raise ValueError("run_manifest.json has an invalid result path")
        path = root / result_path
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not read durable result {path}") from error
        if not isinstance(payload, Mapping):
            raise ValueError(f"Durable result {path} is not an object")
        if (
            payload.get("input_id") != record.get("input_id")
            or payload.get("prompt_sha256") != record.get("prompt_sha256")
            or not isinstance(payload.get("response_text"), str)
        ):
            raise ValueError(f"Durable result conflicts with its prompt: {path}")
        completed[index] = payload["response_text"]
    return completed


def _persist_durable_result(
    path: Path,
    input_id: str,
    prompt_sha256: str,
    response_text: str,
) -> None:
    payload = {
        "schema": _DURABLE_SCHEMA,
        "input_id": input_id,
        "prompt_sha256": prompt_sha256,
        "response_text": response_text,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Durable result conflicts with existing file: {path}")
        return
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    _atomic_text_write(path, serialized)


def _write_or_verify_json(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not read existing durable manifest: {path}") from error
        if existing != payload:
            raise ValueError(
                "Existing durable run conflicts with the supplied prompts or policy"
            )
        return
    _atomic_text_write(path, serialized)


def _atomic_text_write(path: Path, text: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 2) Durable-run wrapper
def _run_durable_prompt_batch(
    prompts: dict[str, str],
    client: _OpenAIClient,
    config: _BatchConfig,
    run_directory: str | Path,
) -> dict[str, str]:
    root = _prepare_run_directory(run_directory)
    manifest = _durable_manifest(
        prompts, model=client.model, max_retries=config.max_retries
    )
    _write_or_verify_json(root / "run_manifest.json", manifest)

    completed_by_index = _load_durable_results(root, manifest)
    pending_jobs = [
        (index, input_id, prompt)
        for index, (input_id, prompt) in enumerate(prompts.items())
        if index not in completed_by_index
    ]
    if pending_jobs:
        pending_prompts = [prompt for _, _, prompt in pending_jobs]

        def run_one(prompt: str, caller: _AdaptiveProviderCaller) -> str:
            completed_prompt = _request_prompt(prompt, client, caller)
            return completed_prompt

        def persist_one(pending_index: int, completed_text: str) -> None:
            index, input_id, _prompt = pending_jobs[pending_index]
            prompt_record = manifest["prompts"][index]
            _persist_durable_result(
                root / prompt_record["result_path"],
                input_id=input_id,
                prompt_sha256=prompt_record["prompt_sha256"],
                response_text=completed_text,
            )

        durable_config = (
            config
            if config.state_path is not None
            else replace(config, state_path=root / "scheduler_state.json")
        )
        _run_adaptive_operations(
            pending_prompts,
            run_one,
            model=client.model,
            scope="run-prompt-batch",
            config=durable_config,
            on_result=persist_one,
        )

    completed_by_index = _load_durable_results(root, manifest)
    if len(completed_by_index) != len(prompts):
        raise RuntimeError("Durable prompt batch did not produce every result")
    responses_by_id = {
        input_id: completed_by_index[index] for index, input_id in enumerate(prompts)
    }
    return responses_by_id


# 3) Main wrapper function
def run_prompt_batch(
    prompts: Sequence[str] | Mapping[str, str],
    llm: Mapping[str, object],
    max_concurrency: int,
    max_retries: int,
    state_path: str | Path | None = None,
    run_directory: str | Path | None = None,
) -> list[str] | dict[str, str]:
    """Run independent prompts and return their response text.

    Concurrency starts conservatively and adapts within ``max_concurrency``.
    Rate-limit responses reduce both concurrency and launch frequency, pause all
    new calls for the provider's requested interval when available, and retry
    only bounded transient failures. When ``state_path`` is supplied, learned
    scheduling hints are saved there without prompts, responses, or credentials.
    Client-side retries are bounded but cannot guarantee exactly-once execution.

    With ``run_directory=None``, ``prompts`` is a sequence and the result is a
    list in prompt order. With ``run_directory`` supplied, ``prompts`` must be a
    mapping of stable input IDs to complete prompts. The manifest is written
    before any provider call, each successful response is persisted immediately,
    and a later invocation resumes only missing IDs. Changed prompts, IDs, model,
    reasoning settings, or retry policy refuse to reuse the directory. The local
    directory contains response text, so callers should choose its location with
    the sensitivity of their prompts and outputs in mind.
    """
    config = _validated_batch_config(
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        state_path=state_path,
    )
    client = resolve_openai_client(
        llm, disable_sdk_retries=True, required_response_methods=("create",)
    )
    if run_directory is not None:
        prompts_by_id = _validate_durable_prompts(prompts)
        responses_by_id = _run_durable_prompt_batch(
            prompts_by_id,
            client,
            config,
            run_directory,
        )
        return responses_by_id

    prompts = _validate_prompts(prompts)
    if not prompts:
        return []

    def run_one(prompt: str, caller: _AdaptiveProviderCaller) -> str:
        completed_prompt = _request_prompt(prompt, client, caller)
        return completed_prompt

    responses = _run_adaptive_operations(
        prompts,
        run_one,
        model=client.model,
        scope="run-prompt-batch",
        config=config,
    )
    return responses
