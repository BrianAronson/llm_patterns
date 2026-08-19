"""Submit and collect durable OpenAI Responses Batch jobs.

The run directory records the exact request identity, chunks, uploads,
submission intents, receipts, downloads, and reconciled results. Reruns can
reuse those artifacts and avoid blindly repeating an ambiguous Batch creation.

This module handles only ``POST /v1/responses`` batches. Use
``run_prompt_batch(...)`` when the local Python process should schedule calls.
"""

# 0) Imports
from __future__ import annotations
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from ._openai_support import resolve_openai_client
from ._prompt_support import build_messages

__all__ = [
    "AmbiguousBatchSubmissionError",
    "BatchPromptResult",
    "OpenAIPromptBatchCollection",
    "OpenAIPromptBatchSubmission",
    "collect_openai_prompt_batch",
    "submit_openai_prompt_batch",
]

_OPENAI_BATCH_ENDPOINT = "/v1/responses"
_OPENAI_BATCH_COMPLETION_WINDOW = "24h"
_OPENAI_BATCH_MAX_REQUESTS = 50_000
_OPENAI_BATCH_MAX_INPUT_BYTES = 200_000_000
_DEFAULT_MAX_INPUT_BYTES = 195_000_000

_PromptOutcome = Literal["succeeded", "failed", "missing", "pending"]
_CollectionStatus = Literal["pending", "completed", "completed_with_failures"]


# Public results
class AmbiguousBatchSubmissionError(RuntimeError):
    """A create call may have reached OpenAI, but no local receipt was saved."""


@dataclass(frozen=True)
class OpenAIPromptBatchSubmission:
    """Provider Batch IDs aligned with the deterministic local chunks."""

    run_directory: Path
    request_count: int
    chunk_count: int
    provider_batch_ids: tuple[str, ...]
    new_upload_count: int
    new_submission_count: int


@dataclass(frozen=True)
class BatchPromptResult:
    """One original prompt aligned with its provider response or failure."""

    request_index: int
    prompt_id: str
    prompt_sha256: str
    outcome: Literal["succeeded", "failed", "missing", "pending"]
    status_code: int | None
    response_body: Mapping[str, Any] | None
    error: Mapping[str, Any] | None
    provider_batch_id: str | None


@dataclass(frozen=True)
class OpenAIPromptBatchCollection:
    """Current provider status plus every result in original prompt order."""

    run_directory: Path
    status: Literal["pending", "completed", "completed_with_failures"]
    chunk_statuses: tuple[str, ...]
    results: tuple[BatchPromptResult, ...]
    checked_batch_count: int
    downloaded_file_count: int

    @property
    def succeeded_count(self) -> int:
        return sum(result.outcome == "succeeded" for result in self.results)

    @property
    def failed_count(self) -> int:
        return sum(result.outcome in {"failed", "missing"} for result in self.results)

    def response_texts(self) -> dict[str, str]:
        """Return Responses API text under stable IDs, only for a clean batch."""
        incomplete = [
            result.prompt_id for result in self.results if result.outcome != "succeeded"
        ]
        if incomplete:
            raise ValueError(
                "Cannot return response texts while requests are incomplete or failed: "
                + ", ".join(incomplete)
            )
        return {
            result.prompt_id: _responses_output_text(
                cast(Mapping[str, Any], result.response_body)
            )
            for result in self.results
        }


# 1) Sub functions
# Local lifecycle records
@dataclass(frozen=True)
class _PreparedOpenAIPromptBatch:
    run_directory: Path
    request_count: int
    chunk_count: int
    batch_identity_sha256: str


@dataclass(frozen=True)
class _LoadedChunk:
    chunk_index: int
    directory: Path
    manifest: Mapping[str, Any]
    input_rows: tuple[Mapping[str, Any], ...]
    requests: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _LoadedBatch:
    run_directory: Path
    manifest: Mapping[str, Any]
    chunks: tuple[_LoadedChunk, ...]


@dataclass(frozen=True)
class _SubmittedChunk:
    provider_batch_id: str
    uploaded: bool
    submitted: bool


@dataclass(frozen=True)
class _CollectedChunk:
    provider_status: str
    results: tuple[BatchPromptResult, ...]
    checked_batch_count: int
    downloaded_file_count: int


@dataclass(frozen=True)
class _ValidatedBatchInputs:
    prompts: dict[str, str]
    model: str
    settings: dict[str, Any]
    system_prompt: str | None
    metadata: dict[str, str]


# Validate and load local request artifacts
def _validate_prepare_inputs(
    prompts: Mapping[str, str],
    model: str,
    settings: Mapping[str, Any] | None,
    system_prompt: str | None,
    metadata: Mapping[str, str] | None,
    max_requests_per_batch: int,
    max_input_bytes: int,
) -> _ValidatedBatchInputs:
    default_response_settings = {"reasoning": {"effort": "none"}}
    if not isinstance(prompts, Mapping):
        raise TypeError("prompts must be a mapping of stable IDs to complete prompts")
    if not prompts:
        raise ValueError("prompts must contain at least one complete prompt")
    prompt_values: dict[str, str] = {}
    for prompt_id, prompt in prompts.items():
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError("prompt IDs must be nonblank strings")
        if prompt_id != prompt_id.strip():
            raise ValueError("prompt IDs must not have surrounding whitespace")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Every prompt must be a nonblank string")
        prompt_values[prompt_id] = prompt
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must not be blank")
    if system_prompt is not None and (
        not isinstance(system_prompt, str) or not system_prompt.strip()
    ):
        raise ValueError("system_prompt must be a nonblank string when supplied")
    if not 1 <= max_requests_per_batch <= _OPENAI_BATCH_MAX_REQUESTS:
        raise ValueError(
            f"max_requests_per_batch must be within 1..{_OPENAI_BATCH_MAX_REQUESTS}"
        )
    if not 1 <= max_input_bytes <= _OPENAI_BATCH_MAX_INPUT_BYTES:
        raise ValueError(
            f"max_input_bytes must be within 1..{_OPENAI_BATCH_MAX_INPUT_BYTES}"
        )
    if settings is not None and not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping when supplied")
    normalized_settings = _canonical_json_object(
        {**default_response_settings, **dict(settings or {})}, "settings"
    )
    reserved_settings = sorted(set(normalized_settings) & {"model", "input"})
    if reserved_settings:
        raise ValueError(
            "OpenAI Batch settings cannot override: " + ", ".join(reserved_settings)
        )
    if "stream" in normalized_settings:
        raise ValueError("OpenAI Batch settings must not include stream")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping when supplied")
    normalized_metadata = {
        str(key): str(value) for key, value in dict(metadata or {}).items()
    }
    if any(
        not key.strip() or not value.strip()
        for key, value in normalized_metadata.items()
    ):
        raise ValueError("OpenAI Batch metadata keys and values must be nonblank")
    reserved_metadata = sorted(
        set(normalized_metadata)
        & {
            "llm_patterns_batch_sha256",
            "llm_patterns_chunk_index",
            "llm_patterns_chunk_count",
            "llm_patterns_input_sha256",
        }
    )
    if reserved_metadata:
        raise ValueError(
            "Reserved OpenAI Batch metadata cannot be overridden: "
            + ", ".join(reserved_metadata)
        )
    if len(normalized_metadata) > 12:
        raise ValueError(
            "metadata may contain at most 12 caller pairs because the helper adds "
            "four reserved reconciliation pairs"
        )
    if any(len(key) > 64 for key in normalized_metadata):
        raise ValueError("OpenAI Batch metadata keys may contain at most 64 characters")
    if any(len(value) > 512 for value in normalized_metadata.values()):
        raise ValueError(
            "OpenAI Batch metadata values may contain at most 512 characters"
        )
    return _ValidatedBatchInputs(
        prompts=prompt_values,
        model=model.strip(),
        settings=normalized_settings,
        system_prompt=system_prompt,
        metadata=normalized_metadata,
    )


def _build_request_rows(
    prompts: Mapping[str, str],
    model: str,
    settings: Mapping[str, Any],
    system_prompt: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[bytes]]:
    prompt_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    input_lines: list[bytes] = []
    for request_index, (prompt_id, prompt) in enumerate(prompts.items()):
        body = {
            "model": model,
            "input": build_messages(prompt, system_prompt),
            **settings,
        }
        input_row = {
            "custom_id": prompt_id,
            "method": "POST",
            "url": _OPENAI_BATCH_ENDPOINT,
            "body": body,
        }
        input_line = _canonical_jsonl_bytes([input_row])
        prompt_rows.append(
            {
                "request_index": request_index,
                "custom_id": prompt_id,
                "prompt": prompt,
                "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                "request_sha256": _sha256_bytes(input_line),
            }
        )
        input_rows.append(input_row)
        input_lines.append(input_line)
    return prompt_rows, input_rows, input_lines


def _chunk_request_rows(
    prompt_rows: Sequence[dict[str, Any]],
    input_rows: Sequence[dict[str, Any]],
    input_lines: Sequence[bytes],
    maximum_requests: int,
    maximum_bytes: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current_prompt_rows: list[dict[str, Any]] = []
    current_input_rows: list[dict[str, Any]] = []
    current_input_lines: list[bytes] = []
    current_bytes = 0
    for prompt_row, input_row, input_line in zip(
        prompt_rows, input_rows, input_lines, strict=True
    ):
        if len(input_line) > maximum_bytes:
            raise ValueError(
                f"Request {input_row['custom_id']} exceeds max_input_bytes by itself"
            )
        should_close = bool(current_input_rows) and (
            len(current_input_rows) >= maximum_requests
            or current_bytes + len(input_line) > maximum_bytes
        )
        if should_close:
            chunks.append(
                {
                    "prompt_rows": current_prompt_rows,
                    "input_rows": current_input_rows,
                    "input_lines": current_input_lines,
                }
            )
            current_prompt_rows = []
            current_input_rows = []
            current_input_lines = []
            current_bytes = 0
        current_prompt_rows.append(prompt_row)
        current_input_rows.append(input_row)
        current_input_lines.append(input_line)
        current_bytes += len(input_line)
    if current_input_rows:
        chunks.append(
            {
                "prompt_rows": current_prompt_rows,
                "input_rows": current_input_rows,
                "input_lines": current_input_lines,
            }
        )
    return chunks


def _load_and_verify_batch(run_directory: Path) -> _LoadedBatch:
    root = run_directory.resolve()
    manifest_path = root / "batch_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"OpenAI Batch manifest does not exist: {manifest_path}"
        )
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("OpenAI Batch manifest has an unsupported schema version")
    if (
        manifest.get("provider") != "openai"
        or manifest.get("endpoint") != _OPENAI_BATCH_ENDPOINT
    ):
        raise ValueError("OpenAI Batch manifest has the wrong provider or endpoint")
    raw_chunks = manifest.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("OpenAI Batch manifest must contain chunks")
    if int(manifest.get("chunk_count", -1)) != len(raw_chunks):
        raise ValueError("OpenAI Batch manifest chunk count is inconsistent")

    loaded_chunks: list[_LoadedChunk] = []
    all_request_indexes: list[int] = []
    all_custom_ids: list[str] = []
    logical_input = bytearray()
    all_prompts: list[str] = []
    for expected_index, raw_chunk in enumerate(raw_chunks, start=1):
        if (
            not isinstance(raw_chunk, dict)
            or int(raw_chunk.get("chunk_index", -1)) != expected_index
        ):
            raise ValueError("OpenAI Batch chunk manifests are out of order")
        input_path = _safe_artifact_path(root, str(raw_chunk.get("input_path", "")))
        request_path = _safe_artifact_path(
            root, str(raw_chunk.get("requests_path", ""))
        )
        input_payload = input_path.read_bytes()
        request_payload = request_path.read_bytes()
        if _sha256_bytes(input_payload) != str(raw_chunk.get("input_sha256")):
            raise ValueError(f"OpenAI Batch chunk {expected_index} input hash changed")
        if _sha256_bytes(request_payload) != str(raw_chunk.get("requests_sha256")):
            raise ValueError(
                f"OpenAI Batch chunk {expected_index} request map hash changed"
            )
        if len(input_payload) != int(raw_chunk.get("input_bytes", -1)):
            raise ValueError(f"OpenAI Batch chunk {expected_index} byte count changed")
        input_rows = tuple(_read_jsonl_bytes(input_payload, input_path))
        requests = tuple(_read_jsonl_bytes(request_payload, request_path))
        if len(input_rows) != len(requests) or len(input_rows) != int(
            raw_chunk.get("request_count", -1)
        ):
            raise ValueError(
                f"OpenAI Batch chunk {expected_index} request counts changed"
            )
        for input_row, request in zip(input_rows, requests, strict=True):
            custom_id = str(input_row.get("custom_id", ""))
            if custom_id != str(request.get("custom_id", "")) or not custom_id:
                raise ValueError("OpenAI Batch input and request-map IDs do not align")
            if (
                input_row.get("method") != "POST"
                or input_row.get("url") != _OPENAI_BATCH_ENDPOINT
            ):
                raise ValueError(
                    "OpenAI Batch input row has the wrong method or endpoint"
                )
            body = input_row.get("body")
            if not isinstance(body, dict) or body.get("model") != manifest.get("model"):
                raise ValueError("OpenAI Batch input row has the wrong model")
            request_index = int(request.get("request_index", -1))
            prompt = str(request.get("prompt", ""))
            if _sha256_bytes(prompt.encode("utf-8")) != str(
                request.get("prompt_sha256", "")
            ):
                raise ValueError("OpenAI Batch request prompt hash changed")
            if _sha256_bytes(_canonical_jsonl_bytes([input_row])) != str(
                request.get("request_sha256", "")
            ):
                raise ValueError("OpenAI Batch request body hash changed")
            all_request_indexes.append(request_index)
            all_custom_ids.append(custom_id)
            all_prompts.append(prompt)
        logical_input.extend(input_payload)
        loaded_chunks.append(
            _LoadedChunk(
                chunk_index=expected_index,
                directory=input_path.parent,
                manifest=raw_chunk,
                input_rows=input_rows,
                requests=requests,
            )
        )

    request_count = int(manifest.get("request_count", -1))
    if all_request_indexes != list(range(request_count)):
        raise ValueError(
            "OpenAI Batch request indexes do not cover original prompt order"
        )
    if len(set(all_custom_ids)) != len(all_custom_ids):
        raise ValueError("OpenAI Batch custom IDs are not globally unique")
    if _sha256_bytes(bytes(logical_input)) != str(manifest.get("logical_input_sha256")):
        raise ValueError("OpenAI Batch logical input hash changed")
    identity = {
        "endpoint": _OPENAI_BATCH_ENDPOINT,
        "model": manifest.get("model"),
        "settings": manifest.get("settings"),
        "system_prompt": manifest.get("system_prompt"),
        "metadata": manifest.get("metadata"),
        "max_requests_per_batch": manifest.get("max_requests_per_batch"),
        "max_input_bytes": manifest.get("max_input_bytes"),
        "prompts": [
            {"prompt_id": prompt_id, "prompt": prompt}
            for prompt_id, prompt in zip(all_custom_ids, all_prompts, strict=True)
        ],
    }
    if _sha256_bytes(_canonical_json_bytes(identity)) != str(
        manifest.get("batch_identity_sha256")
    ):
        raise ValueError("OpenAI Batch identity hash changed")
    return _LoadedBatch(root, manifest, tuple(loaded_chunks))


# Provider receipts and downloads
def _provider_metadata(loaded: _LoadedBatch, chunk: _LoadedChunk) -> dict[str, str]:
    user_metadata = dict(cast(Mapping[str, str], loaded.manifest.get("metadata", {})))
    return {
        **user_metadata,
        "llm_patterns_batch_sha256": str(loaded.manifest["batch_identity_sha256"]),
        "llm_patterns_chunk_index": str(chunk.chunk_index),
        "llm_patterns_chunk_count": str(loaded.manifest["chunk_count"]),
        "llm_patterns_input_sha256": str(chunk.manifest["input_sha256"]),
    }


def _recover_unrecorded_batch(client: Any, intent: Mapping[str, Any]) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    listed = client.batches.list(limit=100)
    iter_pages = getattr(listed, "iter_pages", None)
    pages = iter_pages() if callable(iter_pages) else (listed,)
    for page in pages:
        items = getattr(page, "data", page)
        for item in items:
            record = _batch_record(item)
            if str(record.get("input_file_id")) != str(
                intent["provider_input_file_id"]
            ):
                continue
            if str(record.get("endpoint")) != str(intent["endpoint"]):
                continue
            if record.get("metadata") != intent.get("metadata"):
                continue
            matches.append(record)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("Multiple OpenAI Batches match one saved submission intent")
    raise AmbiguousBatchSubmissionError(
        "A saved OpenAI Batch submission intent has no local receipt and no exact "
        "provider Batch was found in the current listing. The helper will not "
        "resubmit automatically because the earlier create call may be ambiguous."
    )


def _verify_upload_receipt(upload: Mapping[str, Any], chunk: _LoadedChunk) -> None:
    if upload.get("schema_version") != 1:
        raise ValueError("OpenAI Batch upload receipt has an unsupported schema")
    if str(upload.get("input_sha256")) != str(chunk.manifest["input_sha256"]):
        raise ValueError("OpenAI Batch upload receipt input hash mismatch")
    if not str(upload.get("provider_input_file_id", "")).strip():
        raise ValueError("OpenAI Batch upload receipt has no provider file ID")


def _verify_submission_receipt(
    receipt: Mapping[str, Any], loaded: _LoadedBatch, chunk: _LoadedChunk
) -> None:
    if receipt.get("schema_version") != 1:
        raise ValueError("OpenAI Batch submission receipt has an unsupported schema")
    expected = {
        "batch_identity_sha256": loaded.manifest["batch_identity_sha256"],
        "chunk_index": chunk.chunk_index,
        "input_sha256": chunk.manifest["input_sha256"],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"OpenAI Batch submission receipt {key} mismatch")
    if not str(receipt.get("provider_input_file_id", "")).strip():
        raise ValueError("OpenAI Batch submission receipt has no input file ID")
    if not str(receipt.get("provider_batch_id", "")).strip():
        raise ValueError("OpenAI Batch submission receipt has no Batch ID")


def _load_all_submission_receipts(
    loaded: _LoadedBatch,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for chunk in loaded.chunks:
        path = chunk.directory / "submission_receipt.json"
        if not path.is_file():
            raise ValueError(
                f"OpenAI Batch chunk {chunk.chunk_index} is not submitted; rerun "
                "submit_openai_prompt_batch with the exact original inputs"
            )
        receipt = _load_json_object(path)
        _verify_submission_receipt(receipt, loaded, chunk)
        receipts.append(receipt)
    return receipts


def _append_status_event(path: Path, batch_id: str, status: Mapping[str, Any]) -> None:
    event = {
        "checked_at": _utc_now(),
        "provider_batch_id": batch_id,
        "status": status.get("status"),
        "request_counts": status.get("request_counts"),
        "output_file_id": status.get("output_file_id"),
        "error_file_id": status.get("error_file_id"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as output:
        output.write(_canonical_jsonl_bytes([event]))
        output.flush()
        os.fsync(output.fileno())


def _download_terminal_file(
    client: Any,
    chunk: _LoadedChunk,
    raw_file_id: Any,
    artifact_name: str,
    receipt_name: str,
) -> tuple[Path | None, bool]:
    if raw_file_id is None or not str(raw_file_id).strip():
        return None, False
    file_id = str(raw_file_id)
    artifact_path = chunk.directory / artifact_name
    receipt_path = chunk.directory / receipt_name
    expected_bytes = _provider_file_size(client, file_id)
    if artifact_path.exists():
        payload = artifact_path.read_bytes()
        _validate_downloaded_jsonl(payload, artifact_path, expected_bytes)
        digest = _sha256_bytes(payload)
        receipt = {
            "schema_version": 1,
            "provider_file_id": file_id,
            "artifact": artifact_name,
            "sha256": digest,
            "bytes": artifact_path.stat().st_size,
        }
        _write_immutable(receipt_path, _pretty_json_bytes(receipt))
        return artifact_path, False
    if receipt_path.exists():
        raise ValueError(
            f"OpenAI Batch download receipt exists without {artifact_name}"
        )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=artifact_path.parent,
            prefix=f".{artifact_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        streaming_files = getattr(client.files, "with_streaming_response", None)
        if streaming_files is not None:
            with streaming_files.content(file_id) as response:
                stream_to_file = getattr(response, "stream_to_file", None)
                if callable(stream_to_file):
                    stream_to_file(temporary_path)
                else:
                    # Compatibility with binary wrappers that expose a
                    # whole-response writer instead of the streamed name.
                    write_to_file = getattr(response, "write_to_file", None)
                    if not callable(write_to_file):
                        raise TypeError(
                            "Streaming OpenAI file response has no file writer"
                        )
                    write_to_file(temporary_path)
        else:
            response = client.files.content(file_id)
            payload = _file_response_bytes(response)
            with temporary_path.open("wb") as temporary_file:
                temporary_file.write(payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        with temporary_path.open("ab") as downloaded_file:
            downloaded_file.flush()
            os.fsync(downloaded_file.fileno())
        payload = temporary_path.read_bytes()
        _validate_downloaded_jsonl(payload, artifact_path, expected_bytes)
        _publish_temporary_immutable(artifact_path, temporary_path, payload)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    receipt = {
        "schema_version": 1,
        "provider_file_id": file_id,
        "artifact": artifact_name,
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
    }
    _write_immutable(receipt_path, _pretty_json_bytes(receipt))
    return artifact_path, True


def _provider_file_size(client: Any, file_id: str) -> int | None:
    record = client.files.retrieve(file_id)
    value = (
        record.get("bytes")
        if isinstance(record, dict)
        else getattr(record, "bytes", None)
    )
    return value if isinstance(value, int) and value >= 0 else None


def _validate_downloaded_jsonl(
    payload: bytes,
    artifact_path: Path,
    expected_bytes: int | None,
) -> None:
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise ValueError(
            f"Downloaded OpenAI Batch file has {len(payload)} bytes; expected "
            f"{expected_bytes}: {artifact_path}"
        )
    _read_jsonl_bytes(payload, artifact_path)


def _file_response_bytes(response: Any) -> bytes:
    read = getattr(response, "read", None)
    if callable(read):
        payload = read()
    else:
        text_value = getattr(response, "text", None)
        payload = text_value() if callable(text_value) else text_value
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if isinstance(payload, bytes):
        return payload
    raise TypeError("OpenAI Batch file content must be bytes or text")


# Reconcile provider output
def _reconcile_terminal_chunk(
    loaded: _LoadedBatch,
    chunk: _LoadedChunk,
    receipt: Mapping[str, Any],
    provider_status: str,
    output_path: Path | None,
    error_path: Path | None,
) -> tuple[list[BatchPromptResult], dict[str, Any]]:
    output_rows = _read_jsonl_file(output_path) if output_path is not None else []
    error_rows = _read_jsonl_file(error_path) if error_path is not None else []
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    row_sources: dict[str, str] = {}
    for source, rows in (("output", output_rows), ("error", error_rows)):
        for row in rows:
            custom_id = str(row.get("custom_id", "")).strip()
            if not custom_id:
                raise ValueError(f"OpenAI Batch {source} row has a blank custom_id")
            if custom_id in rows_by_id:
                raise ValueError(
                    f"OpenAI Batch custom_id {custom_id!r} appears more than once"
                )
            rows_by_id[custom_id] = row
            row_sources[custom_id] = source
    expected_ids = {str(request["custom_id"]) for request in chunk.requests}
    unknown = sorted(set(rows_by_id) - expected_ids)
    if unknown:
        raise ValueError(
            "OpenAI Batch files contain unknown custom IDs: " + ", ".join(unknown)
        )

    batch_id = str(receipt["provider_batch_id"])
    results: list[BatchPromptResult] = []
    for request in chunk.requests:
        custom_id = str(request["custom_id"])
        row = rows_by_id.get(custom_id)
        if row is None:
            results.append(
                _result_from_request(
                    request,
                    outcome="missing",
                    provider_batch_id=batch_id,
                )
            )
            continue
        response = row.get("response")
        top_error = row.get("error")
        if row_sources[custom_id] == "output" and isinstance(response, dict):
            status_code = _optional_int(response.get("status_code"))
            body = response.get("body")
            body_mapping = dict(body) if isinstance(body, dict) else None
            response_status = (
                body_mapping.get("status") if body_mapping is not None else None
            )
            response_complete = (
                status_code == 200
                and top_error is None
                and response_status == "completed"
                and body_mapping.get("error") is None
            )
            text_error: dict[str, Any] | None = None
            if response_complete:
                try:
                    _responses_output_text(body_mapping)
                except ValueError as error:
                    text_error = {
                        "message": str(error),
                        "response_status": response_status,
                    }
                else:
                    results.append(
                        _result_from_request(
                            request,
                            outcome="succeeded",
                            status_code=status_code,
                            response_body=body_mapping,
                            provider_batch_id=batch_id,
                        )
                    )
                    continue
            error_mapping = text_error or _provider_response_error(
                status_code=status_code, response_body=body_mapping, top_error=top_error
            )
            results.append(
                _result_from_request(
                    request,
                    outcome="failed",
                    status_code=status_code,
                    response_body=body_mapping,
                    error=error_mapping,
                    provider_batch_id=batch_id,
                )
            )
            continue
        error_mapping = (
            dict(top_error)
            if isinstance(top_error, dict)
            else {"message": "Provider error row has no structured error"}
        )
        results.append(
            _result_from_request(
                request,
                outcome="failed",
                error=error_mapping,
                provider_batch_id=batch_id,
            )
        )

    result_payload = _result_rows_bytes(results)
    return results, {
        "schema_version": 1,
        "batch_identity_sha256": loaded.manifest["batch_identity_sha256"],
        "chunk_index": chunk.chunk_index,
        "input_sha256": chunk.manifest["input_sha256"],
        "provider_batch_id": batch_id,
        "provider_status": provider_status,
        "output_file_sha256": (
            _sha256_bytes(output_path.read_bytes()) if output_path is not None else None
        ),
        "error_file_sha256": (
            _sha256_bytes(error_path.read_bytes()) if error_path is not None else None
        ),
        "results_sha256": _sha256_bytes(result_payload),
        "request_count": len(results),
        "succeeded_count": sum(result.outcome == "succeeded" for result in results),
        "failed_count": sum(result.outcome == "failed" for result in results),
        "missing_count": sum(result.outcome == "missing" for result in results),
        "reconciled_at": _utc_now(),
    }


def _provider_response_error(
    status_code: int | None,
    response_body: Mapping[str, Any] | None,
    top_error: Any,
) -> dict[str, Any]:
    """Explain why one Batch row is not a completed Responses result.

    The Batch wrapper can report HTTP 200 even when the nested Responses object
    is ``incomplete``. Keeping that body on ``BatchPromptResult`` preserves any
    partial output for diagnosis, but the row must remain failed so callers
    cannot mistake partial text for a finished answer.
    """
    if isinstance(top_error, dict):
        return dict(top_error)
    if top_error is not None:
        return {"message": str(top_error)}

    response_status = response_body.get("status") if response_body else None
    body_error = response_body.get("error") if response_body else None
    if isinstance(body_error, dict):
        error = dict(body_error)
        error.setdefault("response_status", response_status)
        return error

    if status_code != 200 or response_body is None:
        return {
            "message": "Provider response was not a successful HTTP 200 JSON body",
            "status_code": status_code,
        }

    status_label = response_status if isinstance(response_status, str) else "missing"
    error: dict[str, Any] = {
        "message": f"Responses API result was {status_label!r}, not 'completed'",
        "response_status": response_status,
    }
    incomplete_details = response_body.get("incomplete_details")
    if isinstance(incomplete_details, dict):
        error["incomplete_details"] = dict(incomplete_details)
    return error


def _pending_results(
    chunk: _LoadedChunk, provider_batch_id: str
) -> list[BatchPromptResult]:
    return [
        _result_from_request(
            request,
            outcome="pending",
            provider_batch_id=provider_batch_id,
        )
        for request in chunk.requests
    ]


def _result_from_request(
    request: Mapping[str, Any],
    outcome: _PromptOutcome,
    provider_batch_id: str | None,
    status_code: int | None = None,
    response_body: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
) -> BatchPromptResult:
    return BatchPromptResult(
        request_index=int(request["request_index"]),
        prompt_id=str(request["custom_id"]),
        prompt_sha256=str(request["prompt_sha256"]),
        outcome=outcome,
        status_code=status_code,
        response_body=response_body,
        error=error,
        provider_batch_id=provider_batch_id,
    )


def _result_rows_bytes(results: Sequence[BatchPromptResult]) -> bytes:
    return _canonical_jsonl_bytes(
        [
            {
                "request_index": result.request_index,
                "custom_id": result.prompt_id,
                "prompt_sha256": result.prompt_sha256,
                "outcome": result.outcome,
                "status_code": result.status_code,
                "response_body": result.response_body,
                "error": result.error,
                "provider_batch_id": result.provider_batch_id,
            }
            for result in results
        ]
    )


def _load_saved_chunk_results(
    reconciliation: Mapping[str, Any],
    result_path: Path,
    loaded: _LoadedBatch,
    chunk: _LoadedChunk,
    receipt: Mapping[str, Any],
) -> list[BatchPromptResult]:
    expected = {
        "schema_version": 1,
        "batch_identity_sha256": loaded.manifest["batch_identity_sha256"],
        "chunk_index": chunk.chunk_index,
        "input_sha256": chunk.manifest["input_sha256"],
        "provider_batch_id": receipt["provider_batch_id"],
    }
    for key, value in expected.items():
        if reconciliation.get(key) != value:
            raise ValueError(f"Saved OpenAI Batch reconciliation {key} mismatch")
    for artifact_name, hash_name in (
        ("output.jsonl", "output_file_sha256"),
        ("errors.jsonl", "error_file_sha256"),
    ):
        expected_hash = reconciliation.get(hash_name)
        if expected_hash is None:
            continue
        artifact_path = chunk.directory / artifact_name
        if not artifact_path.is_file() or _sha256_bytes(
            artifact_path.read_bytes()
        ) != str(expected_hash):
            raise ValueError(f"Saved OpenAI Batch {artifact_name} hash changed")
    if not result_path.is_file():
        raise ValueError("OpenAI Batch reconciliation exists without results.jsonl")
    payload = result_path.read_bytes()
    if _sha256_bytes(payload) != str(reconciliation.get("results_sha256")):
        raise ValueError("Saved OpenAI Batch result artifact hash changed")
    rows = _read_jsonl_bytes(payload, result_path)
    if len(rows) != len(chunk.requests):
        raise ValueError("Saved OpenAI Batch result count changed")
    results: list[BatchPromptResult] = []
    for row, request in zip(rows, chunk.requests, strict=True):
        if int(row.get("request_index", -1)) != int(request["request_index"]):
            raise ValueError("Saved OpenAI Batch result order changed")
        if str(row.get("custom_id")) != str(request["custom_id"]):
            raise ValueError("Saved OpenAI Batch result custom_id changed")
        if str(row.get("prompt_sha256")) != str(request["prompt_sha256"]):
            raise ValueError("Saved OpenAI Batch result prompt hash changed")
        if str(row.get("provider_batch_id")) != str(receipt["provider_batch_id"]):
            raise ValueError("Saved OpenAI Batch result provider Batch ID changed")
        outcome = str(row.get("outcome"))
        if outcome not in {"succeeded", "failed", "missing"}:
            raise ValueError("Saved OpenAI Batch result has an invalid outcome")
        response_body = row.get("response_body")
        error = row.get("error")
        results.append(
            BatchPromptResult(
                request_index=int(row["request_index"]),
                prompt_id=str(row["custom_id"]),
                prompt_sha256=str(row["prompt_sha256"]),
                outcome=cast(_PromptOutcome, outcome),
                status_code=_optional_int(row.get("status_code")),
                response_body=(
                    dict(response_body) if isinstance(response_body, dict) else None
                ),
                error=dict(error) if isinstance(error, dict) else None,
                provider_batch_id=str(row["provider_batch_id"]),
            )
        )
    return results


def _responses_output_text(body: Mapping[str, Any]) -> str:
    direct = body.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    chunks: list[str] = []
    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)
                ):
                    chunks.append(str(part["text"]))
    if not chunks:
        raise ValueError("Successful OpenAI Batch response body has no output text")
    return "".join(chunks)


def _batch_record(batch: Any) -> dict[str, Any]:
    if isinstance(batch, dict):
        record = _canonical_json_object(batch, "Batch object")
    elif hasattr(batch, "model_dump"):
        dumped = batch.model_dump(mode="json")
        record = _canonical_json_object(dumped, "Batch object")
    else:
        names = (
            "id",
            "status",
            "endpoint",
            "input_file_id",
            "output_file_id",
            "error_file_id",
            "request_counts",
            "metadata",
            "errors",
            "created_at",
            "in_progress_at",
            "finalizing_at",
            "completed_at",
            "expires_at",
            "failed_at",
            "expired_at",
            "cancelled_at",
        )
        record = {name: getattr(batch, name) for name in names if hasattr(batch, name)}
        record = _canonical_json_object(record, "Batch object")
    if (
        not str(record.get("id", "")).strip()
        or not str(record.get("status", "")).strip()
    ):
        raise ValueError("OpenAI Batch object is missing id or status")
    return record


def _required_object_id(value: Any, label: str) -> str:
    raw_id = value.get("id") if isinstance(value, dict) else getattr(value, "id", None)
    object_id = str(raw_id or "").strip()
    if not object_id:
        raise ValueError(f"OpenAI {label} is missing id")
    return object_id


def _validate_batch_client(service: Any) -> Any:
    files = getattr(service, "files", None)
    batches = getattr(service, "batches", None)
    required_operations = (
        (files, "create", "files.create"),
        (files, "content", "files.content"),
        (files, "retrieve", "files.retrieve"),
        (batches, "create", "batches.create"),
        (batches, "retrieve", "batches.retrieve"),
        (batches, "list", "batches.list"),
    )
    missing = [
        label
        for owner, operation, label in required_operations
        if not callable(getattr(owner, operation, None))
    ]
    if missing:
        raise TypeError(
            "llm['service'] is missing OpenAI Batch operations: " + ", ".join(missing)
        )
    return service


# Artifact encoding and publication
def _safe_artifact_path(root: Path, relative_value: str) -> Path:
    if not relative_value:
        raise ValueError("OpenAI Batch manifest contains a blank artifact path")
    path = (root / relative_value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("OpenAI Batch artifact path escapes the run directory")
    if not path.is_file():
        raise FileNotFoundError(f"OpenAI Batch artifact does not exist: {path}")
    return path


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"Immutable artifact conflicts with existing path: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        _publish_temporary_immutable(path, temporary_path, payload)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _publish_temporary_immutable(
    path: Path,
    temporary_path: Path,
    payload: bytes,
) -> None:
    """Publish a flushed sibling file without replacing existing content."""
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"Immutable artifact conflicts with existing path: {path}")
        return
    try:
        os.link(temporary_path, path)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(
                f"Immutable artifact was written concurrently with different content: {path}"
            )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Artifact contains invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Artifact must contain a JSON object: {path}")
    return value


def _read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl_bytes(path.read_bytes(), path)


def _read_jsonl_bytes(payload: bytes, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"JSONL artifact is not UTF-8: {path}") from error
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"JSONL artifact has invalid row {line_number}: {path}"
            ) from error
        if not isinstance(row, dict):
            raise TypeError(
                f"JSONL artifact row {line_number} is not an object: {path}"
            )
        rows.append(row)
    return rows


def _canonical_json_object(value: Any, label: str) -> dict[str, Any]:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        normalized = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a JSON object") from error
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must be a JSON object")
    return normalized


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


# 2) Batch lifecycle wrappers
def _prepare_openai_prompt_batch(
    prompts: Mapping[str, str],
    run_directory: str | Path,
    model: str,
    settings: Mapping[str, Any] | None = None,
    system_prompt: str | None = None,
    metadata: Mapping[str, str] | None = None,
    max_requests_per_batch: int = _OPENAI_BATCH_MAX_REQUESTS,
    max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES,
) -> _PreparedOpenAIPromptBatch:
    """Write or verify the exact local JSONL and manifest without submitting.

    Caller prompt IDs become provider ``custom_id`` values. Duplicate prompt
    text remains a distinct provider request when it has a distinct ID. Reusing
    the same directory with any changed prompt ID, text, order, model, setting,
    metadata, or limit is rejected.
    """
    validated = _validate_prepare_inputs(
        prompts,
        model,
        settings,
        system_prompt,
        metadata,
        max_requests_per_batch,
        max_input_bytes,
    )
    prompt_rows, input_rows, input_lines = _build_request_rows(
        validated.prompts,
        model=validated.model,
        settings=validated.settings,
        system_prompt=validated.system_prompt,
    )
    chunks = _chunk_request_rows(
        prompt_rows,
        input_rows,
        input_lines,
        maximum_requests=max_requests_per_batch,
        maximum_bytes=max_input_bytes,
    )
    batch_identity = {
        "endpoint": _OPENAI_BATCH_ENDPOINT,
        "model": validated.model,
        "settings": validated.settings,
        "system_prompt": validated.system_prompt,
        "metadata": validated.metadata,
        "max_requests_per_batch": max_requests_per_batch,
        "max_input_bytes": max_input_bytes,
        "prompts": [
            {"prompt_id": prompt_id, "prompt": prompt}
            for prompt_id, prompt in validated.prompts.items()
        ],
    }
    batch_identity_sha256 = _sha256_bytes(_canonical_json_bytes(batch_identity))
    root = Path(run_directory)
    root.mkdir(parents=True, exist_ok=True)

    chunk_manifests: list[dict[str, Any]] = []
    chunk_artifacts: list[tuple[Path, bytes]] = []
    logical_input = bytearray()
    for chunk_index, chunk in enumerate(chunks, start=1):
        chunk_directory = root / "chunks" / f"chunk_{chunk_index:03d}"
        input_payload = b"".join(chunk["input_lines"])
        request_payload = _canonical_jsonl_bytes(chunk["prompt_rows"])
        input_path = chunk_directory / "input.jsonl"
        request_path = chunk_directory / "requests.jsonl"
        chunk_artifacts.extend(
            ((input_path, input_payload), (request_path, request_payload))
        )
        logical_input.extend(input_payload)
        chunk_manifests.append(
            {
                "chunk_index": chunk_index,
                "request_count": len(chunk["input_rows"]),
                "first_request_index": int(chunk["prompt_rows"][0]["request_index"]),
                "last_request_index": int(chunk["prompt_rows"][-1]["request_index"]),
                "input_path": input_path.relative_to(root).as_posix(),
                "input_sha256": _sha256_bytes(input_payload),
                "input_bytes": len(input_payload),
                "requests_path": request_path.relative_to(root).as_posix(),
                "requests_sha256": _sha256_bytes(request_payload),
            }
        )

    manifest = {
        "schema_version": 1,
        "provider": "openai",
        "endpoint": _OPENAI_BATCH_ENDPOINT,
        "completion_window": _OPENAI_BATCH_COMPLETION_WINDOW,
        "model": validated.model,
        "settings": validated.settings,
        "system_prompt": validated.system_prompt,
        "metadata": validated.metadata,
        "request_count": len(prompt_rows),
        "chunk_count": len(chunk_manifests),
        "max_requests_per_batch": max_requests_per_batch,
        "max_input_bytes": max_input_bytes,
        "batch_identity_sha256": batch_identity_sha256,
        "logical_input_sha256": _sha256_bytes(bytes(logical_input)),
        "chunks": chunk_manifests,
    }
    manifest_path = root / "batch_manifest.json"
    manifest_payload = _pretty_json_bytes(manifest)

    if manifest_path.exists():
        _write_immutable(manifest_path, manifest_payload)
    for artifact_path, artifact_payload in chunk_artifacts:
        _write_immutable(artifact_path, artifact_payload)
    _write_immutable(manifest_path, manifest_payload)
    loaded = _load_and_verify_batch(root)
    prepared = _PreparedOpenAIPromptBatch(
        run_directory=loaded.run_directory,
        request_count=int(loaded.manifest["request_count"]),
        chunk_count=int(loaded.manifest["chunk_count"]),
        batch_identity_sha256=str(loaded.manifest["batch_identity_sha256"]),
    )
    return prepared


def _submit_chunk(client: Any, loaded: _LoadedBatch, chunk: _LoadedChunk) -> _SubmittedChunk:
    receipt_path = chunk.directory / "submission_receipt.json"
    if receipt_path.exists():
        receipt = _load_json_object(receipt_path)
        _verify_submission_receipt(receipt, loaded, chunk)
        submitted_chunk = _SubmittedChunk(
            provider_batch_id=str(receipt["provider_batch_id"]),
            uploaded=False,
            submitted=False,
        )
        return submitted_chunk

    upload_path = chunk.directory / "upload_receipt.json"
    uploaded_now = not upload_path.exists()
    if uploaded_now:
        input_path = chunk.directory / "input.jsonl"
        with input_path.open("rb") as input_file:
            uploaded_file = client.files.create(file=input_file, purpose="batch")
        upload = {
            "schema_version": 1,
            "input_sha256": str(chunk.manifest["input_sha256"]),
            "input_bytes": int(chunk.manifest["input_bytes"]),
            "provider_input_file_id": _required_object_id(uploaded_file, "uploaded file"),
            "uploaded_at": _utc_now(),
        }
        _write_immutable(upload_path, _pretty_json_bytes(upload))
    else:
        upload = _load_json_object(upload_path)
        _verify_upload_receipt(upload, chunk)

    provider_metadata = _provider_metadata(loaded, chunk)
    intent = {
        "schema_version": 1,
        "endpoint": _OPENAI_BATCH_ENDPOINT,
        "completion_window": _OPENAI_BATCH_COMPLETION_WINDOW,
        "provider_input_file_id": str(upload["provider_input_file_id"]),
        "metadata": provider_metadata,
        "input_sha256": str(chunk.manifest["input_sha256"]),
    }
    intent_path = chunk.directory / "submission_intent.json"
    intent_already_existed = intent_path.exists()
    _write_immutable(intent_path, _pretty_json_bytes(intent))

    submitted_now = not intent_already_existed
    if intent_already_existed:
        provider_batch = _recover_unrecorded_batch(client, intent)
    else:
        try:
            created = client.batches.create(
                input_file_id=str(upload["provider_input_file_id"]),
                endpoint=_OPENAI_BATCH_ENDPOINT,
                completion_window=_OPENAI_BATCH_COMPLETION_WINDOW,
                metadata=provider_metadata,
            )
            provider_batch = _batch_record(created)
        except Exception as error:
            raise AmbiguousBatchSubmissionError(
                "OpenAI Batch creation failed after the durable submission intent "
                "was written. Rerun the same submission to search provider Batches "
                "before deciding whether any resubmission is safe."
            ) from error

    receipt = {
        "schema_version": 1,
        "batch_identity_sha256": str(loaded.manifest["batch_identity_sha256"]),
        "chunk_index": chunk.chunk_index,
        "input_sha256": str(chunk.manifest["input_sha256"]),
        "provider_input_file_id": str(upload["provider_input_file_id"]),
        "provider_batch_id": str(provider_batch["id"]),
        "provider_status_at_submission": str(provider_batch["status"]),
        "provider_batch": provider_batch,
        "submitted_at": _utc_now(),
    }
    _verify_submission_receipt(receipt, loaded, chunk)
    _write_immutable(receipt_path, _pretty_json_bytes(receipt))
    submitted_chunk = _SubmittedChunk(
        provider_batch_id=str(receipt["provider_batch_id"]),
        uploaded=uploaded_now,
        submitted=submitted_now,
    )
    return submitted_chunk


def _collect_chunk(
    client: Any,
    loaded: _LoadedBatch,
    chunk: _LoadedChunk,
    receipt: Mapping[str, Any],
    wait: bool,
    poll_interval_seconds: float,
) -> _CollectedChunk:
    terminal_batch_statuses = {"completed", "expired", "cancelled", "failed"}
    reconciliation_path = chunk.directory / "reconciliation.json"
    result_path = chunk.directory / "results.jsonl"

    if reconciliation_path.exists():
        reconciliation = _load_json_object(reconciliation_path)
        results = _load_saved_chunk_results(
            reconciliation, result_path, loaded, chunk, receipt
        )
        collected = _CollectedChunk(
            provider_status=str(reconciliation["provider_status"]),
            results=tuple(results),
            checked_batch_count=0,
            downloaded_file_count=0,
        )
        return collected

    batch_id = str(receipt["provider_batch_id"])
    status = _batch_record(client.batches.retrieve(batch_id))
    checked_batch_count = 1
    while wait and str(status["status"]) not in terminal_batch_statuses:
        time.sleep(poll_interval_seconds)
        status = _batch_record(client.batches.retrieve(batch_id))
        checked_batch_count += 1
    _append_status_event(chunk.directory / "status.jsonl", batch_id, status)
    provider_status = str(status["status"])

    if provider_status not in terminal_batch_statuses:
        collected = _CollectedChunk(
            provider_status=provider_status,
            results=tuple(_pending_results(chunk, batch_id)),
            checked_batch_count=checked_batch_count,
            downloaded_file_count=0,
        )
        return collected

    output_path, output_downloaded = _download_terminal_file(
        client,
        chunk,
        status.get("output_file_id"),
        artifact_name="output.jsonl",
        receipt_name="output_download.json",
    )
    error_path, error_downloaded = _download_terminal_file(
        client,
        chunk,
        status.get("error_file_id"),
        artifact_name="errors.jsonl",
        receipt_name="error_download.json",
    )
    results, reconciliation = _reconcile_terminal_chunk(
        loaded,
        chunk,
        receipt,
        provider_status,
        output_path,
        error_path,
    )
    _write_immutable(result_path, _result_rows_bytes(results))
    _write_immutable(reconciliation_path, _pretty_json_bytes(reconciliation))
    downloaded_file_count = int(output_downloaded) + int(error_downloaded)
    collected = _CollectedChunk(
        provider_status=provider_status,
        results=tuple(results),
        checked_batch_count=checked_batch_count,
        downloaded_file_count=downloaded_file_count,
    )
    return collected


# 3) Public wrapper functions
def submit_openai_prompt_batch(
    prompts: Mapping[str, str],
    run_directory: str | Path,
    llm: Mapping[str, object],
    settings: Mapping[str, Any] | None = None,
    system_prompt: str | None = None,
    metadata: Mapping[str, str] | None = None,
    max_requests_per_batch: int = _OPENAI_BATCH_MAX_REQUESTS,
    max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES,
) -> OpenAIPromptBatchSubmission:
    """Prepare and submit every missing chunk, reusing exact saved receipts.

    Upload receipts are saved before Batch creation. A submission intent is also
    saved before calling ``batches.create``. If that call may have succeeded but
    its receipt was lost, a rerun searches existing Batches for the exact input
    file and reserved metadata; it never blindly creates a duplicate Batch.
    """
    llm_client = resolve_openai_client(llm, required_response_methods=())
    batch_client = _validate_batch_client(llm_client.service)
    prepared = _prepare_openai_prompt_batch(
        prompts,
        run_directory=run_directory,
        model=llm_client.model,
        settings=settings,
        system_prompt=system_prompt,
        metadata=metadata,
        max_requests_per_batch=max_requests_per_batch,
        max_input_bytes=max_input_bytes,
    )
    loaded = _load_and_verify_batch(prepared.run_directory)
    submitted_chunks = [
        _submit_chunk(batch_client, loaded, chunk) for chunk in loaded.chunks
    ]

    submission = OpenAIPromptBatchSubmission(
        run_directory=prepared.run_directory,
        request_count=prepared.request_count,
        chunk_count=prepared.chunk_count,
        provider_batch_ids=tuple(
            chunk.provider_batch_id for chunk in submitted_chunks
        ),
        new_upload_count=sum(chunk.uploaded for chunk in submitted_chunks),
        new_submission_count=sum(chunk.submitted for chunk in submitted_chunks),
    )
    return submission


def collect_openai_prompt_batch(
    run_directory: str | Path,
    llm: Mapping[str, object],
    wait: bool = False,
    poll_interval_seconds: float = 60.0,
) -> OpenAIPromptBatchCollection:
    """Check or wait for submitted chunks, then reconcile exact saved results.

    ``wait=False`` performs at most one provider status check per unreconciled
    chunk. ``wait=True`` polls until terminal. Downloaded files and reconciled
    result artifacts are reused without another provider call on later runs.
    """
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be greater than zero")
    loaded = _load_and_verify_batch(Path(run_directory))
    llm_client = resolve_openai_client(llm, required_response_methods=())
    batch_client = _validate_batch_client(llm_client.service)
    if llm_client.model != loaded.manifest.get("model"):
        raise ValueError("llm['model'] does not match the submitted Batch model")
    receipts = _load_all_submission_receipts(loaded)
    collected_chunks = [
        _collect_chunk(
            batch_client,
            loaded,
            chunk,
            receipt,
            wait,
            poll_interval_seconds,
        )
        for chunk, receipt in zip(loaded.chunks, receipts, strict=True)
    ]
    results = [
        result for collected in collected_chunks for result in collected.results
    ]

    ordered = tuple(sorted(results, key=lambda result: result.request_index))
    expected_indexes = list(range(int(loaded.manifest["request_count"])))
    if [result.request_index for result in ordered] != expected_indexes:
        raise ValueError("Collected OpenAI Batch results do not cover prompt order")
    if any(result.outcome == "pending" for result in ordered):
        collection_status: _CollectionStatus = "pending"
    elif all(result.outcome == "succeeded" for result in ordered):
        collection_status = "completed"
    else:
        collection_status = "completed_with_failures"
    collection = OpenAIPromptBatchCollection(
        run_directory=loaded.run_directory,
        status=collection_status,
        chunk_statuses=tuple(chunk.provider_status for chunk in collected_chunks),
        results=ordered,
        checked_batch_count=sum(
            chunk.checked_batch_count for chunk in collected_chunks
        ),
        downloaded_file_count=sum(
            chunk.downloaded_file_count for chunk in collected_chunks
        ),
    )
    return collection
