"""Small in-memory OpenAI Batch fake used only by example 15.

This is provider-free verification scaffolding, not part of the reusable helper
and not code an application needs when it passes a real ``OpenAI()`` service.
It implements only the Files and Batches operations exercised by offline probes.
"""

import json
from pathlib import Path
from typing import Any, BinaryIO, Self


class FakeFileContent:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def stream_to_file(self, path: str | Path) -> None:
        """Match the current SDK's streamed binary-response operation."""
        Path(path).write_bytes(self.payload)


class FakeFiles:
    """Provide the two Files API operations used by the reference."""

    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}
        self.upload_count = 0
        self.download_count = 0

    @property
    def with_streaming_response(self) -> "FakeFiles":
        return self

    def create(self, *, file: BinaryIO, purpose: str) -> dict[str, str]:
        assert purpose == "batch"
        self.upload_count += 1
        file_id = f"file-input-{self.upload_count:03d}"
        self.payloads[file_id] = file.read()
        return {"id": file_id}

    def add_output(self, payload: bytes) -> str:
        file_id = f"file-output-{len(self.payloads) + 1:03d}"
        self.payloads[file_id] = payload
        return file_id

    def content(self, file_id: str) -> FakeFileContent:
        self.download_count += 1
        return FakeFileContent(self.payloads[file_id])

    def retrieve(self, file_id: str) -> dict[str, int | str]:
        return {"id": file_id, "bytes": len(self.payloads[file_id])}


class FakeBatches:
    """Simulate validating/in-progress/completed provider Batch objects."""

    def __init__(self, files: FakeFiles) -> None:
        self.files = files
        self.records: dict[str, dict[str, Any]] = {}
        self.output_ids: dict[str, str] = {}
        self.retrieve_counts: dict[str, int] = {}
        self.create_count = 0
        self.retrieve_count = 0

    def create(
        self,
        *,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict[str, str],
    ) -> dict[str, Any]:
        assert endpoint == "/v1/responses"
        assert completion_window == "24h"
        self.create_count += 1
        batch_id = f"batch-{self.create_count:03d}"
        input_rows = [
            json.loads(line)
            for line in self.files.payloads[input_file_id].decode("utf-8").splitlines()
        ]

        # Provider output order is intentionally reversed. Correct collection
        # must use custom_id and cannot rely on this file order.
        output_rows = []
        for row in reversed(input_rows):
            prompt = row["body"]["input"][-1]["content"]
            if "sparking outlet" in prompt.casefold():
                label = "urgent"
            elif "paint chip" in prompt.casefold():
                label = "routine"
            else:
                label = "priority"
            output_rows.append(
                {
                    "custom_id": row["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "content": [{"type": "output_text", "text": label}],
                                }
                            ],
                        },
                    },
                    "error": None,
                }
            )
        output_payload = b"".join(
            json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in output_rows
        )
        self.output_ids[batch_id] = self.files.add_output(output_payload)
        self.records[batch_id] = {
            "id": batch_id,
            "status": "validating",
            "endpoint": endpoint,
            "input_file_id": input_file_id,
            "output_file_id": None,
            "error_file_id": None,
            "request_counts": {
                "total": len(input_rows),
                "completed": 0,
                "failed": 0,
            },
            "metadata": metadata,
        }
        return dict(self.records[batch_id])

    def retrieve(self, batch_id: str) -> dict[str, Any]:
        self.retrieve_count += 1
        count = self.retrieve_counts.get(batch_id, 0) + 1
        self.retrieve_counts[batch_id] = count
        record = dict(self.records[batch_id])
        if count == 1:
            record["status"] = "in_progress"
        else:
            record["status"] = "completed"
            record["output_file_id"] = self.output_ids[batch_id]
            request_counts = dict(record["request_counts"])
            request_counts["completed"] = request_counts["total"]
            record["request_counts"] = request_counts
        self.records[batch_id] = record
        return dict(record)

    def list(self, *, limit: int) -> list[dict[str, Any]]:
        assert limit == 100
        return [dict(record) for record in self.records.values()]


class FakeOpenAI:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.batches = FakeBatches(self.files)
