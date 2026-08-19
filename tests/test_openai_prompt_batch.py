import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from _openai_batch_fake import FakeOpenAI

from llm_patterns import (
    AmbiguousBatchSubmissionError,
    collect_openai_prompt_batch,
    submit_openai_prompt_batch,
)


class OpenAIPromptBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompts = {
            "inspection-104": (
                "Classify as routine, priority, or urgent.\n\n"
                "A sparking outlet is smoking."
            ),
            "inspection-105": (
                "Classify as routine, priority, or urgent.\n\n"
                "There is a small paint chip."
            ),
        }

    def test_submission_uses_stable_ids_luna_and_no_reasoning_tokens(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            client = FakeOpenAI()
            run_directory = Path(temporary_directory) / "batch"

            submission = submit_openai_prompt_batch(
                self.prompts,
                run_directory=run_directory,
                llm={"service": client, "model": "gpt-5.6-luna"},
            )

            self.assertEqual(submission.request_count, 2)
            self.assertEqual(submission.chunk_count, 1)
            self.assertEqual(submission.provider_batch_ids, ("batch-001",))
            self.assertEqual(submission.new_upload_count, 1)
            self.assertEqual(submission.new_submission_count, 1)
            input_payload = next(
                payload
                for file_id, payload in client.files.payloads.items()
                if file_id.startswith("file-input-")
            )
            rows = [json.loads(line) for line in input_payload.splitlines()]
            self.assertEqual(
                [row["custom_id"] for row in rows],
                ["inspection-104", "inspection-105"],
            )
            self.assertEqual(rows[0]["body"]["model"], "gpt-5.6-luna")
            self.assertEqual(
                rows[0]["body"]["reasoning"],
                {"effort": "none"},
            )

    def test_exact_resubmission_reuses_upload_and_batch_receipts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            client = FakeOpenAI()
            run_directory = Path(temporary_directory) / "batch"
            llm = {"service": client, "model": "test-model"}

            first = submit_openai_prompt_batch(
                self.prompts,
                run_directory=run_directory,
                llm=llm,
            )
            second = submit_openai_prompt_batch(
                self.prompts,
                run_directory=run_directory,
                llm=llm,
            )

            self.assertEqual(first.provider_batch_ids, second.provider_batch_ids)
            self.assertEqual(second.new_upload_count, 0)
            self.assertEqual(second.new_submission_count, 0)
            self.assertEqual(client.files.upload_count, 1)
            self.assertEqual(client.batches.create_count, 1)

    def test_changed_input_conflicts_with_existing_run_directory(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            client = FakeOpenAI()
            run_directory = Path(temporary_directory) / "batch"
            llm = {"service": client, "model": "test-model"}
            submit_openai_prompt_batch(
                self.prompts,
                run_directory=run_directory,
                llm=llm,
            )

            changed = {**self.prompts, "inspection-105": "Changed prompt"}
            with self.assertRaisesRegex(ValueError, "Immutable artifact conflicts"):
                submit_openai_prompt_batch(
                    changed,
                    run_directory=run_directory,
                    llm=llm,
                )

            self.assertEqual(client.batches.create_count, 1)

    def test_collection_restores_original_stable_id_order_and_then_reuses_it(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            client = FakeOpenAI()
            run_directory = Path(temporary_directory) / "batch"
            llm = {"service": client, "model": "test-model"}
            submit_openai_prompt_batch(
                self.prompts,
                run_directory=run_directory,
                llm=llm,
            )

            pending = collect_openai_prompt_batch(run_directory, llm)
            completed = collect_openai_prompt_batch(run_directory, llm)
            cached = collect_openai_prompt_batch(run_directory, llm)

            self.assertEqual(pending.status, "pending")
            self.assertEqual(completed.status, "completed")
            self.assertEqual(
                completed.response_texts(),
                {"inspection-104": "urgent", "inspection-105": "routine"},
            )
            self.assertEqual(cached.response_texts(), completed.response_texts())
            self.assertEqual(cached.checked_batch_count, 0)
            self.assertEqual(cached.downloaded_file_count, 0)
            self.assertEqual(client.batches.retrieve_count, 2)
            self.assertEqual(client.files.download_count, 1)

    def test_incomplete_nested_response_is_preserved_as_failure(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            client = FakeOpenAI()
            run_directory = Path(temporary_directory) / "batch"
            llm = {"service": client, "model": "test-model"}
            submission = submit_openai_prompt_batch(
                {"inspection-104": self.prompts["inspection-104"]},
                run_directory=run_directory,
                llm=llm,
            )
            batch_id = submission.provider_batch_ids[0]
            output_id = client.batches.output_ids[batch_id]
            row = json.loads(client.files.payloads[output_id])
            row["response"]["body"]["status"] = "incomplete"
            row["response"]["body"]["incomplete_details"] = {
                "reason": "max_output_tokens"
            }
            client.files.payloads[output_id] = (
                json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n"
            )

            collect_openai_prompt_batch(run_directory, llm)
            completed = collect_openai_prompt_batch(run_directory, llm)

            self.assertEqual(completed.status, "completed_with_failures")
            self.assertEqual(completed.results[0].outcome, "failed")
            self.assertEqual(
                completed.results[0].error["response_status"],
                "incomplete",
            )
            with self.assertRaisesRegex(ValueError, "incomplete or failed"):
                completed.response_texts()

    def test_completed_response_without_output_text_is_a_failure(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            client = FakeOpenAI()
            run_directory = Path(temporary_directory) / "batch"
            llm = {"service": client, "model": "test-model"}
            submission = submit_openai_prompt_batch(
                {"inspection-104": self.prompts["inspection-104"]},
                run_directory=run_directory,
                llm=llm,
            )
            output_id = client.batches.output_ids[submission.provider_batch_ids[0]]
            row = json.loads(client.files.payloads[output_id])
            row["response"]["body"]["output"] = []
            client.files.payloads[output_id] = (
                json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n"
            )

            collect_openai_prompt_batch(run_directory, llm)
            completed = collect_openai_prompt_batch(run_directory, llm)

            self.assertEqual(completed.status, "completed_with_failures")
            self.assertEqual(completed.results[0].outcome, "failed")
            self.assertIn("no output text", completed.results[0].error["message"])

    def test_ambiguous_create_is_recovered_without_duplicate_submission(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            client = FakeOpenAI()
            run_directory = Path(temporary_directory) / "batch"
            llm = {"service": client, "model": "test-model"}
            create = client.batches.create

            def create_then_disconnect(**request):
                create(**request)
                raise ConnectionError("connection closed after create")

            client.batches.create = create_then_disconnect
            with self.assertRaises(AmbiguousBatchSubmissionError):
                submit_openai_prompt_batch(
                    self.prompts,
                    run_directory=run_directory,
                    llm=llm,
                )

            client.batches.create = create
            recovered = submit_openai_prompt_batch(
                self.prompts,
                run_directory=run_directory,
                llm=llm,
            )

            self.assertEqual(recovered.provider_batch_ids, ("batch-001",))
            self.assertEqual(recovered.new_submission_count, 0)
            self.assertEqual(client.batches.create_count, 1)

    def test_download_byte_mismatch_is_rejected_before_publication(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            client = FakeOpenAI()
            run_directory = Path(temporary_directory) / "batch"
            llm = {"service": client, "model": "test-model"}
            submit_openai_prompt_batch(
                {"inspection-104": self.prompts["inspection-104"]},
                run_directory=run_directory,
                llm=llm,
            )
            collect_openai_prompt_batch(run_directory, llm)
            retrieve = client.files.retrieve

            def wrong_size(file_id):
                record = retrieve(file_id)
                return {**record, "bytes": int(record["bytes"]) + 1}

            client.files.retrieve = wrong_size
            with self.assertRaisesRegex(ValueError, "bytes; expected"):
                collect_openai_prompt_batch(run_directory, llm)

            self.assertFalse(
                (run_directory / "chunks" / "chunk_001" / "output.jsonl").exists()
            )

    def test_chunking_submits_each_bounded_file_and_reconciles_all_prompts(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            client = FakeOpenAI()
            run_directory = Path(temporary_directory) / "batch"
            llm = {"service": client, "model": "test-model"}

            submission = submit_openai_prompt_batch(
                self.prompts,
                run_directory=run_directory,
                llm=llm,
                max_requests_per_batch=1,
            )
            pending = collect_openai_prompt_batch(run_directory, llm)
            completed = collect_openai_prompt_batch(run_directory, llm)

            self.assertEqual(submission.chunk_count, 2)
            self.assertEqual(len(submission.provider_batch_ids), 2)
            self.assertEqual(pending.status, "pending")
            self.assertEqual(
                completed.response_texts(),
                {"inspection-104": "urgent", "inspection-105": "routine"},
            )

    def test_invalid_inputs_fail_before_upload(self) -> None:
        invalid_prompts = (
            ["not a mapping"],
            {},
            {"": "Prompt"},
            {" spaced ": "Prompt"},
            {"valid": ""},
        )
        for prompts in invalid_prompts:
            with TemporaryDirectory() as temporary_directory:
                client = FakeOpenAI()
                with (
                    self.subTest(prompts=prompts),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    submit_openai_prompt_batch(
                        prompts,
                        run_directory=Path(temporary_directory) / "batch",
                        llm={"service": client, "model": "test-model"},
                    )
                self.assertEqual(client.files.upload_count, 0)


if __name__ == "__main__":
    unittest.main()
