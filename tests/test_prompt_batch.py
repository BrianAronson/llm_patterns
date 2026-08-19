import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from llm_patterns import run_prompt_batch
from llm_patterns import _adaptive_openai as adaptive_openai


class _Response:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _Responses:
    def __init__(self, handler) -> None:
        self._handler = handler

    def create(self, **kwargs):
        return self._handler(**kwargs)


class _Service:
    def __init__(self, handler) -> None:
        self.responses = _Responses(handler)
        self.option_calls: list[dict[str, object]] = []

    def with_options(self, **options):
        self.option_calls.append(options)
        return self


class RateLimitError(Exception):
    status_code = 429

    def __init__(self, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.response = SimpleNamespace(status_code=429, headers=headers or {})


class PromptBatchTests(unittest.TestCase):
    def test_returns_responses_in_prompt_order_with_bounded_concurrency(self) -> None:
        lock = threading.Lock()
        active = 0
        peak_active = 0
        delays = {"first": 0.12, "second": 0.08, "third": 0.02}

        def respond(*, model, input, reasoning):
            nonlocal active, peak_active
            self.assertEqual(model, "test-model")
            self.assertEqual(reasoning, {"effort": "none"})
            with lock:
                active += 1
                peak_active = max(peak_active, active)
            try:
                time.sleep(delays[input])
                return _Response(input.upper())
            finally:
                with lock:
                    active -= 1

        service = _Service(respond)
        responses = run_prompt_batch(
            ["first", "second", "third"],
            {"service": service, "model": "test-model"},
            max_concurrency=2,
            max_retries=0,
        )

        self.assertEqual(responses, ["FIRST", "SECOND", "THIRD"])
        self.assertEqual(peak_active, 2)
        self.assertEqual(service.option_calls, [{"max_retries": 0}])

    def test_rate_limit_reduces_pressure_retries_and_saves_only_hints(self) -> None:
        calls = 0

        def respond(*, model, input, reasoning):
            nonlocal calls
            self.assertEqual(reasoning, {"effort": "none"})
            calls += 1
            if calls == 1:
                raise RateLimitError("slow down", {"retry-after": "0.001"})
            return _Response("A-104")

        service = _Service(respond)
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "scheduler.json"
            with (
                patch.object(adaptive_openai._AdaptiveProviderCaller, "base_backoff_seconds", 0.0),
                patch.object(adaptive_openai._AdaptiveProviderCaller, "jitter_seconds", 0.0),
            ):
                responses = run_prompt_batch(
                    ["Extract A-104 from private customer text"],
                    {"service": service, "model": "test-model"},
                    max_concurrency=8,
                    max_retries=1,
                    state_path=state_path,
                )

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            serialized = json.dumps(payload)
            restored = adaptive_openai._AdaptiveScheduler.create(
                maximum_concurrency=8,
                model="test-model",
                scope="run-prompt-batch",
                state_path=state_path,
            )

        self.assertEqual(responses, ["A-104"])
        self.assertEqual(calls, 2)
        self.assertEqual(payload["stable_concurrency"], 4)
        self.assertEqual(restored.current_concurrency, 4)
        self.assertNotIn("private customer text", serialized)
        self.assertNotIn("A-104", serialized)
        self.assertNotIn("test-model", serialized)

    def test_simultaneous_rate_limits_count_as_one_pressure_episode(self) -> None:
        scheduler = adaptive_openai._AdaptiveScheduler.create(
            maximum_concurrency=8,
            model="test-model",
            scope="run-prompt-batch",
            state_path=None,
        )
        error = RateLimitError("slow down", {"retry-after": "1"})

        scheduler.observe_backpressure(error, fallback_delay_seconds=1)
        scheduler.observe_backpressure(error, fallback_delay_seconds=1)

        self.assertEqual(scheduler.current_concurrency, 4)
        self.assertEqual(scheduler.cooldown_level, 1)

    def test_clean_exercised_window_increases_capacity_and_launch_frequency(
        self,
    ) -> None:
        scheduler = adaptive_openai._AdaptiveScheduler.create(
            maximum_concurrency=12,
            model="test-model",
            scope="run-prompt-batch",
            state_path=None,
        )
        scheduler.observe_launch(10)

        for _ in range(10):
            scheduler.observe_success()

        self.assertEqual(scheduler.current_concurrency, 11)
        self.assertAlmostEqual(scheduler.current_start_interval_seconds, 0.045)

    def test_insufficient_quota_is_not_retried_as_transient_pressure(self) -> None:
        calls = 0

        def respond(**kwargs):
            nonlocal calls
            calls += 1
            raise RateLimitError("insufficient_quota")

        with self.assertRaisesRegex(RateLimitError, "insufficient_quota"):
            run_prompt_batch(
                ["complete prompt"],
                {"service": _Service(respond), "model": "test-model"},
                max_concurrency=2,
                max_retries=5,
            )

        self.assertEqual(calls, 1)

    def test_terminal_failure_cancels_unlaunched_work(self) -> None:
        calls: list[str] = []
        active = 0
        lock = threading.Lock()
        started = threading.Barrier(2)

        def respond(*, model, input, reasoning):
            nonlocal active
            self.assertEqual(reasoning, {"effort": "none"})
            with lock:
                calls.append(input)
                active += 1
            try:
                started.wait(timeout=1)
                if input == "fail":
                    raise ValueError("terminal")
                time.sleep(0.05)
                return _Response(input)
            finally:
                with lock:
                    active -= 1

        with self.assertRaisesRegex(ValueError, "terminal"):
            run_prompt_batch(
                ["fail", "already-launched", "never-launched"],
                {"service": _Service(respond), "model": "test-model"},
                max_concurrency=2,
                max_retries=0,
            )

        self.assertEqual(set(calls), {"fail", "already-launched"})
        self.assertEqual(active, 0)

    def test_non_text_response_fails_instead_of_becoming_an_empty_result(self) -> None:
        service = _Service(lambda **kwargs: object())

        with self.assertRaisesRegex(TypeError, "must return response text"):
            run_prompt_batch(
                ["complete prompt"],
                {"service": service, "model": "test-model"},
                max_concurrency=1,
                max_retries=2,
            )

    def test_durable_run_persists_successes_and_resumes_missing_inputs(self) -> None:
        prompts = {
            "a": "prompt A",
            "b": "prompt B",
            "c": "prompt C",
        }
        calls: list[str] = []
        fail_b = True

        def respond(*, model, input, reasoning):
            nonlocal fail_b
            self.assertEqual(model, "test-model")
            self.assertEqual(reasoning, {"effort": "none"})
            calls.append(input)
            if input == "prompt B" and fail_b:
                fail_b = False
                raise ValueError("terminal")
            return _Response(input.upper())

        service = _Service(respond)
        with TemporaryDirectory() as directory:
            run_directory = Path(directory) / "durable"
            with self.assertRaisesRegex(ValueError, "terminal"):
                run_prompt_batch(
                    prompts,
                    {"service": service, "model": "test-model"},
                    max_concurrency=1,
                    max_retries=0,
                    run_directory=run_directory,
                )

            self.assertEqual(calls, ["prompt A", "prompt B"])
            self.assertTrue(
                (run_directory / "results" / "000000.json").exists()
            )
            self.assertFalse(
                (run_directory / "results" / "000001.json").exists()
            )

            resumed = run_prompt_batch(
                prompts,
                {"service": service, "model": "test-model"},
                max_concurrency=1,
                max_retries=0,
                run_directory=run_directory,
            )

        self.assertEqual(
            resumed,
            {"a": "PROMPT A", "b": "PROMPT B", "c": "PROMPT C"},
        )
        self.assertEqual(calls, ["prompt A", "prompt B", "prompt B", "prompt C"])

    def test_durable_run_reuses_completed_results_without_provider_calls(self) -> None:
        prompts = {"a": "prompt A", "b": "prompt B"}
        calls: list[str] = []

        def respond(*, input, **kwargs):
            calls.append(input)
            return _Response(input.upper())

        service = _Service(respond)
        with TemporaryDirectory() as directory:
            run_directory = Path(directory) / "durable"
            first = run_prompt_batch(
                prompts,
                {"service": service, "model": "test-model"},
                max_concurrency=2,
                max_retries=1,
                run_directory=run_directory,
            )
            second = run_prompt_batch(
                prompts,
                {"service": service, "model": "test-model"},
                max_concurrency=2,
                max_retries=1,
                run_directory=run_directory,
            )

            manifest = json.loads(
                (run_directory / "run_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(first, second)
        self.assertEqual(calls, ["prompt A", "prompt B"])
        serialized_manifest = json.dumps(manifest)
        self.assertNotIn("prompt A", serialized_manifest)
        self.assertNotIn("prompt B", serialized_manifest)

    def test_durable_run_rejects_changed_identity_before_provider_calls(self) -> None:
        prompts = {"a": "prompt A"}
        calls = 0

        def respond(**kwargs):
            nonlocal calls
            calls += 1
            return _Response("A")

        service = _Service(respond)
        with TemporaryDirectory() as directory:
            run_directory = Path(directory) / "durable"
            llm = {"service": service, "model": "test-model"}
            run_prompt_batch(
                prompts,
                llm,
                max_concurrency=1,
                max_retries=0,
                run_directory=run_directory,
            )

            with self.assertRaisesRegex(ValueError, "conflicts"):
                run_prompt_batch(
                    {"a": "changed prompt"},
                    llm,
                    max_concurrency=1,
                    max_retries=0,
                    run_directory=run_directory,
                )
            with self.assertRaisesRegex(ValueError, "conflicts"):
                run_prompt_batch(
                    prompts,
                    llm,
                    max_concurrency=1,
                    max_retries=1,
                    run_directory=run_directory,
                )

        self.assertEqual(calls, 1)

    def test_durable_run_requires_stable_id_mapping(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TypeError, "stable input IDs"):
                run_prompt_batch(
                    ["prompt"],
                    {"service": _Service(lambda **kwargs: _Response("ok")), "model": "test-model"},
                    max_concurrency=1,
                    max_retries=0,
                    run_directory=Path(directory) / "durable",
                )

    def test_empty_batch_returns_without_provider_calls(self) -> None:
        calls = 0

        def respond(**kwargs):
            nonlocal calls
            calls += 1
            return _Response("unused")

        responses = run_prompt_batch(
            [],
            {"service": _Service(respond), "model": "test-model"},
            max_concurrency=2,
            max_retries=0,
        )

        self.assertEqual(responses, [])
        self.assertEqual(calls, 0)

    def test_rejects_invalid_batch_policy_before_work(self) -> None:
        service = _Service(lambda **kwargs: _Response("unused"))
        llm = {"service": service, "model": "test-model"}

        with self.assertRaisesRegex(ValueError, "max_concurrency"):
            run_prompt_batch(["prompt"], llm, max_concurrency=0, max_retries=0)
        with self.assertRaisesRegex(ValueError, "max_retries"):
            run_prompt_batch(["prompt"], llm, max_concurrency=1, max_retries=-1)
        with self.assertRaisesRegex(ValueError, "nonblank"):
            run_prompt_batch([""], llm, max_concurrency=1, max_retries=0)


if __name__ == "__main__":
    unittest.main()
