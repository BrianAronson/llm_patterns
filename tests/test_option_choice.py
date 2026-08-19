import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_patterns import OptionChoiceError, choose_from_options
from llm_patterns import _adaptive_openai as adaptive_openai


class _FakeResponses:
    def __init__(self, compact, structured=()):
        self.compact = compact
        self.structured = iter(structured)
        self.create_calls = []
        self.parse_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if isinstance(self.compact, Exception):
            raise self.compact
        return SimpleNamespace(output_text=self.compact)

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        response = next(self.structured)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(output_text=response)


class _FakeService:
    def __init__(self, compact, structured=()):
        self.responses = _FakeResponses(compact, structured)
        self.option_calls = []

    def with_options(self, **options):
        self.option_calls.append(options)
        return self


class RateLimitError(Exception):
    status_code = 429

    def __init__(self) -> None:
        super().__init__("slow down")
        self.response = SimpleNamespace(
            status_code=429,
            headers={"retry-after": "0.001"},
        )


class ChooseFromOptionsTests(unittest.TestCase):
    def setUp(self):
        self.options = {
            "direct": "Scheduled maintenance Sunday at 2:00 AM",
            "promotional": "Big improvements are coming. Don't miss out!",
            "vague": "Important update for your account",
        }
        self.criteria = (
            "Choose the clearest accurate subject line. "
            "Do not reward vague or promotional wording."
        )

    def test_compact_number_resolves_to_stable_id_in_one_call(self):
        service = _FakeService("1")

        choice = choose_from_options(
            "Tell customers about scheduled maintenance Sunday at 2:00 AM.",
            self.options,
            self.criteria,
            {"service": service, "model": "test-model"},
        )

        self.assertEqual(choice, "direct")
        self.assertEqual(len(service.responses.create_calls), 1)
        self.assertEqual(service.responses.parse_calls, [])
        request = service.responses.create_calls[0]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["reasoning"], {"effort": "none"})
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["max_output_tokens"], 32)
        prompt = request["input"][-1]["content"]
        self.assertIn("1:\n  Scheduled maintenance", prompt)
        self.assertIn("3:\n  Important update", prompt)
        self.assertNotIn("direct", prompt)

    def test_exact_echoed_option_avoids_unnecessary_recovery(self):
        service = _FakeService("1. Scheduled maintenance Sunday at 2:00 AM")

        choice = choose_from_options(
            "Tell customers about scheduled maintenance Sunday at 2:00 AM.",
            self.options,
            self.criteria,
            {"service": service, "model": "test-model"},
        )

        self.assertEqual(choice, "direct")
        self.assertEqual(service.responses.parse_calls, [])

    def test_invalid_compact_response_uses_structured_recovery(self):
        service = _FakeService("The first one", ['{"choice":1}'])

        choice = choose_from_options(
            "Tell customers about scheduled maintenance Sunday at 2:00 AM.",
            self.options,
            self.criteria,
            {"service": service, "model": "test-model"},
        )

        self.assertEqual(choice, "direct")
        self.assertEqual(len(service.responses.create_calls), 1)
        self.assertEqual(len(service.responses.parse_calls), 1)
        recovery = service.responses.parse_calls[0]
        self.assertEqual(recovery["text_format"].__name__, "_StructuredOptionChoice")
        self.assertEqual(recovery["max_output_tokens"], 64)
        prompt = recovery["input"][-1]["content"]
        self.assertIn("Selection criteria:\n" + self.criteria, prompt)
        self.assertNotIn("The first one", prompt)

    def test_second_structured_attempt_can_recover(self):
        service = _FakeService(
            "not a choice",
            ['{"choice":9}', '{"choice":1}'],
        )

        choice = choose_from_options(
            "Tell customers about scheduled maintenance Sunday at 2:00 AM.",
            self.options,
            self.criteria,
            {"service": service, "model": "test-model"},
        )

        self.assertEqual(choice, "direct")
        self.assertEqual(len(service.responses.parse_calls), 2)

    def test_final_invalid_response_reports_all_bounded_attempts(self):
        service = _FakeService(
            "not a choice",
            ['{"choice":9}', '{"choice":0}'],
        )

        with self.assertRaises(OptionChoiceError) as caught:
            choose_from_options(
                "Tell customers about scheduled maintenance Sunday at 2:00 AM.",
                self.options,
                self.criteria,
                {"service": service, "model": "test-model"},
            )

        self.assertEqual(len(caught.exception.attempts), 3)
        self.assertEqual(len(service.responses.create_calls), 1)
        self.assertEqual(len(service.responses.parse_calls), 2)

    def test_transport_failure_is_not_silently_retried(self):
        service = _FakeService(RuntimeError("provider unavailable"))

        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            choose_from_options(
                "Tell customers about scheduled maintenance Sunday at 2:00 AM.",
                self.options,
                self.criteria,
                {"service": service, "model": "test-model"},
            )

        self.assertEqual(service.responses.parse_calls, [])

    def test_batch_mode_returns_one_choice_for_every_stable_input_id(self):
        service = _FakeService("1")
        inputs = {
            "message-104": "Tell customers about scheduled maintenance Sunday.",
            "message-105": "Tell staff about scheduled maintenance Monday.",
        }

        choices = choose_from_options(
            inputs,
            self.options,
            self.criteria,
            {"service": service, "model": "test-model"},
            batch=True,
        )

        self.assertEqual(
            choices,
            {"message-104": "direct", "message-105": "direct"},
        )
        self.assertEqual(len(service.responses.create_calls), 2)
        self.assertEqual(service.option_calls, [{"max_retries": 0}])

    def test_batch_retries_only_the_provider_call_that_hit_rate_pressure(self):
        service = _FakeService(
            "not a choice",
            [RateLimitError(), '{"choice":1}'],
        )
        with (
            patch.object(adaptive_openai._AdaptiveProviderCaller, "base_backoff_seconds", 0.0),
            patch.object(adaptive_openai._AdaptiveProviderCaller, "jitter_seconds", 0.0),
        ):
            choices = choose_from_options(
                {"message-104": "Tell customers about scheduled maintenance Sunday."},
                self.options,
                self.criteria,
                {"service": service, "model": "test-model"},
                batch={"max_concurrency": 4, "max_retries": 1},
            )

        self.assertEqual(choices, {"message-104": "direct"})
        self.assertEqual(len(service.responses.create_calls), 1)
        self.assertEqual(len(service.responses.parse_calls), 2)

    def test_batch_mode_rejects_ambiguous_inputs_and_unknown_policy(self):
        service = _FakeService("1")
        llm = {"service": service, "model": "test-model"}

        with self.assertRaisesRegex(TypeError, "mapping of stable input IDs"):
            choose_from_options(
                "single input",
                self.options,
                self.criteria,
                llm,
                batch=True,
            )
        with self.assertRaisesRegex(ValueError, "unsupported batch options"):
            choose_from_options(
                {"message-104": "Input"},
                self.options,
                self.criteria,
                llm,
                batch={"workers": 4},
            )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            choose_from_options(
                {},
                self.options,
                self.criteria,
                llm,
                batch=True,
            )

        self.assertEqual(service.responses.create_calls, [])

    def test_invalid_inputs_fail_before_provider_work(self):
        service = _FakeService("1")
        invalid_calls = (
            ("", self.options, self.criteria),
            ("Input", {}, self.criteria),
            ("Input", {" bad ": "Option"}, self.criteria),
            ("Input", {"a": "Same", "b": "same"}, self.criteria),
            ("Input", self.options, ""),
        )

        for input_text, options, criteria in invalid_calls:
            with (
                self.subTest(input_text=input_text, options=options, criteria=criteria),
                self.assertRaises(ValueError),
            ):
                choose_from_options(
                    input_text,
                    options,
                    criteria,
                    {"service": service, "model": "test-model"},
                )

        self.assertEqual(service.responses.create_calls, [])


if __name__ == "__main__":
    unittest.main()
