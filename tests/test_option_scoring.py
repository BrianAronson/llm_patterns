import unittest
from types import SimpleNamespace

from llm_patterns import OptionScoringError, score_options


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
        return SimpleNamespace(output_text=next(self.structured))


class _FakeService:
    def __init__(self, compact, structured=()):
        self.responses = _FakeResponses(compact, structured)
        self.option_calls = []

    def with_options(self, **options):
        self.option_calls.append(options)
        return self


class ScoreOptionsTests(unittest.TestCase):
    def setUp(self):
        self.options = {
            "roads": "Road and pavement repair",
            "water": "Hydrants, water mains, and leaks",
            "trees": "Fallen trees and dangerous branches",
        }
        self.rubric = (
            "5 = direct fit; 4 = good fit; 3 = plausible; "
            "2 = poor fit; 1 = clearly wrong."
        )

    def test_compact_success_scores_every_stable_option_in_one_call(self):
        service = _FakeService('{"c1":2,"c2":5,"c3":1}')

        scores = score_options(
            "A hydrant has been leaking since this morning.",
            self.options,
            self.rubric,
            {"service": service, "model": "test-model"},
        )

        self.assertEqual(scores, {"roads": 2, "water": 5, "trees": 1})
        self.assertEqual(len(service.responses.create_calls), 1)
        self.assertEqual(service.responses.parse_calls, [])
        request = service.responses.create_calls[0]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["reasoning"], {"effort": "none"})
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["max_output_tokens"], 128)
        prompt = request["input"][-1]["content"]
        self.assertIn("c1:\n  Road and pavement repair", prompt)
        self.assertIn("c3:\n  Fallen trees and dangerous branches", prompt)

    def test_invalid_compact_response_uses_one_structured_recovery(self):
        service = _FakeService(
            '{"c1":2,"c2":5}',
            ['{"scores":[2,5,1]}'],
        )

        scores = score_options(
            "A hydrant has been leaking since this morning.",
            self.options,
            self.rubric,
            {"service": service, "model": "test-model"},
        )

        self.assertEqual(scores, {"roads": 2, "water": 5, "trees": 1})
        self.assertEqual(len(service.responses.create_calls), 1)
        self.assertEqual(len(service.responses.parse_calls), 1)
        recovery = service.responses.parse_calls[0]
        self.assertEqual(recovery["text_format"].__name__, "_StructuredOptionScores")
        prompt = recovery["input"][-1]["content"]
        self.assertIn("Position 1:\n  Road and pavement repair", prompt)
        self.assertNotIn('{"c1":2,"c2":5}', prompt)

    def test_second_structured_attempt_is_the_final_recovery(self):
        service = _FakeService(
            "not json",
            ['{"scores":[5]}', '{"scores":[2,5,1]}'],
        )

        scores = score_options(
            "A hydrant has been leaking since this morning.",
            self.options,
            self.rubric,
            {"service": service, "model": "test-model"},
        )

        self.assertEqual(scores["water"], 5)
        self.assertEqual(len(service.responses.parse_calls), 2)

    def test_final_invalid_response_reports_all_bounded_attempts(self):
        service = _FakeService(
            "not json",
            ['{"scores":[5]}', '{"scores":[8,8,8]}'],
        )

        with self.assertRaises(OptionScoringError) as caught:
            score_options(
                "A hydrant has been leaking since this morning.",
                self.options,
                self.rubric,
                {"service": service, "model": "test-model"},
            )

        self.assertEqual(len(caught.exception.attempts), 3)
        self.assertEqual(len(service.responses.create_calls), 1)
        self.assertEqual(len(service.responses.parse_calls), 2)

    def test_transport_failure_is_not_silently_retried(self):
        service = _FakeService(RuntimeError("provider unavailable"))

        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            score_options(
                "A hydrant has been leaking since this morning.",
                self.options,
                self.rubric,
                {"service": service, "model": "test-model"},
            )

        self.assertEqual(service.responses.parse_calls, [])

    def test_batch_mode_returns_complete_scores_for_every_stable_input_id(self):
        service = _FakeService('{"c1":2,"c2":5,"c3":1}')
        inputs = {
            "report-104": "A hydrant has been leaking since this morning.",
            "report-105": "Water is bubbling through a crack in the street.",
        }

        scores = score_options(
            inputs,
            self.options,
            self.rubric,
            {"service": service, "model": "test-model"},
            batch={"max_concurrency": 8},
        )

        expected = {"roads": 2, "water": 5, "trees": 1}
        self.assertEqual(
            scores,
            {"report-104": expected, "report-105": expected},
        )
        self.assertEqual(len(service.responses.create_calls), 2)
        self.assertEqual(service.option_calls, [{"max_retries": 0}])

    def test_invalid_inputs_fail_before_provider_work(self):
        service = _FakeService('{"c1":5}')

        invalid_calls = (
            ("", self.options, self.rubric),
            ("Input", {}, self.rubric),
            ("Input", {" blank ": "Description"}, self.rubric),
            ("Input", self.options, ""),
        )
        for input_text, options, rubric in invalid_calls:
            with (
                self.subTest(input_text=input_text, options=options, rubric=rubric),
                self.assertRaises(ValueError),
            ):
                score_options(
                    input_text,
                    options,
                    rubric,
                    {"service": service, "model": "test-model"},
                )

        self.assertEqual(service.responses.create_calls, [])


if __name__ == "__main__":
    unittest.main()
