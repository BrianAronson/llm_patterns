import json
import unittest
from threading import Lock
from types import SimpleNamespace

from llm_patterns import StructuredRecordError, generate_structured_records


def _schema_plan(questions, types=None, nullable=()):
    types = types or {}
    return json.dumps(
        {
            "fields": [
                {
                    "field_name": field_name,
                    "value_type": types.get(field_name, "string"),
                    "nullable": field_name in nullable,
                    "format_instruction": f"Return one {types.get(field_name, 'string')} value.",
                }
                for field_name in questions
            ]
        }
    )


class _FakeResponses:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.lock = Lock()

    def parse(self, **request):
        with self.lock:
            self.calls.append(request)
            response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(output_text=response)


class _FakeService:
    def __init__(self, responses):
        self.responses = _FakeResponses(responses)
        self.option_calls = []

    def with_options(self, **options):
        self.option_calls.append(options)
        return self


class StructuredRecordTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "lesson-101": "A beginner spreadsheet formulas lesson.",
            "lesson-201": "An intermediate pandas data-cleaning lesson.",
        }
        self.questions = {
            "title": "What should the exercise be called?",
            "duration_minutes": "How many whole minutes should it take?",
            "learning_objectives": "Which skills should the learner gain?",
        }
        self.types = {
            "title": "string",
            "duration_minutes": "integer",
            "learning_objectives": "string_list",
        }
        self.instructions = "Create one coherent practical exercise."

    def test_infers_one_shared_schema_and_reuses_it_for_original_ids(self):
        service = _FakeService(
            [
                _schema_plan(self.questions, self.types),
                '{"title":"Formula basics","duration_minutes":20,"learning_objectives":["Use SUM"]}',
                '{"title":"Clean missing values","duration_minutes":40,"learning_objectives":["Handle nulls"]}',
            ]
        )

        records = generate_structured_records(
            context=self.context,
            questions=self.questions,
            instructions=self.instructions,
            llm={"service": service, "model": "gpt-5.6-luna"},
        )

        self.assertEqual(list(records), list(self.context))
        self.assertEqual(
            records["lesson-101"],
            {
                "title": "Formula basics",
                "duration_minutes": 20,
                "learning_objectives": ["Use SUM"],
            },
        )
        self.assertEqual(len(service.responses.calls), 3)
        schema_request, first_record_request, second_record_request = (
            service.responses.calls
        )
        self.assertEqual(schema_request["model"], "gpt-5.6-luna")
        self.assertEqual(schema_request["reasoning"], {"effort": "none"})
        self.assertIn(self.instructions, schema_request["input"][-1]["content"])
        self.assertIn("duration_minutes", schema_request["input"][-1]["content"])
        self.assertNotIn(self.context["lesson-101"], schema_request["input"][-1]["content"])
        output_schema = first_record_request["text_format"].model_json_schema()
        self.assertEqual(list(output_schema["properties"]), list(self.questions))
        self.assertEqual(output_schema["properties"]["title"]["type"], "string")
        self.assertEqual(
            output_schema["properties"]["duration_minutes"]["type"], "integer"
        )
        self.assertEqual(
            output_schema["properties"]["learning_objectives"]["type"], "array"
        )
        self.assertFalse(output_schema["additionalProperties"])
        self.assertIs(
            first_record_request["text_format"],
            second_record_request["text_format"],
        )

    def test_question_fields_can_be_human_readable_and_nullable(self):
        questions = {
            "Short title": "What should the exercise be called?",
            "Prerequisite": "What prior knowledge is required, or null?",
        }
        service = _FakeService(
            [
                _schema_plan(questions, nullable={"Prerequisite"}),
                '{"Short title":"Formula basics","Prerequisite":null}',
            ]
        )

        records = generate_structured_records(
            context={"lesson-101": self.context["lesson-101"]},
            questions=questions,
            instructions=self.instructions,
            llm={"service": service, "model": "test-model"},
        )

        self.assertEqual(
            records["lesson-101"],
            {"Short title": "Formula basics", "Prerequisite": None},
        )

    def test_invalid_schema_plan_gets_one_contextual_correction(self):
        invalid_questions = {
            "title": self.questions["title"],
            "unexpected": "An invented field",
        }
        service = _FakeService(
            [
                _schema_plan(invalid_questions),
                _schema_plan(self.questions, self.types),
                '{"title":"Formula basics","duration_minutes":20,"learning_objectives":["Use SUM"]}',
            ]
        )

        generate_structured_records(
            context={"lesson-101": self.context["lesson-101"]},
            questions=self.questions,
            instructions=self.instructions,
            llm={"service": service, "model": "test-model"},
        )

        recovery_messages = service.responses.calls[1]["input"]
        self.assertEqual(
            [message["role"] for message in recovery_messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertIn("duration_minutes", recovery_messages[-1]["content"])
        self.assertIn("unexpected", recovery_messages[-1]["content"])

    def test_missing_or_extra_record_fields_get_one_contextual_correction(self):
        service = _FakeService(
            [
                _schema_plan(self.questions, self.types),
                '{"title":"Formula basics","duration_minutes":20,"extra":"remove me"}',
                '{"title":"Formula basics","duration_minutes":20,"learning_objectives":["Use SUM"]}',
            ]
        )

        records = generate_structured_records(
            context={"lesson-101": self.context["lesson-101"]},
            questions=self.questions,
            instructions=self.instructions,
            llm={"service": service, "model": "test-model"},
        )

        self.assertEqual(set(records["lesson-101"]), set(self.questions))
        recovery_messages = service.responses.calls[2]["input"]
        self.assertEqual(
            [message["role"] for message in recovery_messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertIn("learning_objectives", recovery_messages[-1]["content"])
        self.assertIn("extra", recovery_messages[-1]["content"])

    def test_final_invalid_record_names_the_input_and_bounded_attempts(self):
        invalid = '{"title":"Formula basics"}'
        service = _FakeService(
            [_schema_plan(self.questions, self.types), invalid, invalid]
        )

        with self.assertRaises(StructuredRecordError) as caught:
            generate_structured_records(
                context={"lesson-101": self.context["lesson-101"]},
                questions=self.questions,
                instructions=self.instructions,
                llm={"service": service, "model": "test-model"},
            )

        self.assertEqual(caught.exception.input_id, "lesson-101")
        self.assertEqual(len(caught.exception.attempts), 2)
        self.assertIn("lesson-101", str(caught.exception))
        self.assertIn("duration_minutes", str(caught.exception))

    def test_final_invalid_schema_plan_fails_before_record_calls(self):
        invalid_plan = _schema_plan({"wrong": "An invented field"})
        service = _FakeService([invalid_plan, invalid_plan])

        with self.assertRaisesRegex(ValueError, "shared record schema"):
            generate_structured_records(
                context={"lesson-101": self.context["lesson-101"]},
                questions=self.questions,
                instructions=self.instructions,
                llm={"service": service, "model": "test-model"},
            )

        self.assertEqual(len(service.responses.calls), 2)

    def test_batch_mode_uses_adaptive_calls_and_preserves_input_order(self):
        service = _FakeService(
            [
                _schema_plan(self.questions, self.types),
                '{"title":"Formula basics","duration_minutes":20,"learning_objectives":["Use SUM"]}',
                '{"title":"Clean missing values","duration_minutes":40,"learning_objectives":["Handle nulls"]}',
            ]
        )

        records = generate_structured_records(
            context=self.context,
            questions=self.questions,
            instructions=self.instructions,
            llm={"service": service, "model": "test-model"},
            batch=True,
        )

        self.assertEqual(list(records), list(self.context))
        self.assertEqual(service.option_calls, [{"max_retries": 0}])
        self.assertEqual(len(service.responses.calls), 3)

    def test_transport_failure_is_not_treated_as_record_validation(self):
        service = _FakeService(
            [_schema_plan(self.questions, self.types), RuntimeError("provider unavailable")]
        )

        with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
            generate_structured_records(
                context={"lesson-101": self.context["lesson-101"]},
                questions=self.questions,
                instructions=self.instructions,
                llm={"service": service, "model": "test-model"},
            )

    def test_invalid_inputs_fail_before_provider_work(self):
        service = _FakeService([])
        llm = {"service": service, "model": "test-model"}
        invalid_calls = (
            ({}, self.questions, self.instructions),
            ({" bad ": "Subject"}, self.questions, self.instructions),
            ({"id": ""}, self.questions, self.instructions),
            ({"id": "Subject"}, {}, self.instructions),
            ({"id": "Subject"}, {" bad ": "Question"}, self.instructions),
            ({"id": "Subject"}, {"field": ""}, self.instructions),
            ({"id": "Subject"}, self.questions, ""),
        )

        for context, questions, instructions in invalid_calls:
            with self.subTest(context=context, questions=questions):
                with self.assertRaises((TypeError, ValueError)):
                    generate_structured_records(
                        context=context,
                        questions=questions,
                        instructions=instructions,
                        llm=llm,
                    )

        self.assertEqual(service.responses.calls, [])


if __name__ == "__main__":
    unittest.main()
