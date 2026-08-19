import unittest
from types import SimpleNamespace

from llm_patterns import SelectedLineEditError, edit_selected_lines


class _FakeResponses:
    def __init__(self, compact: str, structured=()) -> None:
        self.compact = compact
        self.structured = iter(structured)
        self.create_calls: list[dict[str, object]] = []
        self.parse_calls: list[dict[str, object]] = []

    def create(self, **request):
        self.create_calls.append(request)
        return SimpleNamespace(output_text=self.compact)

    def parse(self, **request):
        self.parse_calls.append(request)
        return SimpleNamespace(output_text=next(self.structured))


class _FakeService:
    def __init__(self, compact: str, structured=()) -> None:
        self.responses = _FakeResponses(compact, structured)


class EditSelectedLinesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = (
            "# Service report\n"
            "- Impact: 23 requests failed.\n"
            "- Status: The problem is done.\n"
            "- Next step: Monitor error rates for 24 hours.\n"
        )
        self.instruction = "Clarify the status using the supplied context."
        self.context = "Service was restored at 14:20 UTC."

    def test_uses_the_document_as_context_and_applies_one_exact_line(self) -> None:
        service = _FakeService('{"e1":"- Status: Service was restored at 14:20 UTC."}')

        edited = edit_selected_lines(
            self.document,
            ["- Status: The problem is done."],
            self.instruction,
            {"service": service, "model": "gpt-5.6-luna"},
            context=self.context,
        )

        self.assertEqual(
            edited,
            self.document.replace(
                "- Status: The problem is done.",
                "- Status: Service was restored at 14:20 UTC.",
            ),
        )
        self.assertEqual(service.responses.parse_calls, [])
        request = service.responses.create_calls[0]
        self.assertEqual(request["model"], "gpt-5.6-luna")
        self.assertEqual(request["reasoning"], {"effort": "none"})
        prompt = request["input"][-1]["content"]
        self.assertIn("- Status: The problem is done.", prompt)
        self.assertIn(self.context, prompt)
        self.assertIn("23 requests failed", prompt)
        self.assertIn("Monitor error rates", prompt)
        self.assertIn("Document:", prompt)

    def test_empty_selection_returns_original_without_provider_work(self) -> None:
        service = _FakeService("not used")

        edited = edit_selected_lines(
            self.document,
            [],
            self.instruction,
            {"service": service, "model": "test-model"},
        )

        self.assertEqual(edited, self.document)
        self.assertEqual(service.responses.create_calls, [])
        self.assertEqual(service.responses.parse_calls, [])

    def test_invalid_compact_mapping_uses_structured_recovery(self) -> None:
        service = _FakeService(
            '{"e2":"- Status: wrong target"}',
            ['{"e1":"- Status: Service was restored at 14:20 UTC."}'],
        )

        edited = edit_selected_lines(
            self.document,
            ["- Status: The problem is done."],
            self.instruction,
            {"service": service, "model": "test-model"},
            context=self.context,
        )

        self.assertIn("- Status: Service was restored at 14:20 UTC.", edited)
        self.assertEqual(len(service.responses.create_calls), 1)
        self.assertEqual(len(service.responses.parse_calls), 1)
        recovery = service.responses.parse_calls[0]
        self.assertEqual(recovery["text_format"].__name__, "SelectedLineReplacements")
        self.assertIn(
            "Extra inputs are not permitted", recovery["input"][-1]["content"]
        )

    def test_multiline_and_markdown_changes_are_rejected_before_application(
        self,
    ) -> None:
        service = _FakeService(
            '{"e1":"Status: changed shape"}',
            [
                '{"e1":"- Status: first line\\nsecond line"}',
                '{"e1":"- Status: still invalid\\nsecond line"}',
            ],
        )

        with self.assertRaises(SelectedLineEditError) as caught:
            edit_selected_lines(
                self.document,
                ["- Status: The problem is done."],
                self.instruction,
                {"service": service, "model": "test-model"},
            )

        self.assertEqual(len(caught.exception.attempts), 3)
        self.assertIn("Markdown prefix", caught.exception.attempts[0].errors[0])
        self.assertIn("remain one line", caught.exception.attempts[-1].errors[0])

    def test_multiple_selected_lines_are_applied_in_one_pass(self) -> None:
        service = _FakeService(
            '{"e1":"- Status: Service was restored at 14:20 UTC.",'
            '"e2":"- Impact: 23 requests failed during deployment."}'
        )

        edited = edit_selected_lines(
            self.document,
            [
                "- Status: The problem is done.",
                "- Impact: 23 requests failed.",
            ],
            self.instruction,
            {"service": service, "model": "test-model"},
        )

        self.assertIn("- Impact: 23 requests failed during deployment.\n", edited)
        self.assertIn("- Status: Service was restored at 14:20 UTC.\n", edited)

    def test_preserves_crlf_and_missing_final_newline(self) -> None:
        document = "# Update\r\n- Status: unclear"
        service = _FakeService('{"e1":"- Status: restored"}')

        edited = edit_selected_lines(
            document,
            ["- Status: unclear"],
            self.instruction,
            {"service": service, "model": "test-model"},
        )

        self.assertEqual(edited, "# Update\r\n- Status: restored")

    def test_invalid_selection_fails_before_provider_work(self) -> None:
        invalid_selections = (
            "- Status: The problem is done.",
            [3],
            ["- Missing: line"],
            ["- Status: The problem is done.", "- Status: The problem is done."],
            ["- Status: first line\nsecond line"],
        )
        for selected_lines in invalid_selections:
            service = _FakeService("not used")
            with (
                self.subTest(selected_lines=selected_lines),
                self.assertRaises((TypeError, ValueError)),
            ):
                edit_selected_lines(
                    self.document,
                    selected_lines,
                    self.instruction,
                    {"service": service, "model": "test-model"},
                )
            self.assertEqual(service.responses.create_calls, [])

    def test_blank_selected_line_fails_before_provider_work(self) -> None:
        service = _FakeService("not used")

        with self.assertRaisesRegex(ValueError, "must not contain blank text"):
            edit_selected_lines(
                "# Update\n\n- Status: restored\n",
                [""],
                self.instruction,
                {"service": service, "model": "test-model"},
            )

        self.assertEqual(service.responses.create_calls, [])

    def test_repeated_selected_text_is_rejected_as_ambiguous(self) -> None:
        service = _FakeService("not used")

        with self.assertRaisesRegex(ValueError, "occurs more than once"):
            edit_selected_lines(
                "# Update\n- Status: unclear\n- Status: unclear\n",
                ["- Status: unclear"],
                self.instruction,
                {"service": service, "model": "test-model"},
            )

        self.assertEqual(service.responses.create_calls, [])


if __name__ == "__main__":
    unittest.main()
