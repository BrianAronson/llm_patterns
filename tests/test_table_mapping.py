import unittest
from types import SimpleNamespace

from llm_patterns import map_table_columns


class _Responses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def parse(self, **request):
        self.calls.append(request)
        return SimpleNamespace(output_text='{"id":"ticket_no","text":"full_notes"}')


class _Service:
    def __init__(self) -> None:
        self.responses = _Responses()


class TableMappingTests(unittest.TestCase):
    def test_uses_luna_without_reasoning_tokens_and_returns_valid_mapping(self) -> None:
        service = _Service()

        mapping = map_table_columns(
            source_columns=["ticket_no", "full_notes", "created_at"],
            target_fields={
                "id": "Unique identifier for each record",
                "text": "Main text to analyze",
            },
            llm={"service": service, "model": "gpt-5.6-luna"},
        )

        self.assertEqual(mapping, {"id": "ticket_no", "text": "full_notes"})
        self.assertEqual(service.responses.calls[0]["model"], "gpt-5.6-luna")
        self.assertEqual(
            service.responses.calls[0]["reasoning"],
            {"effort": "none"},
        )


if __name__ == "__main__":
    unittest.main()
