import unittest
from pathlib import Path

from llm_patterns._adaptive_openai import _batch_config


class AdaptiveOpenAIBatchConfigTests(unittest.TestCase):
    def test_false_disables_batch_mode(self) -> None:
        self.assertIsNone(_batch_config(False))

    def test_true_uses_documented_defaults_without_writing_state(self) -> None:
        config = _batch_config(True)

        self.assertIsNotNone(config)
        if config is None:
            self.fail("batch=True did not produce a configuration")
        self.assertEqual(config.max_concurrency, 100)
        self.assertEqual(config.max_retries, 5)
        self.assertIsNone(config.state_path)

    def test_dictionary_overrides_only_supplied_defaults(self) -> None:
        config = _batch_config(
            {
                "max_concurrency": 40,
                "state_path": "artifacts/concurrency.json",
            }
        )

        self.assertIsNotNone(config)
        if config is None:
            self.fail("batch options did not produce a configuration")
        self.assertEqual(config.max_concurrency, 40)
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.state_path, Path("artifacts/concurrency.json"))


if __name__ == "__main__":
    unittest.main()
