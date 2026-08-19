"""Deterministic boundaries around common LLM operations."""

from .bounded_text_edits import SelectedLineEditError, edit_selected_lines
from .openai_prompt_batch import (
    AmbiguousBatchSubmissionError,
    BatchPromptResult,
    OpenAIPromptBatchCollection,
    OpenAIPromptBatchSubmission,
    collect_openai_prompt_batch,
    submit_openai_prompt_batch,
)
from .option_choice import OptionChoiceError, choose_from_options
from .option_scoring import OptionScoringError, score_options
from .prompt_batch import run_prompt_batch
from .structured_records import StructuredRecordError, generate_structured_records
from .table_mapping import TableMappingError, map_table_columns

__all__ = [
    "AmbiguousBatchSubmissionError",
    "BatchPromptResult",
    "OpenAIPromptBatchCollection",
    "OpenAIPromptBatchSubmission",
    "OptionChoiceError",
    "OptionScoringError",
    "SelectedLineEditError",
    "StructuredRecordError",
    "TableMappingError",
    "choose_from_options",
    "collect_openai_prompt_batch",
    "edit_selected_lines",
    "generate_structured_records",
    "map_table_columns",
    "run_prompt_batch",
    "score_options",
    "submit_openai_prompt_batch",
]
