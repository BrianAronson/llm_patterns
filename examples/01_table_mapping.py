# 0) Imports
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from llm_patterns import map_table_columns

# 1) Constants
load_dotenv()
llm = {"service": OpenAI(), "model": "gpt-5.6-luna"}
target_fields = {
    "id": "Unique identifier for each record",
    "text": "Main text to analyze",
}

# 2) Examples

# a) Via source columns alone
source_column_names = ["ticket_no", "subject_line", "full_notes", "created_at"]
mapping_from_names = map_table_columns(
    source_columns=source_column_names,
    target_fields=target_fields,
    llm=llm,
)
print(mapping_from_names)
# Expected output:
# {'id': 'ticket_no', 'text': 'full_notes'}

# b) Via source columns and examples
source_columns_with_examples = {
    "ticket_no": ["T-104", "T-105"],
    "subject_line": ["Broken hydrant", "Missed inspection"],
    "full_notes": ["Resident reported a leak.", "Inspector found no access."],
    "created_at": ["2026-08-01", "2026-08-02"],
}
mapping_from_examples = map_table_columns(
    source_columns=source_columns_with_examples,
    target_fields=target_fields,
    llm=llm,
)
print(mapping_from_examples)
# Expected output:
# {'id': 'ticket_no', 'text': 'full_notes'}

# c) Via dataframe
source_dataframe = pd.DataFrame(source_columns_with_examples)
mapping_from_dataframe = map_table_columns(
    source_columns=source_dataframe,
    target_fields=target_fields,
    llm=llm,
)
print(mapping_from_dataframe)
# Expected output:
# {'id': 'ticket_no', 'text': 'full_notes'}
