# Generate data rows by asking the same questions about each subject

You are building a training catalog from a collection of informal lesson briefs. Each row needs the same fields, but defining and maintaining a Pydantic class for a one-off dataset feels heavier than the task itself. Writing a separate prompt for every cell is also repetitive and can produce answers that do not belong together.

`generate_structured_records(...)` accepts a `context` mapping from stable row IDs to the text the model should use for each row, plus a plain `questions` dictionary. Each question key becomes a column. Before processing the rows, one LLM call chooses a value type and format for every column, and the helper turns that plan into a Pydantic model. The same model is reused for every row. Field names, types, and stable IDs are checked locally. Whether the inferred format is a good choice, and whether the answers make sense together, still depends on the model. Invalid schema plans and invalid records each receive at most one correction.

Records run sequentially by default. Set `batch=True` to use adaptive concurrency and bounded transient retries, or pass a dictionary to override those defaults. The single schema call happens before either path, so its cost is shared by the whole collection.

The shape is inferred once and then reused: `shared questions + instructions → one Pydantic schema → one record per subject → nested dictionary ready for rows`.

```python
# 0) Imports
from dotenv import load_dotenv
from openai import OpenAI
from llm_patterns import generate_structured_records

# 1) Constants
load_dotenv()
llm = {"service": OpenAI(), "model": "gpt-5.6-luna"}
questions = {
    "title": "What exact exercise title is stated in the brief?",
    "difficulty": "What difficulty is stated in the brief?",
    "duration_minutes": "How many whole minutes are stated in the brief?",
    "prerequisite": "What prerequisite is stated in the brief? Return null when it says None.",
    "learning_objectives": "What two learning objectives are stated in the brief? Return them as a list.",
}
instructions = "Turn each brief into one coherent training-catalog row. Copy the stated values without embellishing them."
context = {
    "spreadsheet-formulas": "Title: Formula Foundations. Difficulty: beginner. Duration: 20 minutes. Prerequisite: None. Learning objectives: Write a SUM formula; Copy it down a column.",
    "python-cleaning": "Title: Cleaning Missing Values. Difficulty: intermediate. Duration: 40 minutes. Prerequisite: Basic Python syntax. Learning objectives: Find null values; Fill or remove them.",
}

# 2) Examples

# a) Via sequential calls
sequential_records = generate_structured_records(
    context=context,
    questions=questions,
    instructions=instructions,
    llm=llm,
)
print(sequential_records)
# Expected output:
# {'spreadsheet-formulas': {'title': 'Formula Foundations', 'difficulty': 'beginner', 'duration_minutes': 20, 'prerequisite': None, 'learning_objectives': ['Write a SUM formula', 'Copy it down a column']}, 'python-cleaning': {'title': 'Cleaning Missing Values', 'difficulty': 'intermediate', 'duration_minutes': 40, 'prerequisite': 'Basic Python syntax', 'learning_objectives': ['Find null values', 'Fill or remove them']}}

# b) Via adaptive batch defaults
default_batch_records = generate_structured_records(
    context=context,
    questions=questions,
    instructions=instructions,
    llm=llm,
    batch=True,
)
print(default_batch_records)
# Expected output:
# {'spreadsheet-formulas': {'title': 'Formula Foundations', 'difficulty': 'beginner', 'duration_minutes': 20, 'prerequisite': None, 'learning_objectives': ['Write a SUM formula', 'Copy it down a column']}, 'python-cleaning': {'title': 'Cleaning Missing Values', 'difficulty': 'intermediate', 'duration_minutes': 40, 'prerequisite': 'Basic Python syntax', 'learning_objectives': ['Find null values', 'Fill or remove them']}}

# c) Via the same batch with custom settings
custom_batch_records = generate_structured_records(
    context=context,
    questions=questions,
    instructions=instructions,
    llm=llm,
    batch={
        "max_concurrency": 40,
        "max_retries": 3,
        "state_path": "artifacts/record_concurrency.json",
    },
)
print(custom_batch_records)
# Expected output:
# {'spreadsheet-formulas': {'title': 'Formula Foundations', 'difficulty': 'beginner', 'duration_minutes': 20, 'prerequisite': None, 'learning_objectives': ['Write a SUM formula', 'Copy it down a column']}, 'python-cleaning': {'title': 'Cleaning Missing Values', 'difficulty': 'intermediate', 'duration_minutes': 40, 'prerequisite': 'Basic Python syntax', 'learning_objectives': ['Find null values', 'Fill or remove them']}}
```
