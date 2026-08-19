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
