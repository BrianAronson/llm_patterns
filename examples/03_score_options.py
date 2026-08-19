# 0) Imports
from dotenv import load_dotenv
from openai import OpenAI
from llm_patterns import score_options

# 1) Constants
load_dotenv()
llm = {"service": OpenAI(), "model": "gpt-5.6-luna"}
options = {
    "key-rotation": "Rotate API keys safely by overlapping old and new credentials",
    "key-creation": "Create and securely store a new API key",
    "team-access": "Invite teammates and manage their account roles",
}
rubric = "5 = directly answers the question; 4 = very useful; 3 = partly useful; 2 = only loosely related; 1 = unrelated."
batch_inputs = {
    "question-104": "How do I replace an API key without breaking production requests?",
    "question-105": "Where should a newly created API key be stored?",
}

# 2) Examples

# a) Via one input
single_scores = score_options(
    input_text=batch_inputs["question-104"],
    options=options,
    rubric=rubric,
    llm=llm,
)
print(single_scores)
# Expected output:
# {'key-rotation': 5, 'key-creation': 3, 'team-access': 1}

# b) Via a batch with default settings
default_batch_scores = score_options(
    input_text=batch_inputs,
    options=options,
    rubric=rubric,
    llm=llm,
    batch=True,
)
print(default_batch_scores)
# Expected output:
# {'question-104': {'key-rotation': 5, 'key-creation': 3, 'team-access': 1}, 'question-105': {'key-rotation': 2, 'key-creation': 5, 'team-access': 1}}

# c) Via the same batch with custom settings
custom_batch_scores = score_options(
    input_text=batch_inputs,
    options=options,
    rubric=rubric,
    llm=llm,
    batch={
        "max_concurrency": 40,
        "max_retries": 3,
        "state_path": "artifacts/scoring_concurrency.json",
    },
)
print(custom_batch_scores)
# Expected output:
# {'question-104': {'key-rotation': 5, 'key-creation': 3, 'team-access': 1}, 'question-105': {'key-rotation': 2, 'key-creation': 5, 'team-access': 1}}
