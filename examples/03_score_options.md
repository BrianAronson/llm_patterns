# Score every search result in one call

You run a help center where ordinary search can find broadly related articles but cannot tell you how directly each one answers a user's question. You may want to show several strong results, keep ties, or reject everything below a threshold rather than choose one winner. `score_options(...)` gives the LLM the question, every retrieved article, and your rubric, then returns scores under the stable article IDs only after each article receives exactly one legal score from 1 to 5.

For many questions that share the same articles and rubric, pass their stable IDs and text in a mapping and set `batch=True`. Each question remains a separate model assignment, and results return under the same IDs. The default uses a concurrency ceiling of 100 and up to five transient retries with no state file; pass a dictionary to override those settings.

```python
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
```
