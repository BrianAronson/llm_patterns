# Run a durable prompt batch through OpenAI

You have thousands of independent records to classify, and the results do not need to be immediate. OpenAI Batch can run them after your local process exits, using separate rate limits at a lower cost than synchronous requests. `submit_openai_prompt_batch(...)` writes exact `/v1/responses` JSONL, splits oversized collections, and records immutable upload and submission receipts. `collect_openai_prompt_batch(...)` can run later to check status, download terminal files, reject incomplete responses, and restore successful text under your stable prompt IDs.

Submission and collection are separate because a hosted Batch can take time. The cell below submits the work and then waits for collection to finish. Those two sections can also run in different processes. Downloaded artifacts remain in `run_directory`, so a completed collection does not need to contact OpenAI again.

The lifecycle is `stable IDs and complete prompts → Responses JSONL and receipts → provider-hosted execution → downloaded files → exact ID reconciliation`.

```python
# 0) Imports
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from llm_patterns import collect_openai_prompt_batch, submit_openai_prompt_batch

# 1) Constants
load_dotenv()
llm = {"service": OpenAI(), "model": "gpt-5.6-luna"}
run_directory = Path("artifacts/inspection-batch-v1")
prompts = {
    "inspection-104": "Classify this inspection note as routine, priority, or urgent. Return only the label.\n\nA sparking outlet is smoking.",
    "inspection-105": "Classify this inspection note as routine, priority, or urgent. Return only the label.\n\nThere is a small paint chip.",
}

# 2) Examples

# a) Submit once
submission = submit_openai_prompt_batch(
    prompts=prompts,
    run_directory=run_directory,
    llm=llm,
)
print(submission.provider_batch_ids)
# Expected output resembles:
# ('batch_abc123',)

# b) Wait for completion and collect
collection = collect_openai_prompt_batch(
    run_directory=run_directory,
    llm=llm,
    wait=True,
    poll_interval_seconds=10,
)
print(collection.status)
# Expected output:
# completed

if collection.status == "completed":
    print(collection.response_texts())
    # Expected output:
    # {'inspection-104': 'urgent', 'inspection-105': 'routine'}
```
