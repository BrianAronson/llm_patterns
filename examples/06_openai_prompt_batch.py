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
