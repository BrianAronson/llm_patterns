# 0) Imports
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from llm_patterns import run_prompt_batch

# 1) Constants
load_dotenv()
llm = {"service": OpenAI(), "model": "gpt-5.6-luna"}
max_concurrency = 100
max_retries = 5
messages = {
    "message-001": "Order A-104 arrived with a cracked lid.",
    "message-002": "Please cancel B-205 before it ships.",
    "message-003": "Tracking for C-306 has not moved in four days.",
}
prompts = {
    message_id: f"""Extract the order ID from this customer message.

Return only the order ID.

Customer message:
{message}
"""
    for message_id, message in messages.items()
}

# 2) Examples

# a) Via an in-memory batch
order_ids = run_prompt_batch(
    prompts=list(prompts.values()),
    llm=llm,
    max_concurrency=max_concurrency,
    max_retries=max_retries,
    state_path="artifacts/openai_prompt_batch_concurrency.json",
)
print(order_ids)
# Expected output:
# ['A-104', 'B-205', 'C-306']

# b) Via a durable local batch
durable_order_ids = run_prompt_batch(
    prompts=prompts,
    llm=llm,
    max_concurrency=max_concurrency,
    max_retries=max_retries,
    run_directory=Path("artifacts/order-id-run-v1"),
)
print(durable_order_ids)
# Expected output:
# {'message-001': 'A-104', 'message-002': 'B-205', 'message-003': 'C-306'}
