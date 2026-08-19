# Route a request to one allowed destination

You have a support inbox where every request must be routed to an existing queue, but users describe problems in their own words and do not know your internal queue names. `choose_from_options(...)` gives the LLM the request and every allowed queue, then resolves its compact numeric answer back to one stable queue ID. The model cannot invent a new destination.

For many requests that share the same queues and criteria, pass their stable IDs and text in a mapping and set `batch=True`. Each request remains a separate model assignment, and results return under the same IDs. The default uses a concurrency ceiling of 100 and up to five transient retries with no state file; pass a dictionary to override those settings.

```python
# 0) Imports
from dotenv import load_dotenv
from openai import OpenAI
from llm_patterns import choose_from_options

# 1) Constants
load_dotenv()
llm = {"service": OpenAI(), "model": "gpt-5.6-luna"}
options = {
    "billing": "Charges, invoices, refunds, and payment issues",
    "account_access": "Sign-in, password, and account access problems",
    "technical": "Product errors, outages, and unexpected behavior",
}
criteria = "Choose the queue responsible for resolving the request."
batch_inputs = {
    "ticket-104": "I was charged twice for the same invoice.",
    "ticket-105": "I cannot sign in after resetting my password.",
    "ticket-106": "The application crashes whenever I upload a file.",
}

# 2) Examples

# a) Via one input
single_choice = choose_from_options(
    input_text=batch_inputs["ticket-104"],
    options=options,
    criteria=criteria,
    llm=llm,
)
print(single_choice)
# Expected output:
# billing

# b) Via a batch with default settings
default_batch_choices = choose_from_options(
    input_text=batch_inputs,
    options=options,
    criteria=criteria,
    llm=llm,
    batch=True,
)
print(default_batch_choices)
# Expected output:
# {'ticket-104': 'billing', 'ticket-105': 'account_access', 'ticket-106': 'technical'}

# c) Via the same batch with custom settings
custom_batch_choices = choose_from_options(
    input_text=batch_inputs,
    options=options,
    criteria=criteria,
    llm=llm,
    batch={
        "max_concurrency": 40,
        "max_retries": 3,
        "state_path": "artifacts/choice_concurrency.json",
    },
)
print(custom_batch_choices)
# Expected output:
# {'ticket-104': 'billing', 'ticket-105': 'account_access', 'ticket-106': 'technical'}
```
