# Edit part of a document without regenerating the whole thing

You have a service report whose status line is vague. The model needs the rest of the report for context, but returning a complete rewrite would spend output tokens on unchanged text and could introduce unrelated edits. `edit_selected_lines(...)` sends the complete document as read-only context and asks only for replacements to the selected text. It verifies exact selections, rejects multiline replacements or changed Markdown structure, and applies valid changes locally.

The helper adds input tokens for the selection and response instructions, but avoids returning the unchanged document. For the small example below, a compact full-rewrite prompt is approximately 85 input and 58 output tokens. The bounded version is approximately 139 input and 21 output tokens. At the current short-context Luna rates of `$0.20` per million input tokens and `$1.20` per million output tokens, that is about 39% lower cost for a successful first response. These are local estimates that exclude provider message framing; a validation-recovery call would add more tokens.

```python
# 0) Imports
from dotenv import load_dotenv
from openai import OpenAI
from llm_patterns import edit_selected_lines

# 1) Constants
load_dotenv()
llm = {"service": OpenAI(), "model": "gpt-5.6-luna"}
document = """# Service report
- Impact: 23 checkout requests failed.
- Recovery: Service was restored at 14:20 UTC and has remained stable.
- Status: The problem is done.
- Next step: Monitor error rates for 24 hours.
"""
instruction = "Rewrite the selected status line to include the restoration time and say that the service remains stable."

# 2) Examples

# a) Via the exact text to edit
edited_document = edit_selected_lines(
    document=document,
    selected_lines=["- Status: The problem is done."],
    instruction=instruction,
    llm=llm,
)
print(edited_document)
# Expected output:
# # Service report
# - Impact: 23 checkout requests failed.
# - Recovery: Service was restored at 14:20 UTC and has remained stable.
# - Status: Service was restored at 14:20 UTC and remains stable.
# - Next step: Monitor error rates for 24 hours.
```
