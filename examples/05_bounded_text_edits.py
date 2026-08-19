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
