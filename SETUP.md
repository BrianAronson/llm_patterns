# Setup and verification

This repository requires Python 3.11 or newer and uses `uv` for its environment and build commands.

## Install

Install the base package and the dependencies used by the runnable examples:

```bash
uv sync --extra openai --extra pandas
```

## OpenAI credentials

Put your API key in a gitignored `.env` file at the repository root:

```text
OPENAI_API_KEY=your-key
```

The examples load that file with `load_dotenv()` and pass the client and model explicitly:

```python
llm = {"service": OpenAI(), "model": "gpt-5.6-luna"}
```

The helpers use reasoning effort `none`, so these calls do not allocate reasoning tokens.

## Run an example

Each `.py` companion makes real provider calls and runs every variation shown in its Markdown walkthrough. Example 06 submits a hosted OpenAI Batch and waits for it to finish.

Run examples from the repository root:

```bash
uv run python examples/01_table_mapping.py
```

## Verify locally

The tests exercise the same contracts with offline provider fakes and do not spend API tokens:

```bash
uv run python -m unittest discover -s tests -v
uv build
```
