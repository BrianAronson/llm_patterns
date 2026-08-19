# LLM patterns

I keep this repository for LLM code I expect to copy or reuse. Each pattern gives the model one recognizable job and keeps the parts that must be exact in Python: allowed IDs, schema coverage, retry limits, concurrency, saved progress, or the precise text that may change. It is not meant to be a general LLM framework.

The numbered walkthroughs are the main documentation. Each has a compact explanation and a matching `.py` file that runs every variation. The examples use OpenAI; the tests exercise the same contracts offline. All runnable examples load `OPENAI_API_KEY` from `.env` and use Luna with reasoning effort set to `none`. See [Setup and verification](SETUP.md) for installation and commands.

## Patterns

- **Map someone else's table into your schema.** `map_table_columns(...)` can use column names, representative values, or a DataFrame to identify the right source for every required field. It refuses missing, invented, or reused columns. [Walkthrough](examples/01_table_mapping.md)
- **Make one choice from a closed set.** `choose_from_options(...)` lets the model reason about the choice, but the returned value must be an ID you supplied. The same call handles one input or a batch. [Walkthrough](examples/02_choose_from_options.md)
- **Score a whole candidate set without forcing a winner.** `score_options(...)` returns a 1-to-5 score for every supplied ID, which leaves room for ties, thresholds, or several strong results. [Walkthrough](examples/03_score_options.md)
- **Run independent prompts without choosing one fixed concurrency forever.** `run_prompt_batch(...)` increases or reduces pressure as calls succeed or hit rate limits, preserves input order, and can resume locally from saved results. [Walkthrough](examples/04_prompt_batch.md)
- **Give the model the whole document without letting it rewrite the whole document.** `edit_selected_lines(...)` uses the full text as context, accepts replacements only for exact selected lines, and leaves everything else untouched. [Walkthrough](examples/05_bounded_text_edits.md)
- **Send a large job to OpenAI and collect it later without guessing what already ran.** `submit_openai_prompt_batch(...)` and `collect_openai_prompt_batch(...)` split work under provider limits, save receipts, verify downloads, and restore results under your original IDs. [Walkthrough](examples/06_openai_prompt_batch.md)
- **Turn repeated questions into typed rows without writing a model class.** `generate_structured_records(...)` infers one Pydantic shape, reuses it for every supplied context, and returns a dictionary ready to become rows. [Walkthrough](examples/07_structured_records.md)
