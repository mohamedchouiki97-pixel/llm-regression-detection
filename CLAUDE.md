# Model Regression Detection System

## What this project is
A CI pipeline that tests an LLM feature against a hand-labeled golden dataset whenever a prompt or model changes. It scores outputs, compares them to the previous run, detects regressions, alerts Slack, and blocks PRs with critical regressions.

The LLM feature under test is a customer support email classifier. It is intentionally simple. The product is the evaluation machinery around it.

## How we work (read this every session)
- Work one phase at a time. Do not start the next phase until I confirm the current one.
- Before writing code for a phase, propose the approach and file changes, then wait for approval.
- After finishing a phase, summarize: what you built, key design decisions, and what could break.
- Write tests for scoring, diffing, and statistics logic. These must be correct without calling an LLM.
- Never generate golden dataset entries. I write those by hand. You may build tooling to validate them.
- Keep it simple. No abstractions we do not need yet.
- Writing style for docs and comments: no em dashes or double dashes.

## Tech stack
- Python 3.11+, managed with uv
- OpenAI API: gpt-4o-mini for the classifier, gpt-4o for LLM-as-judge
- Pydantic v2 for all schemas
- SQLite for run history, JSON for the golden dataset, YAML for prompts
- Jinja2 for the HTML report
- pytest
- GitHub Actions, Docker, Slack incoming webhooks

## Repo layout
```
prompts/            versioned prompt YAML files
data/golden.json    golden dataset (hand-written)
src/
  models.py         Pydantic schemas
  classifier.py     the LLM feature under test
  prompts.py        prompt loading
  runner.py         async eval runner
  scoring.py        scoring dimensions + LLM judge
  compare.py        diffing + statistics + drift
  store.py          SQLite run history
  report.py         HTML report
  alerts.py         Slack
  cli.py            entry point
templates/          Jinja2 report template
tests/
.github/workflows/
```

---

## Phase 1: The feature under test
Goal: a classifier whose prompt is a parameter, with structured, validated output.

Build:
- `Category` enum: billing, technical, account, general
- `ClassificationResult`: category + one sentence summary
- `PromptConfig`: version id, created_at, model, system_prompt, few_shot_examples
- `classify_email(email_text, config)` using OpenAI structured outputs, temperature 0
- Return the result plus metadata: latency_ms, input_tokens, output_tokens
- Prompt files in `prompts/` (start with `v1.yaml`), loaded and validated into `PromptConfig`

Done when:
- A CLI command classifies a sample email with a chosen prompt version and prints result + metadata
- Invalid prompt YAML fails with a clear error

## Phase 2: Golden dataset infrastructure
Goal: a schema and validator. I write the actual cases.

Build:
- `GoldenCase` schema: id (stable, e.g. `gc-001`), email, expected_category, ideal_summary, difficulty (easy, medium, hard), tags (e.g. ambiguous, typo, short, mixed_language, sarcasm), notes
- Dataset file has a top-level version and changelog
- Validation command: checks schema, unique ids, category balance, and prints a coverage table by category and difficulty
- Seed file with 3 example cases only, clearly marked as placeholders

Done when:
- Validator passes on a valid file and gives useful errors on a broken one
- (Me) I have written 50 to 100 cases, including deliberate edge cases

## Phase 3: Eval engine
Goal: run every case, score on several dimensions, store everything.

Build:
- Async runner with a concurrency limit and retries with backoff
- Cache LLM responses keyed by hash of (prompt version, model, email) so reruns are cheap during development; cache can be disabled
- Scoring per case:
  - category_match (binary)
  - summary_score 1 to 5 from an LLM judge with a written rubric, returning score + short reasoning as structured output
  - latency_ms, tokens, estimated cost
- A case "passes" if category matches and summary_score >= a configurable threshold
- Store each run in SQLite: run metadata (run id, prompt version, model, dataset version, git sha, timestamp) and per-case results

Done when:
- One command runs the full dataset and stores a run
- Unit tests cover scoring logic with fake LLM outputs

## Phase 4: Comparison and statistics
Goal: the core value. Diff a run against a baseline and decide if it is a real regression.

Build:
- Compare run vs baseline (default: latest run on main): overall pass rate delta, per-category accuracy delta, per-difficulty delta, regressions (pass to fail), improvements (fail to pass)
- Configurable thresholds: warning and critical deltas (start at 3% and 8%)
- McNemar exact test on flipped cases to report whether the change is statistically meaningful; report the p value alongside the threshold result
- Slow drift: 7-run moving average of pass rate; warn if it falls below a configurable floor even when no single run alerted
- Final status: pass, warn, or fail

Done when:
- Unit tests cover diffing, thresholds, McNemar, and drift with hand-built fake runs
- Measure judge noise: run the same prompt twice and report how many cases flip; document it

## Phase 5: Reporting and alerts
Build:
- HTML report: run metadata, scorecard vs baseline, regressed cases with old vs new output side by side (including judge reasoning), trend chart of the last N runs (inline SVG or Chart.js from a CDN)
- Slack message via webhook: status, headline numbers, link to report. Skip gracefully if no webhook is set.

Done when:
- Changing a prompt to something deliberately worse produces a report that clearly shows the regressed cases

## Phase 6: CI/CD and Docker
Build:
- GitHub Action triggered on PRs that touch `prompts/` or `data/`
- Runs eval, uploads HTML report as an artifact, posts a PR comment with status and headline numbers, fails the check on critical regressions
- Dockerfile packaging the runner, dataset, and report layer; config via env vars (OPENAI_API_KEY, SLACK_WEBHOOK_URL, thresholds)

Done when:
- A PR with a bad prompt change is blocked, and a good one passes

## Phase 7: Portfolio polish
- README written like internal onboarding docs: what it does, setup, how to add golden cases, how to tune thresholds, architecture decisions with rationale
- Short write-up on one design decision (e.g. why statistical testing and slow drift are tracked separately from threshold deltas)
- (Me) Record a 3 minute Loom: change prompt, PR, eval runs, Slack alert, report walkthrough
