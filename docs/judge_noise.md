# Judge and classifier noise

Measured 2026-09-24 on prompt v1 (gpt-4o-mini, temperature 0), judge j1 (gpt-4o, temperature 0), dataset v3 (100 cases), summary pass threshold 4.

## Why this matters

The regression gate warns at a 3 point drop in pass rate and fails at 8. Those numbers only mean something if an unchanged prompt does not move the pass rate by that much on its own. If identical runs flipped 3 cases, a 3 point warning would fire on noise alone.

## Method

Three independent, uncached executions of the same prompt against the same dataset:

| Run | Notes |
|---|---|
| `20260923T233117-9df5d7` | Replays the answers of the first real run (2026-09-23) from the cache |
| `20260924T153412-98c363` | `--no-cache`, stored on branch `experiment/judge-noise` |
| `20260924T153448-113d04` | `--no-cache`, stored on branch `experiment/judge-noise` |

The two new runs were stored on a non main branch so they stay out of the baseline and drift history. Each pair was compared case by case. To separate judge noise from classifier noise, score changes were also counted only on cases where the classifier wrote exactly the same summary in both runs, so the judge saw identical input.

## Results

| Pair | Category flips | Summary text changed | Score changed | Score changed, identical summary | Pass/fail flips |
|---|---|---|---|---|---|
| 9df5d7 vs 98c363 | 0 | 16 | 7 | 4 of 84 | 0 |
| 9df5d7 vs 113d04 | 0 | 21 | 7 | 3 of 79 | 0 |
| 98c363 vs 113d04 | 0 | 15 | 2 | 1 of 85 | 0 |

All three runs scored 92/100 and failed the same 8 cases (gc-001, gc-020, gc-044, gc-063, gc-081, gc-088, gc-089, gc-092). Mean summary score moved between 4.73 and 4.78.

## Findings

1. **Categories are stable.** Across 300 classifications, no predicted category changed. The category dimension is effectively deterministic at temperature 0 for this prompt.
2. **Summaries are not.** Temperature 0 still produced different summary wording in 15 to 21 percent of cases between runs.
3. **The judge is slightly noisy on its own.** With identical input, the judge changed its score on 1 to 5 percent of cases. Including classifier wording changes, 2 to 7 scores changed per pair.
4. **Every score change was between 4 and 5.** The judge used only scores 4 and 5 in all three runs, and the pass threshold is 4, so no score change crossed the threshold and pass/fail was perfectly stable.
5. **Mean summary score moves by about 0.05 from noise alone.** Treat a mean summary delta under 0.1 as noise.

## Implications for thresholds

- The 3 point warn and 8 point fail thresholds are safe for this prompt: the measured noise in pass rate is 0 points.
- That safety depends on summaries scoring well above the threshold. A weaker prompt whose summaries sit near a score of 3 or 4 would put cases on the pass/fail boundary, where the observed judge noise (up to 5 percent of cases) could flip them. Rerun this measurement when a prompt produces many scores of 3 or 4.
- The judge never gave a score below 4 on this prompt. It is not yet shown that it can detect bad summaries. The deliberately worse prompt in Phase 5 will test this.

## How to repeat

```
GIT_BRANCH=experiment/judge-noise uv run mrd run --prompt v1 --no-cache
GIT_BRANCH=experiment/judge-noise uv run mrd run --prompt v1 --no-cache
uv run mrd compare --run <second run id> --baseline <first run id>
```

`mrd compare` prints changed predictions and changed summary scores directly. Cost is about $0.30 for the two runs.
