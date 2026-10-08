# Judge vs human agreement

**Status: not measured yet.** The tooling is in place; the human scoring has not been done.

## Why this matters

Every pass/fail decision on summaries comes from the gpt-4o judge. [judge_noise.md](judge_noise.md) shows the judge is consistent with itself, but consistency is not correctness: a judge that is stable and wrong would pass every check there. This measurement asks whether a human, applying the same rubric, gives the same scores.

## Method

1. **Sample.** `mrd export-judge-sample --run <id> --n 30` draws 30 cases the judge scored, spread as evenly as possible across the judge's score levels, with a fixed seed. Stratifying on the judge's score matters here: on a good prompt the judge gives almost only 4s and 5s, so a random sample would contain little but agreement on 5 and say nothing about whether the judge catches weak summaries. Errored cases are skipped.
2. **Score blind.** The CSV holds the case id, email, ideal summary, and model summary, in shuffled order. The judge's score and reasoning are left out, so the human score is not anchored to them. The human scores each summary 1 to 5 using the judge's rubric in `src/scoring.py`:
   - 5: Same meaning as the reference and just as specific.
   - 4: Correct, but misses a minor detail from the reference.
   - 3: Partly correct, or too vague for a support agent to act on.
   - 2: Misses the customer's main request or problem.
   - 1: Wrong, invented, or unrelated to the email.
3. **Compare.** `mrd judge-agreement --csv <path> --run <id>` reports:
   - **Exact agreement**: share of cases with the same score.
   - **Pass/fail agreement**: share of cases where both put the summary on the same side of the run's summary threshold (4). This is the number that matters for the gate.
   - **Quadratic weighted Cohen's kappa**: agreement corrected for chance, where a 2 vs 5 disagreement costs nine times as much as a 4 vs 5. It is undefined when both raters used a single score.
   - **Confusion table**: rows are the human score, columns the judge score. Counts above the diagonal mean the judge scored higher than the human.

Caveats: 30 cases give a wide confidence interval on kappa, and stratifying on the judge's score over-represents low scores compared with a normal run. Read the result as "does the judge rank summaries the way a human does", not as the agreement rate on a typical run.

## Results

To be filled in after scoring.

| Item | Value |
|---|---|
| Run | |
| Prompt, judge | |
| Date scored | |
| Cases | |
| Exact agreement | |
| Pass/fail agreement | |
| Weighted kappa | |

Confusion table:

```
(paste the output of mrd judge-agreement here)
```

Findings:

- 
