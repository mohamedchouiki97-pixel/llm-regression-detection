# Judge vs human agreement

Measured 2026-10-09 on 30 cases from the deliberately degraded prompt's run. Short version: the judge agrees with a human only fairly (quadratic weighted kappa 0.35) and is stricter than the human on terse summaries, mostly because it penalizes missing details that are in the email but not in the reference summary.

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

| Item | Value |
|---|---|
| Run | `20260924T154822-89f688` (experiments/degraded.yaml: no category definitions, summaries of five words or fewer) |
| Prompt, judge | degraded, gpt-4o-mini; judge j1 (the undated gpt-4o alias, which resolves to the same `gpt-4o-2024-08-06` snapshot as j2) |
| Date scored | 2026-10-09 |
| Cases | 30, stratified on the judge's score: 5 scored 2, 12 scored 3, 12 scored 4, 1 scored 5 (every 2 and 5 in the run) |
| Exact agreement | 30.0% (9/30) |
| Pass/fail agreement | 60.0% (18/30, pass at >= 4) |
| Weighted kappa | 0.35 (quadratic) |

Confusion table (rows: human score, columns: judge score):

```
           1    2    3    4    5
      1    0    0    0    0    0
      2    0    0    0    0    0
      3    0    3    4    2    0
      4    0    2    6    4    0
      5    0    0    2    6    1
```

The filled sample is [judge_sample_20260924T154822-89f688.csv](judge_sample_20260924T154822-89f688.csv). Reproduce with `uv run mrd judge-agreement --csv docs/judge_sample_20260924T154822-89f688.csv --run 20260924T154822-89f688` (the run lives in a local `runs.db`).

## Findings

1. **Agreement is fair, not good.** A kappa of 0.35 means the judge ranks summaries in roughly the same order as the human, with a lot of disagreement. Most disagreements are one point apart; none are more than two.
2. **The judge is stricter than the human on terse summaries.** It scored lower than the human on 19 cases and higher on 2. The human passed 21 of 30 summaries; the judge passed 13. Of the 12 pass/fail disagreements, 10 are the human passing a summary the judge failed.
3. **The judge grades against the email, not only the reference.** The rubric says to use the reference summary as the standard. In 6 of the 12 pass/fail disagreements (gc-020, gc-023, gc-029, gc-057, gc-060, gc-075), the judge's reasoning cites a missing detail that is in the email but not in the reference, such as "the customer has tried to contact support twice" or "the badge shows 12 unread notifications".

## Implications

- **For this gate, the error is in the safe direction, but it inflates the size of a regression.** A judge that is harsher than a human on vague summaries makes a bad prompt look worse, not better. A rough estimate: applying the human's pass rate for each judge score in this sample (2: 2 of 5, 3: 8 of 12, 4: 10 of 12, 5: 1 of 1) to the 90 degraded cases with the right category gives a pass rate near 66%, against the judge's 42%. PR #2 would still have failed, by roughly 27 points instead of 49. The baseline was not re-scored by a human, so this is an estimate, not a measurement.
- **It is a risk near the threshold.** A prompt that writes short but correct summaries could be failed by the judge where a human would pass it. That is a false alarm, not a missed regression, but it would cost reviewer time.
- **Not measured: good prompts.** This sample comes from a deliberately bad prompt. On v1 and v2 the judge gives only 4s and 5s, and whether a human agrees there is still open.
- **Possible fix.** Tell the judge explicitly not to penalize details that are absent from the reference summary, bump `JUDGE_VERSION`, and repeat this measurement and [judge_noise.md](judge_noise.md). Not done yet.
