# Why the gate uses separate signals instead of one number

The obvious design for an eval gate is one number and one line: compute the pass rate, compare it with the baseline, fail below some delta. This project started there and ended with four signals reported side by side: a **threshold** on the pass rate change, a **significance test** on the cases that flipped, a **drift** check over recent history, and an **error rate** that says whether the run measured anything at all. This note explains why, using what actually happened while building it.

## The four questions

Each signal answers a different question, and a single number can only answer one of them.

| Signal | Question | Implementation |
|---|---|---|
| Threshold | Is the change big enough to matter? | Overall pass rate drop: warn at 3 points, fail at 8 |
| Significance | Is the change real or noise? | Exact McNemar test on cases that flipped between the two runs |
| Drift | Are small changes adding up? | 7 run moving average of pass rate on main, warn below 88% |
| Error rate | Did we measure it? | Fail as "eval incomplete" when more than 5% of cases errored |

The interesting cases are the ones where the signals disagree. Every row below happened in this repository.

## When they agree: the bad prompt

[PR #2](https://github.com/mohamedchouiki97-pixel/llm-regression-detection/pull/2) replaced the prompt with a "shorter" one: no category definitions, summaries of five words or fewer. The pass rate fell from 93% to 44%: 51 cases regressed, 2 improved, McNemar p = 3e-13. The threshold says it matters, the test says it is real, nothing errored. Any design catches this, and it is the case that makes a one-number gate look sufficient.

## Big enough, but not proven: prompt v2

[PR #1](https://github.com/mohamedchouiki97-pixel/llm-regression-detection/pull/1) introduced v2, which writes the labeling guide's rules into the prompt. Its final version moved the pass rate from 93% to 97%: 5 improvements, 1 regression, p = 0.22.

A threshold alone would report "+4 points" and invite someone to write "v2 is 4 points better" in a changelog. The significance test says something more careful: a 5 to 1 split among 6 flipped cases happens by chance about one time in five. v2 is probably better, and the per-case evidence (the fixed cases are exactly the SSO and invite emails it targeted) supports that, but 100 cases cannot prove it. Reporting both numbers keeps the claim honest without blocking a good change.

The reverse matters for blocking. If the gate required significance before failing, a real 5 point regression spread over few flips could pass as "not significant". With 100 cases the test has limited power, so the threshold decides the status and the p value explains it.

## Fooled together: the rate limited run

The first CI run of PR #1 reported this:

> FAIL: pass rate dropped 9.0% (fail at 8%); 13 regressions, 4 improvements, McNemar p = 0.049, significant

Both signals agreed, and both were wrong. Nine of the thirteen "regressions" were judge calls that failed under OpenAI rate limits and were counted as failing cases. The threshold saw a 9 point drop; the significance test saw a lopsided flip count. Neither was designed to ask whether the numbers were measurements at all.

That is why error rate became its own signal:

- Cases that errored in either run are left out of every delta and out of the significance test, so missing data cannot pose as a regression.
- More than 5% errors fails the check with a distinct reason, "eval incomplete", because a change that was not measured should not be approved either.

After the fix, the same PR reported +1 point with no errors. Later the OpenAI account ran out of credits during a main branch run. 63 of 100 cases errored, and the gate reported "eval incomplete" instead of a drop of roughly 60 points. The next run then exposed a second gap: that broken run had become the baseline. Error-heavy runs are now excluded from baseline and drift lookups. A missing measurement needs its own signal because it contaminates every other signal that reads it.

## Invisible to both: slow drift

Consider five prompt PRs in a row, each losing 1.5 points. No single PR crosses the 3 point warning, and a 1 or 2 case difference is never significant. Each PR looks fine against its own baseline, which is the previous PR's result. After five merges the classifier is 7.5 points worse, and no alert ever fired.

Comparing each change only with its immediate predecessor cannot see this, by construction. Drift compares against a longer window: the 7 run moving average of pass rate on main, with a warning when it falls below a floor. It only speaks once the window is full, rather than drawing conclusions from two points, and it warns rather than fails, because by the time drift shows up the damage is already merged. Its job is to start a conversation about the trend, not to block the PR that happens to be last.

## Why the threshold numbers are what they are

Thresholds are only meaningful relative to noise. Three identical runs of the same prompt flipped 0 cases between pass and fail: categories were identical across 300 classifications, and although 1 to 5 percent of judge scores changed, every change was between 4 and 5, above the pass threshold ([judge_noise.md](judge_noise.md)). On 100 cases one flip is one point, so a 3 point warning sits above observed noise and an 8 point failure is far above it. Those numbers should be measured again whenever the judge, model, or dataset changes. A weaker prompt whose summaries score around 3 or 4 would put cases on the pass/fail boundary, where judge noise could flip them.

## What collapsing them would cost

A single combined score (for example, "fail if the drop exceeds 3 points and p < 0.05") would have:

- approved nothing it should not have in the bad prompt case, but also
- reported v2's first run as a significant regression caused by a rate limit,
- stayed silent through a slow slide of 1.5 points per PR, and
- given a reviewer one number with no way to tell which of these situations they were in.

Keeping the signals separate costs a slightly longer PR comment. In return, each failure mode that happened here has a signal whose job is to catch it, and a reviewer can see which one fired and why.

## What is still not covered

- **Per-category damage.** v2's first version passed overall while losing 7.7 points on billing. Per-category deltas are reported, not gated, because about 25 cases per category make single flips look like 4 point swings. A human reading the table caught it. A per-category gate with a wide threshold, or a significance test per category, would be the next step with a larger dataset.
- **Judge drift.** If the judge model changes behavior over time, every signal inherits it. Pinning a dated judge model and re-measuring noise periodically would address that.
