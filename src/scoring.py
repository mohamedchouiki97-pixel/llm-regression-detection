"""Per-case scoring: category match, LLM judged summary quality, and estimated cost."""

from openai import AsyncOpenAI

from src.models import CaseResult, ClassifyOutput, GoldenCase, JudgeOutput, JudgeVerdict

JUDGE_MODEL = "gpt-4o"
# Bump when the rubric or judge prompt changes: scores from different versions are not comparable.
JUDGE_VERSION = "j1"

DEFAULT_SUMMARY_THRESHOLD = 4

# Estimated USD per million tokens as (input, output). Update when OpenAI changes pricing.
PRICES_PER_MILLION = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

JUDGE_SYSTEM_PROMPT = """\
You grade summaries of customer support emails.

You receive the customer's email, an ideal reference summary written by a human, and a candidate summary \
written by a model. Score how well the candidate captures what the customer wants or reports, using the \
reference as the standard.

Rubric:
5: Same meaning as the reference and just as specific.
4: Correct, but misses a minor detail from the reference.
3: Partly correct, or too vague for a support agent to act on.
2: Misses the customer's main request or problem.
1: Wrong, invented, or unrelated to the email.

Judge meaning, not wording. Do not reward or penalize length or phrasing on its own. \
Explain your reasoning in one or two sentences before giving the score."""


class JudgeError(Exception):
    pass


def build_judge_messages(email: str, ideal_summary: str, candidate_summary: str) -> list[dict]:
    user = (
        f"<email>\n{email}\n</email>\n\n"
        f"<reference_summary>\n{ideal_summary}\n</reference_summary>\n\n"
        f"<candidate_summary>\n{candidate_summary}\n</candidate_summary>"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


async def judge_summary(
    email: str,
    ideal_summary: str,
    candidate_summary: str,
    client: AsyncOpenAI,
    model: str = JUDGE_MODEL,
) -> JudgeOutput:
    completion = await client.chat.completions.parse(
        model=model,
        messages=build_judge_messages(email, ideal_summary, candidate_summary),
        response_format=JudgeVerdict,
        temperature=0,
    )
    message = completion.choices[0].message
    if message.refusal:
        raise JudgeError(f"judge refused: {message.refusal}")
    if message.parsed is None:
        raise JudgeError("judge returned no parsed output")

    usage = completion.usage
    return JudgeOutput(
        verdict=message.parsed,
        model=model,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated USD cost, or None for a model with no known price."""
    # Longest prefix first, so a dated snapshot like gpt-4o-mini-2024-07-18 prices as gpt-4o-mini.
    for name in sorted(PRICES_PER_MILLION, key=len, reverse=True):
        if model == name or model.startswith(f"{name}-"):
            input_price, output_price = PRICES_PER_MILLION[name]
            return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return None


def score_case(
    case: GoldenCase,
    output: ClassifyOutput | None,
    judge: JudgeOutput | None,
    threshold: int = DEFAULT_SUMMARY_THRESHOLD,
    cached: bool = False,
    error: str | None = None,
) -> CaseResult:
    """Combine whatever finished for this case into a result. A case with an error never passes."""
    result = CaseResult(
        case_id=case.id,
        expected_category=case.expected_category,
        difficulty=case.difficulty,
        split=case.split,
        tags=list(case.tags),
        cached=cached,
        error=error,
    )

    if output is not None:
        result.predicted_category = output.result.category
        result.category_match = output.result.category == case.expected_category
        result.summary = output.result.summary
        result.latency_ms = output.latency_ms
        result.input_tokens = output.input_tokens
        result.output_tokens = output.output_tokens
        result.cost_usd = estimate_cost(output.model, output.input_tokens, output.output_tokens)

    if judge is not None:
        result.summary_score = judge.verdict.score
        result.judge_reasoning = judge.verdict.reasoning
        result.judge_cost_usd = estimate_cost(judge.model, judge.input_tokens, judge.output_tokens)

    result.passed = (
        error is None
        and result.category_match
        and result.summary_score is not None
        and result.summary_score >= threshold
    )
    return result
