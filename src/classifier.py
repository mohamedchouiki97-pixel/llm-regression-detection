"""The LLM feature under test: a customer support email classifier."""

import time

from openai import AsyncOpenAI

from src.models import ClassificationResult, ClassifyOutput, PromptConfig


class ClassificationError(Exception):
    pass


def build_messages(email_text: str, config: PromptConfig) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": config.system_prompt}]
    for example in config.few_shot_examples:
        answer = ClassificationResult(category=example.category, summary=example.summary)
        messages.append({"role": "user", "content": example.email})
        messages.append({"role": "assistant", "content": answer.model_dump_json()})
    messages.append({"role": "user", "content": email_text})
    return messages


async def classify_email(
    email_text: str,
    config: PromptConfig,
    client: AsyncOpenAI,
) -> ClassifyOutput:
    """The caller owns the client and is responsible for closing it."""
    start = time.perf_counter()
    completion = await client.chat.completions.parse(
        model=config.model,
        messages=build_messages(email_text, config),
        response_format=ClassificationResult,
        temperature=0,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    message = completion.choices[0].message
    if message.refusal:
        raise ClassificationError(f"model refused: {message.refusal}")
    if message.parsed is None:
        raise ClassificationError("model returned no parsed output")

    usage = completion.usage
    return ClassifyOutput(
        result=message.parsed,
        prompt_version=config.version,
        model=config.model,
        latency_ms=latency_ms,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )
