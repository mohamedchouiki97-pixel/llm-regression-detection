"""Pydantic schemas shared across the project."""

from datetime import date, datetime
from enum import Enum

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class ClassificationResult(BaseModel):
    """What the classifier returns. Also used as the OpenAI structured output schema."""

    category: Category
    summary: str = Field(description="One sentence summary of the customer's request.")


class FewShotExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=1)
    category: Category
    summary: str = Field(min_length=1)


class PromptConfig(BaseModel):
    """A versioned prompt, loaded from prompts/<version>.yaml."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    created_at: datetime
    model: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)


class ClassifyOutput(BaseModel):
    """A classification plus the metadata needed for evaluation."""

    result: ClassificationResult
    prompt_version: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


# Golden dataset

NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Tag = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(_[a-z0-9]+)*$")]


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Split(str, Enum):
    """dev cases may be inspected while improving prompts; test cases are held out for measuring."""

    DEV = "dev"
    TEST = "test"


class GoldenCase(BaseModel):
    """One hand-labeled test case. Ids are stable across dataset versions and never reused."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^gc-\d{3,}$")
    email: NonBlankStr
    expected_category: Category
    ideal_summary: NonBlankStr
    difficulty: Difficulty
    tags: list[Tag] = Field(default_factory=list)
    notes: str = ""
    split: Split

    @field_validator("tags")
    @classmethod
    def tags_unique(cls, tags: list[str]) -> list[str]:
        duplicates = sorted({tag for tag in tags if tags.count(tag) > 1})
        if duplicates:
            raise ValueError(f"duplicate tags: {', '.join(duplicates)}")
        return tags


class ChangelogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    date: date
    changes: NonBlankStr


class GoldenDataset(BaseModel):
    """The whole golden dataset file: a version, its changelog, and the cases."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    changelog: list[ChangelogEntry] = Field(min_length=1)
    cases: list[GoldenCase] = Field(min_length=1)


# Eval runs


class JudgeVerdict(BaseModel):
    """The judge's structured output. Reasoning comes first so the model thinks before it scores."""

    reasoning: str = Field(description="One or two sentences explaining the score.")
    score: Literal[1, 2, 3, 4, 5]


class JudgeOutput(BaseModel):
    verdict: JudgeVerdict
    model: str
    input_tokens: int
    output_tokens: int


class CaseResult(BaseModel):
    """One golden case scored in one run. Difficulty and tags are copied so old runs group correctly."""

    case_id: str
    expected_category: Category
    predicted_category: Category | None = None
    category_match: bool = False
    summary: str | None = None
    summary_score: int | None = None
    judge_reasoning: str | None = None
    passed: bool = False
    error: str | None = None
    difficulty: Difficulty
    split: Split
    tags: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    judge_cost_usd: float | None = None
    # True when the classifier answer came from the cache, so its latency is from an earlier call.
    cached: bool = False


class RunRecord(BaseModel):
    run_id: str
    created_at: datetime
    prompt_version: str
    prompt_hash: str
    model: str
    judge_model: str
    judge_version: str
    dataset_version: int
    summary_threshold: int
    git_sha: str
    git_branch: str
    git_dirty: bool
    n_cases: int
    n_passed: int
    pass_rate: float
    n_errors: int
    classifier_cost_usd: float | None
    judge_cost_usd: float | None
    n_cached: int
    results: list[CaseResult]


# Comparison


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class MetricDelta(BaseModel):
    name: str
    baseline: float
    run: float
    delta: float


class GroupDelta(BaseModel):
    """Pass rate of one group (a category, difficulty, or split) in both runs."""

    group: str
    baseline_n: int
    baseline_rate: float
    run_n: int
    run_rate: float
    delta: float


class CaseChange(BaseModel):
    """A case whose pass/fail outcome flipped between the baseline and the run."""

    case_id: str
    before: CaseResult
    after: CaseResult


class DriftResult(BaseModel):
    window: int
    floor: float
    run_ids: list[str]
    moving_average: float | None
    below_floor: bool


class Comparison(BaseModel):
    run_id: str
    baseline_id: str | None
    status: Status
    reasons: list[str]
    notes: list[str]
    warn_delta: float
    fail_delta: float
    overall: list[MetricDelta]
    by_category: list[GroupDelta]
    by_difficulty: list[GroupDelta]
    by_split: list[GroupDelta]
    regressions: list[CaseChange]
    improvements: list[CaseChange]
    unmatched_case_ids: list[str]
    prediction_changes: int
    score_changes: int
    mcnemar_p: float | None
    drift: DriftResult
