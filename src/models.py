"""Pydantic schemas shared across the project."""

from datetime import date, datetime
from enum import Enum

from typing import Annotated

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
