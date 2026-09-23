"""Load versioned prompt files from prompts/ into PromptConfig."""

import hashlib
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.models import PromptConfig

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptLoadError(Exception):
    pass


def resolve_prompt_path(version_or_path: str) -> Path:
    """Accept a version id like 'v1' or a path to a YAML file."""
    candidate = Path(version_or_path)
    if candidate.suffix in (".yaml", ".yml"):
        return candidate
    return PROMPTS_DIR / f"{version_or_path}.yaml"


def load_prompt(version_or_path: str) -> PromptConfig:
    path = resolve_prompt_path(version_or_path)
    if not path.is_file():
        raise PromptLoadError(f"{path}: prompt file not found")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PromptLoadError(f"{path}: invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise PromptLoadError(f"{path}: expected a mapping at the top level, got {type(raw).__name__}")

    try:
        config = PromptConfig.model_validate(raw)
    except ValidationError as exc:
        lines = [f"{path}: invalid prompt config"]
        for err in exc.errors():
            location = ".".join(str(part) for part in err["loc"]) or "(root)"
            lines.append(f"  {location}: {err['msg']}")
        raise PromptLoadError("\n".join(lines)) from exc

    # The filename is the version id people pass on the command line, so they must agree.
    if config.version != path.stem:
        raise PromptLoadError(
            f"{path}: version field '{config.version}' does not match filename '{path.stem}'"
        )

    return config


def prompt_hash(config: PromptConfig) -> str:
    """Hash of the full prompt config, so editing a prompt without bumping its version is still detected."""
    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()[:16]
