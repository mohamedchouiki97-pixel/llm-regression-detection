"""File cache for LLM responses, so development reruns are cheap."""

import hashlib
import json
import os
from pathlib import Path

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "llm"


def make_key(*parts: str) -> str:
    return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()


class Cache:
    """One JSON file per key. Pass directory=None to disable caching."""

    def __init__(self, directory: Path | None):
        self.directory = directory

    def get(self, key: str) -> dict | None:
        if self.directory is None:
            return None
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A corrupt entry is treated as a miss and overwritten on the next set.
            return None

    def set(self, key: str, value: dict) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(value), encoding="utf-8")
        os.replace(tmp, path)
