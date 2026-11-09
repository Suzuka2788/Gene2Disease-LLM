"""File access. A missing input is always an error, never an empty result."""
from __future__ import annotations

import json
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Iterable


class MissingInputError(FileNotFoundError):
    """A required input file does not exist."""


def require_file(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise MissingInputError(f"required input is missing: {resolved}")
    return resolved


def read_text(path: str | Path) -> str:
    return require_file(path).read_text(encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(read_text(path))


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for number, line in enumerate(read_text(path).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
    return rows


def read_toml(path: str | Path) -> dict:
    with require_file(path).open("rb") as handle:
        return tomllib.load(handle)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_text(path: str | Path, text: str) -> None:
    _atomic_write(Path(path), text)


def write_json(path: str | Path, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(Path(path), text)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    _atomic_write(Path(path), "".join(line + "\n" for line in lines))
