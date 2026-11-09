"""The frozen protocol: the only way to read a locked evaluation split.

The guard is against reading test data by accident before the protocol is fixed, and
for an audit trail afterwards; it is not a defence against deliberate tampering.

`write_manifest` (the "freeze" stage) records once, with exclusive file creation:
the digest of the package source, of every configuration allowed to run on a locked
split (keyed by its path inside the git repository), of the data files those
configurations read, of the model weight files, and the judge run ids. `authorize`
then admits a configuration only if the manifest is committed to git and unmodified,
the source still has the frozen digest, the configuration file and its data files are
byte-identical to the frozen ones. Any later change refuses the run; a new freeze
would be a second, visible manifest.
"""
from __future__ import annotations

import datetime
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import corpus, io

PACKAGE = Path(__file__).resolve().parent
LOCKED_STAGES = frozenset({"judge", "compare"})


def code_digest(package: Path = PACKAGE) -> str:
    """Digest of every .py file under the package, by relative path and content."""
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(package).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_splits(data: dict) -> set[str]:
    """The locked splits a data section asks for."""
    locked = {"biored": corpus.LOCKED_BIORED_SPLITS,
              "biotriplex": corpus.LOCKED_BIOTRIPLEX_SPLITS}.get(data.get("dataset"))
    return set(data.get("splits", [])) & set(locked or ())


def data_files(data: dict) -> list[Path]:
    """The files a data section reads."""
    if data["dataset"] == "biored":
        return [Path(data["folder"]) / f"{split}.BioC.JSON" for split in sorted(data["splits"])]
    if data["dataset"] == "biotriplex":
        return [Path(data["release"])]
    raise ValueError(f"unknown dataset: {data['dataset']}")


def _git(args: list[str], where: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=where, capture_output=True, text=True)


def repository_root(where: Path) -> Path:
    result = _git(["rev-parse", "--show-toplevel"], where)
    if result.returncode != 0:
        raise corpus.LockedSplitError(f"{where} is not inside a git repository")
    return Path(result.stdout.strip()).resolve()


def repository_key(path: str | Path, root: Path) -> str:
    """A file's path inside the repository, the same from any working directory."""
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise corpus.LockedSplitError(f"{path} is outside the repository {root}") from error


def data_key(path: str | Path, root: Path) -> str:
    """Data may live outside the repository (the corpora are never committed), so a
    data file is keyed by its repository path when inside, else by its absolute path."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    return resolved.as_posix()


def is_committed_unmodified(path: Path) -> bool:
    where = path.resolve().parent
    tracked = _git(["ls-files", "--error-unmatch", path.name], where).returncode == 0
    clean = _git(["diff", "--quiet", "HEAD", "--", path.name], where).returncode == 0
    return tracked and clean


def write_manifest(config: dict) -> dict:
    """The "freeze" stage. experiments.run writes the result with exclusive creation."""
    section = config["freeze"]
    manifest_path = Path(config["run"]["output"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    root = repository_root(manifest_path.resolve().parent)
    configs, data, run_ids, judge_tables = {}, {}, [], []
    for name in section["configs"]:
        settings = io.read_toml(name)
        if settings["run"].get("freeze_manifest") is None or \
                Path(settings["run"]["freeze_manifest"]).resolve() != manifest_path.resolve():
            raise ValueError(f"{name} does not name this manifest as its freeze_manifest")
        if settings["run"]["stage"] not in LOCKED_STAGES:
            raise ValueError(f"{name}: only judge and compare stages may be frozen")
        configs[repository_key(name, root)] = file_digest(name)
        for path in data_files(settings["data"]):
            data[data_key(path, root)] = file_digest(path)
        if settings["run"]["stage"] == "judge":
            run_ids.append(settings["judge"]["run_id"])
            # The experimental condition: data and every judge setting except the id.
            condition = {"data": settings["data"],
                         "judge": {k: v for k, v in settings["judge"].items() if k != "run_id"}}
            judge_tables.append(condition)
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("two frozen judge configurations share a run id")
    if any(judge_tables.count(table) > 1 for table in judge_tables):
        raise ValueError("two frozen judge configurations have the same condition")
    return {"frozen_on": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "code_sha256": code_digest(),
            "configs": configs,
            "data_sha256": data,
            "judge_run_ids": sorted(run_ids),
            "weights_sha256": {Path(path).name: file_digest(os.path.expandvars(path))
                               for path in section.get("weights", [])},
            # Execution scripts outside the package (driver, server launch), verified
            # by the driver before it starts.
            "scripts_sha256": {repository_key(path, root): file_digest(path)
                               for path in section.get("scripts", [])},
            "protocol": section.get("protocol", {}),
            "notes": section.get("notes", "")}


@dataclass(frozen=True)
class Authorization:
    """Proof, for one execution, that a configuration may read locked splits."""
    manifest: str
    manifest_sha256: str
    judge_run_ids: frozenset[str]


def authorize(config_path: str | Path, config: dict,
              committed=is_committed_unmodified) -> Authorization | None:
    """None when the configuration asks for no locked split; raises when it asks for one
    without a valid freeze. Checks run before any data is read."""
    if not locked_splits(config.get("data", {})):
        return None
    if config["run"].get("stage") not in LOCKED_STAGES:
        raise corpus.LockedSplitError("only judge and compare may read locked splits")
    name = config["run"].get("freeze_manifest")
    if not name:
        raise corpus.LockedSplitError("locked splits need a freeze_manifest in [run]")
    path = Path(name)
    if not path.is_file():
        raise corpus.LockedSplitError(f"the protocol is not frozen yet: {name} is missing")
    if not committed(path):
        raise corpus.LockedSplitError(f"{name} is not committed to git or has been modified")
    manifest = io.read_json(path)
    root = repository_root(path.resolve().parent)
    if manifest.get("code_sha256") != code_digest():
        raise corpus.LockedSplitError("package source changed since the freeze")
    key = repository_key(config_path, root)
    if manifest.get("configs", {}).get(key) != file_digest(config_path):
        raise corpus.LockedSplitError(f"{key} is not in the frozen manifest or has changed")
    for data_path in data_files(config["data"]):
        key = data_key(data_path, root)
        if manifest.get("data_sha256", {}).get(key) != file_digest(data_path):
            raise corpus.LockedSplitError(f"data file {key} is not frozen or has changed")
    return Authorization(name, file_digest(path), frozenset(manifest.get("judge_run_ids", [])))
