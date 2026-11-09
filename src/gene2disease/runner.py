"""Judgment and comparison stages. Per-paper records go under runs/, which is never
committed; the returned dictionaries are aggregate only."""
from __future__ import annotations

import datetime
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import adjudicate, candidates, corpus, graph, io, scoring, stats
from .schema import Assertion, Certainty, Disposition, PairRecord, Polarity, Provenance, RelationClass
from .scoring import Counts


def load_given_entities(data: dict, allow_locked: bool = False
                        ) -> dict[str, tuple[corpus.Document, list[corpus.Mention], set]]:
    """Data with its annotated entities. Locked splits stay refused unless the caller
    was authorized by the frozen manifest (freeze.authorize, through experiments.run)."""
    if data["dataset"] == "biored":
        return corpus.load_biored(data["folder"], set(data["splits"]), allow_locked)
    if data["dataset"] == "biotriplex":
        papers = corpus.load_biotriplex(data["release"], set(data["splits"]), allow_locked)
        return {key: (paper.document, paper.mentions, paper.gold_pairs())
                for key, paper in papers.items()}
    raise ValueError(f"unknown dataset: {data['dataset']}")


def _names(mentions: list[corpus.Mention]) -> dict[str, str]:
    names: dict[str, str] = {}
    for mention in mentions:
        names.setdefault(mention.entity, mention.text)
    return names


def _serialize(record: PairRecord) -> dict:
    return asdict(record)


def _deserialize(row: dict) -> PairRecord:
    assertions = [Assertion(item["gene"], item["disease"], Polarity(item["polarity"]),
                            Certainty(item["certainty"]), RelationClass(item["relation_class"]),
                            Provenance(item["provenance"]), tuple(item["sentence_ids"]))
                  for item in row["assertions"]]
    return PairRecord(row["gene"], row["disease"], Disposition(row["disposition"]), assertions,
                      row["reason"])


def request_settings(settings: dict) -> dict:
    """The settings that change what is sent to the model, shared by judge and F0."""
    output_format = settings.get("output_format", "json")
    if output_format not in adjudicate.OUTPUT_FORMATS:
        raise ValueError(f"unknown output format: {output_format}")
    request = {"model_id": settings["model_id"], "output_format": output_format,
               "thinking": bool(settings.get("thinking", False)),
               "cache_prompt": bool(settings.get("cache_prompt", True)),
               "output_base_tokens": settings["output_base_tokens"],
               "output_tokens_per_pair": settings["output_tokens_per_pair"],
               "output_extra_tokens": settings.get("output_extra_tokens", 0)}
    # Recorded only when not the default, so runs made before the setting existed keep
    # their identity.
    instructions = settings.get("instructions", "minimal")
    if instructions not in adjudicate.INSTRUCTION_SETS:
        raise ValueError(f"unknown instructions: {instructions}")
    if instructions != "minimal":
        request["instructions"] = instructions
    return request


def read_api_key(settings: dict) -> str | None:
    """The key stays in its own file outside the project; it is never recorded."""
    path = settings.get("api_key_file")
    if not path:
        return None
    return io.read_text(Path(os.path.expandvars(path)).expanduser()).strip()


def make_backend(settings: dict, request: dict) -> adjudicate.LocalOpenAIBackend:
    return adjudicate.LocalOpenAIBackend(
        settings["base_url"], request["model_id"], settings["timeout_seconds"],
        api_key=read_api_key(settings), thinking=request["thinking"],
        cache_prompt=request["cache_prompt"])


def make_budget(request: dict) -> adjudicate.OutputBudget:
    return adjudicate.OutputBudget(request["output_base_tokens"],
                                   request["output_tokens_per_pair"],
                                   request["output_extra_tokens"])


INPUT_SCOPES = ("full", "abstract")

def _is_abstract_section(section: str) -> bool:
    """BioTriplex names the section "Abstract", "ABSTRACT" or, once, "Abstract Rev"."""
    name = section.strip().lower()
    return name == "title" or name.startswith("abstract")


def restrict_input(document: corpus.Document, scope: str) -> corpus.Document:
    """The document the model may read. "abstract" keeps only title and abstract blocks;
    the pair universe still comes from every mention, so pairs that never co-occur there
    are recorded as not assessed and count as missed against the same gold."""
    if scope == "full":
        return document
    if scope != "abstract":
        raise ValueError(f"unknown input scope: {scope}")
    kept = [block for block in document.blocks if _is_abstract_section(block.section)]
    if not kept:
        raise ValueError(f"no title or abstract blocks in {document.doc_id}")
    return corpus.Document(document.doc_id, kept)


def _run_identity(config: dict, papers: list[str]) -> dict:
    settings, data = config["judge"], config["data"]
    identity = {"dataset": data["dataset"], "splits": sorted(data["splits"]),
                "form": settings["form"], "chunk_size": settings.get("chunk_size"),
                "level": settings["level"], **request_settings(settings), "papers": papers}
    scope = settings.get("input_scope", "full")
    if scope not in INPUT_SCOPES:
        raise ValueError(f"unknown input scope: {scope}")
    if scope != "full":
        identity["input_scope"] = scope
    return identity


def _claim_run(run_dir: Path, identity: dict) -> None:
    """One run id means one configuration. Reuse with other settings is refused."""
    path = run_dir / "run.json"
    if path.is_file() and io.read_json(path) != identity:
        raise ValueError(f"run {run_dir.name} already exists with different settings")
    io.write_json(path, identity)


LOCKED_JOURNAL = "locked_evaluation.json"


def _start_locked_run(run_dir: Path, run_id: str, locked) -> None:
    """Before any locked data is read. The run id must be frozen; a run directory that
    exists without this journal holds output from elsewhere (a foreign cache) and is
    refused. Every start is appended, so a resume after interruption is visible."""
    if run_id not in locked.judge_run_ids:
        raise corpus.LockedSplitError(f"run id {run_id} is not in the frozen manifest")
    path = run_dir / LOCKED_JOURNAL
    if run_dir.exists() and not path.is_file():
        raise corpus.LockedSplitError(f"{run_dir} exists without a locked-evaluation journal")
    journal = io.read_json(path) if path.is_file() else {
        "manifest": locked.manifest, "manifest_sha256": locked.manifest_sha256, "starts": []}
    if journal["manifest_sha256"] != locked.manifest_sha256:
        raise corpus.LockedSplitError(f"{run_dir} was started under another manifest")
    journal["starts"].append(datetime.datetime.now(datetime.timezone.utc)
                             .isoformat(timespec="seconds"))
    io.write_json(path, journal)


def judge(config: dict, locked=None) -> dict:
    """`locked` is the freeze.Authorization that experiments.run obtained for this
    configuration file; without it locked splits stay refused. A locked run holds an
    exclusive lock file for its whole execution, so a second launch fails at once; a
    lock left by a crash must be removed by hand after checking that no run is active."""
    if locked is None:
        return _judge(config, None)
    run_dir = Path(config["run"]["runs_dir"]) / config["judge"]["run_id"]
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = run_dir.parent / f"{run_dir.name}.lock"
    try:
        handle = open(lock, "x", encoding="utf-8")
    except FileExistsError as error:
        raise corpus.LockedSplitError(f"{lock} exists: another run is active or crashed") from error
    try:
        handle.write(datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
        handle.close()
        return _judge(config, locked)
    finally:
        lock.unlink(missing_ok=True)


def _judge(config: dict, locked) -> dict:
    settings = config["judge"]
    run_dir = Path(config["run"]["runs_dir"]) / settings["run_id"]
    if locked is not None:
        _start_locked_run(run_dir, settings["run_id"], locked)
    papers = load_given_entities(config["data"], locked is not None)
    selected = sorted(papers)[:settings.get("max_papers")]
    identity = _run_identity(config, selected)
    _claim_run(run_dir, identity)
    metered = adjudicate.MeteredBackend(make_backend(settings, identity))
    backend = adjudicate.CachedBackend(metered, run_dir / "cache")
    budget = make_budget(identity)
    dispositions: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for paper_id in selected:
        document, mentions, _ = papers[paper_id]
        document = restrict_input(document, identity.get("input_scope", "full"))
        result = adjudicate.judge_paper(document, mentions, settings["level"], settings["form"],
                                        backend, _names(mentions), budget, frozenset(),
                                        settings.get("chunk_size"), identity["output_format"],
                                        identity.get("instructions", "minimal"))
        operational = sum(call["outcome"] == "failed" for call in result.call_log)
        if operational and settings.get("stop_on_operational_failure", False):
            # Nothing is written for this paper; its answered calls are in the cache, so
            # running the configuration again resumes here once the server is fixed.
            raise RuntimeError(f"{paper_id}: {operational} operational call failures; "
                               "the run stopped before recording this paper")
        io.write_jsonl(run_dir / paper_id / "pairs.jsonl",
                       [_serialize(record) for record in result.records])
        io.write_jsonl(run_dir / paper_id / "calls.jsonl", result.call_log)
        dispositions.update(record.disposition.value for record in result.records)
        totals.update(context_refused_calls=sum(call["outcome"] == "context_refused"
                                                for call in result.call_log))
        totals.update(logical_requests=result.calls, failed_calls=result.failed_calls,
                      truncated_calls=result.truncated_calls,
                      blocked_calls=result.blocked_calls,
                      rejected_assertions=result.rejected_assertions)
    return {"run_id": settings["run_id"], "form": settings["form"], "level": settings["level"],
            "model_id": settings["model_id"], "dataset": config["data"]["dataset"],
            "papers": len(selected), "dispositions": dict(dispositions), **totals,
            "backend_attempts": metered.attempts, "backend_seconds": metered.seconds,
            "prompt_characters": metered.prompt_characters,
            "response_characters": metered.response_characters, "cache_hits": backend.hits}


def read_edges(run_dir: Path, paper_id: str, expected: set[tuple[str, str]],
               view: str) -> set[tuple[str, str]]:
    """Every expected pair must have exactly one record. Model omissions are explicit
    records, so a short file means a damaged run, not a missing prediction."""
    records = [_deserialize(row) for row in io.read_jsonl(run_dir / paper_id / "pairs.jsonl")]
    keys = [(record.gene, record.disease) for record in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"incomplete pair records in {run_dir.name}")
    return {(record.gene, record.disease) for record in graph.edges(records, view)}


COMPARABLE_DEFAULTS = {"instructions": "minimal", "input_scope": "full"}
# Settings a comparison may vary on purpose, one at a time, when its config names it.
VARIABLE_SETTINGS = frozenset({"output_format", "instructions", "input_scope"})


def _check_comparable(first: Path, second: Path, selected: list[str], data: dict,
                      allow_differing: frozenset[str] = frozenset()) -> None:
    unknown = allow_differing - VARIABLE_SETTINGS
    if unknown:
        raise ValueError(f"these settings cannot differ between compared runs: {sorted(unknown)}")
    identities = [io.read_json(path / "run.json") for path in (first, second)]
    for identity in identities:
        if identity["papers"] != selected or identity["dataset"] != data["dataset"]:
            raise ValueError("run does not match the papers being compared")
    for key in ("splits", "level", "model_id", "output_format", "thinking", "cache_prompt",
                "output_base_tokens", "output_tokens_per_pair", "output_extra_tokens",
                "instructions", "input_scope"):
        if key in allow_differing:
            continue
        default = COMPARABLE_DEFAULTS.get(key)
        if identities[0].get(key, default) != identities[1].get(key, default):
            raise ValueError(f"runs differ in {key}")


def _band(load: int, edges: list[int]) -> str:
    for edge in edges:
        if load <= edge:
            return f"<={edge}"
    return f">{edges[-1]}"


def window_exposures(run_dir: Path, paper_id: str) -> dict[tuple[str, str, str], dict]:
    """Each (window, gene, disease) a run judged, with that window's own outcome (main
    view), the window's pair load and the size of the request that listed the pair."""
    exposures = {}
    for call in io.read_jsonl(run_dir / paper_id / "calls.jsonl"):
        for key, outcome in call["pair_outcomes"].items():
            gene, disease = key.split("|", 1)
            exposures[(call["window_id"], gene, disease)] = {
                "outcome": outcome, "window_pairs": call["window_pairs"],
                "request_pairs": call["pairs"]}
    return exposures


def _window_level(first: Path, second: Path, papers: dict, selected: list[str],
                  edges: list[int]) -> dict:
    """Gold pairs counted once per window that holds them, each judged by that window's
    own answer, so the stratum is the load of the window that produced the answer.
    This is a descriptive stratification of the paired comparison; windows of different
    load also differ in length and difficulty, so it does not by itself identify a
    batching threshold."""
    by_load: dict[str, Counter] = {}
    by_request: dict[str, dict[str, Counter]] = {"first": {}, "second": {}}
    for paper_id in selected:
        gold = papers[paper_id][2]
        one, two = window_exposures(first, paper_id), window_exposures(second, paper_id)
        if set(one) != set(two):
            raise ValueError(f"runs judged different windows in {paper_id}")
        for key, left in one.items():
            pair = (key[1], key[2])
            if pair not in gold:
                continue
            right = two[key]
            hit_first, hit_second = left["outcome"] == "edge", right["outcome"] == "edge"
            counts = by_load.setdefault(_band(left["window_pairs"], edges), Counter())
            # int(): an empty Counter stores update() values as given, so a band with
            # one exposure would otherwise report true/false instead of 1/0.
            counts.update(gold_exposures=1, found_first=int(hit_first),
                          found_second=int(hit_second),
                          found_both=int(hit_first and hit_second),
                          only_first=int(hit_first and not hit_second),
                          only_second=int(hit_second and not hit_first))
            for name, side, hit in (("first", left, hit_first), ("second", right, hit_second)):
                band = by_request[name].setdefault(_band(side["request_pairs"], edges), Counter())
                band.update(gold_exposures=1, found=int(hit))
    return {"gold_exposures_by_window_load": {name: dict(c) for name, c in sorted(by_load.items())},
            "gold_exposures_by_request_size": {
                name: {band: dict(c) for band, c in sorted(bands.items())}
                for name, bands in by_request.items()}}


def strict_edges(run_dir: Path, paper_id: str) -> set[tuple[str, str]]:
    """Main-view edges of a paper under the strict reading: the union over windows of
    pairs whose answer, parsed without canonicalization, was an edge."""
    found = set()
    for call in io.read_jsonl(run_dir / paper_id / "calls.jsonl"):
        if "raw_pair_outcomes" not in call:
            raise ValueError(f"no strict reading recorded in {run_dir.name}/{paper_id}")
        found |= {tuple(key.split("|", 1)) for key, value in call["raw_pair_outcomes"].items()
                  if value == "edge"}
    return found


def _complete_windows(first: Path, second: Path, papers: dict, selected: list[str]) -> dict:
    """Secondary diagnostic: only windows whose every call parsed in both runs, gold
    pairs counted once per window and judged by that window's answer. Failures stay in
    the primary result; this only shows how much of the difference they explain."""
    total = kept = 0
    counts: Counter = Counter()
    for paper_id in selected:
        gold = papers[paper_id][2]
        calls = {name: io.read_jsonl(run / paper_id / "calls.jsonl")
                 for name, run in (("first", first), ("second", second))}
        windows = {call["window_id"] for call in calls["first"]}
        failed = {call["window_id"] for side in calls.values() for call in side
                  if call["outcome"] != "ok"}
        total += len(windows)
        kept += len(windows - failed)
        exposures = {name: {(call["window_id"], *key.split("|", 1)): value
                            for call in side if call["window_id"] not in failed
                            for key, value in call["pair_outcomes"].items()}
                     for name, side in calls.items()}
        for key, outcome in exposures["first"].items():
            if (key[1], key[2]) in gold:
                counts.update(gold_exposures=1, found_first=int(outcome == "edge"),
                              found_second=int(exposures["second"].get(key) == "edge"))
    result = {"windows": total, "windows_complete_in_both": kept, **counts}
    if counts["gold_exposures"]:
        result["recall_first"] = counts["found_first"] / counts["gold_exposures"]
        result["recall_second"] = counts["found_second"] / counts["gold_exposures"]
    return result


def compare(config: dict, locked=None) -> dict:
    """Paired comparison of two runs on the same papers. Any omission or failure is
    simply a missing prediction, so no paper is excluded for an outcome of the form.
    The main view is scored twice, as canonicalized (the primary result) and under the
    strict reading, with the same papers, denominators and bootstrap. On locked splits
    both runs must be frozen runs started under the same manifest."""
    settings = config["compare"]
    if locked is not None:
        for run_id in (settings["first"], settings["second"]):
            journal = Path(config["run"]["runs_dir"]) / run_id / LOCKED_JOURNAL
            if run_id not in locked.judge_run_ids or not journal.is_file() or \
                    io.read_json(journal)["manifest_sha256"] != locked.manifest_sha256:
                raise corpus.LockedSplitError(f"{run_id} is not a frozen locked run")
    if settings["report_precision"] and config["data"]["dataset"] == "biotriplex":
        raise ValueError("BioTriplex relation exhaustiveness is unconfirmed: report recall only")
    section = config["stats"]
    bootstrap = stats.BootstrapSettings(section["n_resamples"], section["seed"], section["level"])
    papers = load_given_entities(config["data"], locked is not None)
    selected = sorted(papers)[:settings.get("max_papers")]
    runs_dir = Path(config["run"]["runs_dir"])
    first, second = runs_dir / settings["first"], runs_dir / settings["second"]
    allow_differing = frozenset(settings.get("allow_differing", []))
    _check_comparable(first, second, selected, config["data"], allow_differing)
    view = settings["view"]
    units: list[tuple[Counts, Counts]] = []
    strict_units: list[tuple[Counts, Counts]] = []
    for paper_id in selected:
        document, mentions, gold = papers[paper_id]
        expected = candidates.all_pairs(mentions)
        edges_first = read_edges(first, paper_id, expected, view)
        edges_second = read_edges(second, paper_id, expected, view)
        units.append((scoring.compare_sets(gold, edges_first),
                      scoring.compare_sets(gold, edges_second)))
        if view == "main":
            strict_units.append((scoring.compare_sets(gold, strict_edges(first, paper_id)),
                                 scoring.compare_sets(gold, strict_edges(second, paper_id))))

    def metric(function, index):
        return lambda sample: scoring.micro(function, [unit[index] for unit in sample])

    metrics = {"recall": scoring.recall}
    if settings["report_precision"]:
        metrics.update(precision=scoring.precision, f1=scoring.f1)

    def scored(paired: list[tuple[Counts, Counts]]) -> dict:
        out = {}
        for name, function in metrics.items():
            estimate = stats.paired_difference(paired, metric(function, 0), metric(function, 1),
                                               bootstrap)
            out[name] = {"first": metric(function, 0)(paired),
                         "second": metric(function, 1)(paired),
                         "difference": asdict(estimate),
                         "reading": stats.classify_difference(estimate, settings["delta"])}
        return out

    result: dict = {"first": settings["first"], "second": settings["second"], "view": view,
                    "papers": len(units), "delta": settings["delta"], **scored(units)}
    if allow_differing:
        result["allowed_to_differ"] = sorted(allow_differing)
    if view == "main":
        result["strict_reading"] = scored(strict_units)
        if "input_scope" in allow_differing:
            # The two runs read different windows, so window-matched diagnostics do not
            # apply; the paper-level comparison above is the result.
            result["window_level"] = None
            result["complete_windows_diagnostic"] = None
        else:
            result["window_level"] = _window_level(first, second, papers, selected,
                                                   settings["load_bands"])
            result["complete_windows_diagnostic"] = _complete_windows(first, second, papers,
                                                                      selected)
    return result
