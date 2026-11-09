"""F0: local feasibility on one fixed, stratified sample of development windows.

Every condition (form, output format, thinking, prompt caching) runs on the same
selected windows, so conditions differ only in the setting under test. Per-call records
and per-window outcomes go under runs/, which is never committed; the returned
dictionary is aggregate only.
"""
from __future__ import annotations

import json
import random
import statistics
import subprocess
import threading
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import adjudicate, candidates, corpus, io, runner
from .candidates import Candidate


@dataclass(frozen=True)
class WindowItem:
    dataset: str
    paper_id: str
    candidate: Candidate
    characters: int
    load: int
    stratum: str = ""

    @property
    def window_id(self) -> str:
        return self.candidate.window.window_id


def band(load: int, edges: list[int]) -> str:
    for edge in edges:
        if load <= edge:
            return f"<={edge}"
    return f">{edges[-1]}"


def candidate_windows(name: str, papers: dict, level: str) -> list[WindowItem]:
    items = []
    for paper_id in sorted(papers):
        document, mentions, _ = papers[paper_id]
        for candidate in candidates.build_candidates(document, mentions, level):
            items.append(WindowItem(name, paper_id, candidate,
                                    candidates.window_characters(document, candidate.window),
                                    len(candidate.pairs())))
    return items


def select_windows(items: list[WindowItem], edges: list[int], per_stratum: int,
                   seed: int) -> list[WindowItem]:
    """Strata are load band x (short or long passage, split at the median length), so
    the sample spans both. The heaviest and the longest window are always included."""
    if not items:
        return []
    median = statistics.median(item.characters for item in items)
    strata: dict[str, list[WindowItem]] = {}
    for item in sorted(items, key=lambda value: value.window_id):
        length = "long" if item.characters > median else "short"
        strata.setdefault(f"{band(item.load, edges)}/{length}", []).append(item)
    rng = random.Random(seed)
    chosen: dict[str, WindowItem] = {}
    for name in sorted(strata):
        members = strata[name]
        for item in rng.sample(members, min(per_stratum, len(members))):
            chosen[item.window_id] = WindowItem(item.dataset, item.paper_id, item.candidate,
                                                item.characters, item.load, name)
    for label, key in (("extreme/heaviest", lambda value: (value.load, value.characters)),
                       ("extreme/longest", lambda value: (value.characters, value.load))):
        item = max(items, key=key)
        if item.window_id not in chosen:
            chosen[item.window_id] = WindowItem(item.dataset, item.paper_id, item.candidate,
                                                item.characters, item.load, label)
    return sorted(chosen.values(), key=lambda value: (value.load, value.window_id))


def subset(selection: list[WindowItem], limit: int | None, seed: int = 0) -> list[WindowItem]:
    """At most `limit` windows from each dataset and stratum, drawn at random with the
    seed, so a smaller condition still spans every load band and passage length without
    favouring the lighter windows of a band."""
    if not limit:
        return selection
    groups: dict[tuple[str, str], list[WindowItem]] = {}
    for item in selection:
        groups.setdefault((item.dataset, item.stratum), []).append(item)
    rng = random.Random(seed)
    kept = {item.window_id for key in sorted(groups)
            for item in rng.sample(groups[key], min(limit, len(groups[key])))}
    return [item for item in selection if item.window_id in kept]


class MemorySampler:
    """Peak GPU memory in MiB while a condition runs, from nvidia-smi if present."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _read(self) -> int | None:
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True).stdout
            return max(int(line) for line in output.split() if line.strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._read()
            if value is not None:
                self.peak = max(self.peak or 0, value)
            self._stop.wait(self.interval)

    def __enter__(self) -> "MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *arguments) -> None:
        self._stop.set()
        self._thread.join(timeout=15)


def server_properties(settings: dict) -> dict:
    """Context size and build as the running server reports them."""
    headers = {}
    key = runner.read_api_key(settings)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(f"{settings['base_url'].rstrip('/')}/props", None, headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            props = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return {"available": False}
    generation = props.get("default_generation_settings") or {}
    return {"available": True, "n_ctx": generation.get("n_ctx", props.get("n_ctx")),
            "total_slots": props.get("total_slots"), "build_info": props.get("build_info"),
            "model_alias": props.get("model_alias")}


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    size = len(vector)
    rows = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda index: abs(rows[index][column]))
        if abs(rows[pivot][column]) < 1e-12:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for index in range(size):
            if index != column:
                factor = rows[index][column] / rows[column][column]
                rows[index] = [a - factor * b for a, b in zip(rows[index], rows[column])]
    return [rows[index][size] / rows[index][index] for index in range(size)]


def _features(call: dict, prefix_reuse: bool) -> dict[str, float]:
    """Pairs listed; and, with prefix reuse, a first-call flag and the raw window
    characters on the first call of a window only, because later calls on the same
    window reuse the server's prompt cache for the shared passage. Without reuse every
    call is charged the full window."""
    first = 1.0 if (call.get("first_in_window", True) or not prefix_reuse) else 0.0
    return {"pairs": float(call["pairs"]), "first_call": first,
            "new_window_characters": first * float(call["window_characters"])}


def _call_seconds(call: dict) -> float:
    """A logical call's cost includes a blocked first attempt, if there was one."""
    return float(call["wall_seconds"]) + float((call.get("first_attempt") or {})
                                               .get("wall_seconds") or 0.0)


def _measured(calls: list[dict]) -> list[dict]:
    """Calls with a measurement, from this run or replayed with the original call's
    measurement from the answer cache. A retried call's first attempt is not a row."""
    return [call for call in calls if "wall_seconds" in call and "window_characters" in call]


def fit_seconds(calls: list[dict], prefix_reuse: bool = True) -> dict | None:
    """Least squares of a logical call's wall seconds on the features above. A feature
    that never varies in these calls is dropped, so pairwise fits no per-pair term and
    joint fits no first-call term (both are absorbed by the intercept). The fit is an
    in-sample description of this sample; it is not validated on other windows."""
    rows = _measured(calls)
    if len(rows) < 3:
        return None
    table = [_features(call, prefix_reuse) for call in rows]
    names = [name for name in ("pairs", "first_call", "new_window_characters")
             if len({row[name] for row in table}) > 1]
    features = [[1.0] + [row[name] for name in names] for row in table]
    target = [_call_seconds(call) for call in rows]
    width = len(features[0])
    normal = [[sum(row[i] * row[j] for row in features) for j in range(width)]
              for i in range(width)]
    right = [sum(row[i] * value for row, value in zip(features, target)) for i in range(width)]
    solution = _solve(normal, right)
    if solution is None:
        return None
    fitted = dict(zip(["intercept"] + [f"per_{name}" for name in names], solution))
    predictions = [sum(c * x for c, x in zip(solution, row)) for row in features]
    mean = statistics.fmean(target)
    total = sum((value - mean) ** 2 for value in target)
    residual = sum((value - guess) ** 2 for value, guess in zip(target, predictions))
    fitted["r_squared"] = 1 - residual / total if total else None
    fitted["calls"] = len(rows)
    by_dataset: dict[str, list[float]] = {}
    for call, guess in zip(rows, predictions):
        sums = by_dataset.setdefault(call.get("dataset", "all"), [0.0, 0.0])
        sums[0] += guess
        sums[1] += _call_seconds(call)
    # Total predicted over total measured, per dataset: how well the model transfers
    # between the two kinds of text it was fitted on.
    fitted["predicted_over_measured_by_dataset"] = {
        name: guess / measured if measured else None
        for name, (guess, measured) in sorted(by_dataset.items())}
    return fitted


def project_hours(fit: dict | None, form: str, workload: dict, chars_per_token: float,
                  prefix_reuse: bool = True) -> dict:
    """Projected GPU hours on each workload from the fitted per-call model. Only the
    block level is projected, because F0 samples block windows. Pair exposures count a
    pair once per window it appears in (the uncapped pairwise call count), which is what
    joint calls list; windows are the joint call count; window characters are the sum
    over windows (the joint passage total), or, without prefix reuse in pairwise form,
    the sum over pairwise calls. Chunked form is not projected."""
    if fit is None:
        return {}
    result = {}
    for name, levels in workload.items():
        level = levels["levels"]["block"]
        windows = level["joint_calls"]
        exposures = level["pairwise_calls"]
        characters = level["joint_passage_tokens_estimated"] * chars_per_token
        if form == "pairwise" and not prefix_reuse:
            windows = exposures
            characters = level["pairwise_passage_tokens_estimated"] * chars_per_token
        if form == "pairwise":
            calls = exposures
        elif form == "joint":
            calls = windows
        else:
            continue
        seconds = (fit["intercept"] * calls + fit.get("per_pairs", 0.0) * exposures
                   + fit.get("per_first_call", 0.0) * windows
                   + fit.get("per_new_window_characters", 0.0) * characters)
        result[name] = {"calls": calls, "hours": seconds / 3600}
    return result


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def at(share: float) -> float:
        return ordered[min(len(ordered) - 1, int(share * (len(ordered) - 1) + 0.5))]
    return {"min": ordered[0], "p50": at(0.5), "p90": at(0.9), "max": ordered[-1]}


def _sum(calls: list[dict], key: str) -> float:
    return sum(call.get(key) or 0 for call in calls)


ANSWERED = {"edge", "negative_only", "no_assertion"}


def summarize(name: str, condition: dict, calls: list[dict], windows: list[dict],
              peak_mib: int | None, report_precision: set[str]) -> dict:
    """Token and timing figures come from measured calls: made in this run, or replayed
    from the answer cache with the original call's measurement."""
    real = _measured(calls)
    ok = [call for call in calls if call["outcome"] == "ok"]
    listed = sum(len(call["pair_outcomes"]) for call in calls)
    answered = sum(value in ANSWERED for call in calls for value in call["pair_outcomes"].values())
    retried = [call["first_attempt"] for call in calls if call.get("first_attempt")]
    completion = [call.get("completion_tokens") for call in real if call.get("completion_tokens")]
    budget_use = [call["completion_tokens"] / call["max_tokens"] for call in real
                  if call.get("completion_tokens")]
    generation_ms = _sum(real, "generation_ms")
    prompt_ms = _sum(real, "prompt_ms")
    quality: dict = {}
    for window in windows:
        entry = quality.setdefault(window["dataset"], Counter())
        entry["windows"] += 1
        entry["gold_pairs"] += window["gold"]
        entry["gold_found"] += window["gold_found"]
        entry["strict_gold_found"] += window.get("strict_gold_found", 0)
        entry["strict_predicted_edges"] += window.get("strict_predicted_edges", 0)
        entry["predicted_edges"] += window["predicted_edges"]
        entry["predicted_edges_in_gold"] += window["predicted_in_gold"]
        for outcome, count in window["pair_outcomes"].items():
            entry[f"pairs_{outcome}"] += count
    # Gold is document-level: a gold pair counts in a window when both entities occur in
    # it, whether or not that window states the relation. These are F0 diagnostics, not
    # the EF estimates.
    for dataset, entry in quality.items():
        entry = dict(entry)
        entry["recall_of_cooccurring_document_gold"] = (
            entry["gold_found"] / entry["gold_pairs"] if entry["gold_pairs"] else None)
        entry["strict_recall_of_cooccurring_document_gold"] = (
            entry["strict_gold_found"] / entry["gold_pairs"] if entry["gold_pairs"] else None)
        if dataset in report_precision and entry["predicted_edges"]:
            entry["edges_matching_document_gold"] = (
                entry["predicted_edges_in_gold"] / entry["predicted_edges"])
        quality[dataset] = entry
    return {
        "condition": name, **condition,
        "windows": len(windows), "calls": len(calls), "measured_calls": len(real),
        "cache_hits_this_run": sum(bool(call.get("cache_hit")) for call in calls),
        "retried_calls": len(retried),
        "outcomes": dict(Counter(call["outcome"] for call in calls)),
        "parsed_call_rate": len(ok) / len(calls) if calls else None,
        "answered_pair_rate": answered / listed if listed else None,
        "raw_outcomes": dict(Counter(call.get("raw_outcome", call["outcome"]) for call in calls)),
        "rejected_assertions": _sum(calls, "rejected_assertions"),
        "raw_rejected_assertions": _sum(calls, "raw_rejected_assertions"),
        "pairs_listed": _sum(calls, "pairs"),
        "wall_seconds": _sum(real, "wall_seconds") + _sum(retried, "wall_seconds"),
        # Cost totals include blocked first attempts; the budget statistics below use the
        # answer-producing attempt only. A replayed call loses its first attempt (the
        # block retry sits above the answer cache); `retried_calls` shows if any exist.
        "prompt_tokens": _sum(real, "prompt_tokens") + _sum(retried, "prompt_tokens"),
        "evaluated_prompt_tokens": (_sum(real, "evaluated_prompt_tokens")
                                    + _sum(retried, "evaluated_prompt_tokens")),
        "cached_prompt_tokens": _sum(real, "cached_prompt_tokens"),
        "completion_tokens": _sum(real, "completion_tokens") + _sum(retried, "completion_tokens"),
        "reasoning_characters": _sum(real, "reasoning_characters"),
        "prompt_tokens_per_second": (_sum(real, "evaluated_prompt_tokens") / prompt_ms * 1000
                                     if prompt_ms else None),
        "generation_tokens_per_second": (_sum(real, "completion_tokens") / generation_ms * 1000
                                         if generation_ms else None),
        "completion_tokens_per_call": _quantiles(completion),
        "completion_tokens_per_listed_pair": (
            _sum([c for c in real if c.get("completion_tokens")], "completion_tokens")
            / _sum([c for c in real if c.get("completion_tokens")], "pairs")
            if completion else None),
        "budget_share_used": _quantiles(budget_use),
        "calls_over_80_percent_of_budget": sum(share > 0.8 for share in budget_use),
        "peak_total_gpu_memory_mib_including_desktop": peak_mib,
        "quality": quality,
    }


def _window_record(item: WindowItem, gold: set, outcome: adjudicate.WindowOutcome) -> dict:
    pairs = item.candidate.pairs()
    results = {pair: outcome.pair_outcome(pair) for pair in pairs}
    gold_here = gold & pairs
    edges = {pair for pair, value in results.items() if value == "edge"}
    # Strict reading: the same answers parsed without canonicalization.
    strict_edges = {tuple(key.split("|", 1)) for call in outcome.call_log
                    for key, value in (call.get("raw_pair_outcomes") or {}).items()
                    if value == "edge"}
    return {"window_id": item.window_id, "paper_id": item.paper_id, "dataset": item.dataset,
            "stratum": item.stratum, "load": item.load, "characters": item.characters,
            "gold": len(gold_here), "gold_found": len(gold_here & edges),
            "predicted_edges": len(edges), "predicted_in_gold": len(edges & gold_here),
            "strict_gold_found": len(gold_here & strict_edges),
            "strict_predicted_edges": len(strict_edges),
            "pair_outcomes": dict(Counter(results.values())),
            "pairs": {f"{gene}|{disease}": value for (gene, disease), value in sorted(results.items())}}


def _inference_memory(condition_dir: Path, calls: list[dict], peak: int | None) -> int | None:
    """Peak memory observed while this condition made real model calls. A run served
    entirely from the answer cache keeps the earlier observation instead of its own."""
    path = condition_dir / "memory.json"
    stored = io.read_json(path)["peak_mib"] if path.is_file() else None
    made_calls = any(not call.get("cache_hit") for call in calls)
    if made_calls and peak is not None:
        stored = max(stored or 0, peak)
        io.write_json(path, {"peak_mib": stored})
    return stored


def run(config: dict) -> dict:
    settings = config["f0"]
    level, edges = settings["level"], settings["load_bands"]
    run_dir = Path(config["run"]["runs_dir"]) / settings["run_id"]
    loaded: dict[str, dict] = {}
    selection: list[WindowItem] = []
    # An independent sample leaves out every window an earlier F0 run selected, so a
    # timing model fitted there can be checked on windows it never saw.
    excluded = {row["window_id"] for path in settings.get("exclude_selections", [])
                for row in io.read_jsonl(path)}
    for data in settings["datasets"]:
        papers = runner.load_given_entities(data)
        loaded[data["name"]] = papers
        items = [item for item in candidate_windows(data["name"], papers, level)
                 if item.window_id not in excluded]
        if settings.get("random_windows_per_dataset"):
            # A simple random sample: ordinary windows in their natural proportions,
            # for checking a timing model fitted on the stratified sample.
            count = min(settings["random_windows_per_dataset"], len(items))
            chosen = random.Random(settings["seed"]).sample(
                sorted(items, key=lambda value: value.window_id), count)
            selection += sorted((WindowItem(i.dataset, i.paper_id, i.candidate, i.characters,
                                            i.load, "random") for i in chosen),
                                key=lambda value: (value.load, value.window_id))
        else:
            selection += select_windows(items, edges, settings["windows_per_stratum"],
                                        settings["seed"])
    io.write_jsonl(run_dir / "selection.jsonl", [
        {"dataset": item.dataset, "paper_id": item.paper_id, "window_id": item.window_id,
         "stratum": item.stratum, "load": item.load, "characters": item.characters}
        for item in selection])
    workload = io.read_json(settings["workload_results"])
    projected_workloads = {key: workload[key] for key in settings["project_workloads"]}
    report_precision = set(settings.get("report_precision_for", []))
    # A replay from the answer cache may run while another model is served, or none;
    # the properties seen when this model was served are kept for that case.
    server = server_properties(settings)
    stored = run_dir / "server.json"
    if server.get("available") and server.get("model_alias") in (None, settings["model_id"]):
        io.write_json(stored, server)
    elif stored.is_file():
        server = {**io.read_json(stored), "replayed": True}
    results = []
    for condition in settings["conditions"]:
        name = condition["name"]
        request = runner.request_settings({**settings, **condition})
        windows = subset(selection, condition.get("per_stratum_limit"), settings["seed"])
        identity = {"level": level, "form": condition["form"],
                    "chunk_size": condition.get("chunk_size"), **request,
                    "windows": [item.window_id for item in windows]}
        condition_dir = run_dir / name
        runner._claim_run(condition_dir, identity)
        backend = adjudicate.CachedBackend(
            adjudicate.MeteredBackend(runner.make_backend(settings, request)),
            condition_dir / "cache")
        budget = runner.make_budget(request)
        calls, records = [], []
        with MemorySampler() as memory:
            for item in windows:
                document, mentions, gold = loaded[item.dataset][item.paper_id]
                outcome = adjudicate.judge_window(
                    document, item.candidate, condition["form"], backend,
                    runner._names(mentions), budget, condition.get("chunk_size"),
                    request["output_format"])
                for entry in outcome.call_log:
                    entry.update(dataset=item.dataset, stratum=item.stratum)
                calls += outcome.call_log
                records.append(_window_record(item, gold, outcome))
        io.write_jsonl(condition_dir / "calls.jsonl", calls)
        # Every invocation's records are kept, including failures a later rerun retried.
        history = condition_dir / "calls_history.jsonl"
        previous = io.read_jsonl(history) if history.is_file() else []
        io.write_jsonl(history, previous + calls)
        io.write_jsonl(condition_dir / "windows.jsonl", records)
        summary = summarize(name, {key: value for key, value in condition.items() if key != "name"},
                            calls, records, _inference_memory(condition_dir, calls, memory.peak),
                            report_precision)
        reuse = request["cache_prompt"]
        fit = fit_seconds(calls, reuse)
        summary["seconds_model"] = fit
        summary["projected_hours_block"] = project_hours(
            fit, condition["form"], projected_workloads, workload["chars_per_token_assumed"],
            reuse)
        results.append(summary)
    by_stratum = Counter(item.stratum.split("/")[0] for item in selection)
    return {"model_id": settings["model_id"], "level": level, "server": server,
            "selected_windows": len(selection),
            "selected_by_dataset": dict(Counter(item.dataset for item in selection)),
            "selected_by_load_band": dict(by_stratum),
            "selected_pairs": sum(item.load for item in selection),
            "conditions": results}
