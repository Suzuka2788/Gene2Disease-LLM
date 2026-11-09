"""F0 report: matched comparisons on common windows and the plan-wide runtime gate.

Reads F0 result files (aggregate) and the per-window records under runs/ (local only).
The returned dictionary is aggregate only.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import io
from .f0 import ANSWERED, _call_seconds, _features, _measured


def _windows(run_dir: Path, condition: str) -> dict[str, dict]:
    return {row["window_id"]: row for row in io.read_jsonl(run_dir / condition / "windows.jsonl")}


def _calls(run_dir: Path, condition: str) -> list[dict]:
    return io.read_jsonl(run_dir / condition / "calls.jsonl")


def matched(run_dir: Path, first: str, second: str, precision_for: set[str],
            second_dir: Path | None = None) -> dict:
    """Both conditions on the windows they share, so a difference is not a sample
    difference. Gold is document-level gold co-occurring in the window. The second
    condition may come from another run on the same windows (a repeat, or another
    cache setting); its label then carries the run name."""
    other = second_dir or run_dir
    label_second = second if second_dir is None else f"{second_dir.name}/{second}"
    if label_second == first:
        label_second = f"{label_second} (second run)"
    one, two = _windows(run_dir, first), _windows(other, second)
    common = sorted(set(one) & set(two))
    calls = {first: [c for c in _calls(run_dir, first) if c["window_id"] in common],
             label_second: [c for c in _calls(other, second) if c["window_id"] in common]}
    second = label_second
    result: dict = {"first": first, "second": second, "common_windows": len(common),
                    "by_dataset": {}}
    failed = {c["window_id"] for side in calls.values() for c in side if c["outcome"] != "ok"}
    complete = [w for w in common if w not in failed]
    # Secondary diagnostic: only windows whose every call parsed in both conditions.
    result["complete_windows_diagnostic"] = {
        "windows": len(complete),
        "gold": sum(one[w]["gold"] for w in complete),
        "found_first": sum(one[w]["gold_found"] for w in complete),
        "found_second": sum(two[w]["gold_found"] for w in complete)}
    for name, rows in ((first, one), (second, two)):
        side = calls[name]
        listed = sum(len(c["pair_outcomes"]) for c in side)
        answered = sum(v in ANSWERED for c in side for v in c["pair_outcomes"].values())
        first_attempts = [c["first_attempt"] for c in side if c.get("first_attempt")]
        result[name] = {
            "calls": len(side),
            "outcomes": dict(Counter(c["outcome"] for c in side)),
            "answered_pair_rate": answered / listed if listed else None,
            "wall_seconds": sum(_call_seconds(c) for c in side if "wall_seconds" in c),
            "completion_tokens": (sum(c.get("completion_tokens") or 0 for c in side)
                                  + sum(a.get("completion_tokens") or 0 for a in first_attempts))}
    for window_id in common:
        dataset = one[window_id]["dataset"]
        entry = result["by_dataset"].setdefault(dataset, Counter())
        entry["windows"] += 1
        entry["gold"] += one[window_id]["gold"]
        for label, rows in (("first", one), ("second", two)):
            entry[f"found_{label}"] += rows[window_id]["gold_found"]
            entry[f"strict_found_{label}"] += rows[window_id].get("strict_gold_found", 0)
            entry[f"edges_{label}"] += rows[window_id]["predicted_edges"]
            entry[f"edges_in_gold_{label}"] += rows[window_id]["predicted_in_gold"]
        pairs_one, pairs_two = one[window_id]["pairs"], two[window_id]["pairs"]
        entry["pairs"] += len(pairs_one)
        entry["same_pair_outcome"] += sum(pairs_one[key] == pairs_two.get(key) for key in pairs_one)
    for dataset, entry in result["by_dataset"].items():
        entry = dict(entry)
        for label in ("first", "second"):
            entry[f"recall_{label}"] = entry[f"found_{label}"] / entry["gold"] if entry["gold"] else None
            entry[f"strict_recall_{label}"] = (entry[f"strict_found_{label}"] / entry["gold"]
                                               if entry["gold"] else None)
            if dataset in precision_for and entry[f"edges_{label}"]:
                entry[f"edges_matching_gold_{label}"] = (entry[f"edges_in_gold_{label}"]
                                                        / entry[f"edges_{label}"])
        entry["pair_agreement"] = entry["same_pair_outcome"] / entry["pairs"] if entry["pairs"] else None
        result["by_dataset"][dataset] = entry
    return result


def validate(fit: dict | None, calls: list[dict], prefix_reuse: bool) -> dict:
    """A timing model fitted on one sample, applied to calls of an independent sample:
    total predicted over total measured seconds, overall and per dataset."""
    rows = _measured(calls)
    if fit is None or not rows:
        return {"calls": len(rows), "predicted_over_measured": None}
    sums: dict[str, list[float]] = {}
    for call in rows:
        features = _features(call, prefix_reuse)
        guess = (fit["intercept"] + fit.get("per_pairs", 0.0) * features["pairs"]
                 + fit.get("per_first_call", 0.0) * features["first_call"]
                 + fit.get("per_new_window_characters", 0.0) * features["new_window_characters"])
        for key in ("all", call.get("dataset", "all")):
            pair = sums.setdefault(key, [0.0, 0.0])
            pair[0] += guess
            pair[1] += _call_seconds(call)
    ratios = {key: guess / measured if measured else None
              for key, (guess, measured) in sorted(sums.items())}
    return {"calls": len(rows), "measured_seconds": sums["all"][1],
            "predicted_over_measured": ratios.pop("all"), "by_dataset": ratios}


def schedule_hours(conditions: dict[str, dict], plan: list[dict], margin: float,
                   correction: dict[str, float] | None = None) -> dict:
    """Plan-wide GPU hours for one model: each plan item takes a workload's projected
    hours from the named condition, times a multiplier (for splits estimated from a
    measured one), then the development-rerun margin is added. With a correction (the
    predicted-over-measured ratio of that condition on an independent sample), each
    item is also divided by it, so an under-predicting model is scaled up."""
    if correction:
        corrected = schedule_hours(
            {name: {**cond, "projected_hours_block": {
                key: {**value, "hours": value["hours"] / correction[name]}
                for key, value in cond["projected_hours_block"].items()}}
             if correction.get(name) else cond for name, cond in conditions.items()},
            plan, margin)
        result = schedule_hours(conditions, plan, margin)
        result["corrected_by_independent_sample"] = {
            "ratios": correction, "hours": corrected["hours"],
            "hours_with_margin": corrected["hours_with_margin"]}
        return result
    items, total, complete = [], 0.0, True
    for entry in plan:
        projected = conditions[entry["condition"]]["projected_hours_block"].get(entry["workload"])
        if projected is None:
            # No usable timing fit for this condition: the schedule is incomplete.
            complete = False
            items.append({"item": entry["item"], "condition": entry["condition"], "hours": None})
            continue
        hours = projected["hours"] * entry["multiplier"]
        total += hours
        items.append({"item": entry["item"], "condition": entry["condition"], "hours": hours})
    return {"items": items, "complete": complete, "hours": total,
            "hours_with_margin": total * (1 + margin)}


def report(config: dict) -> dict:
    """The gate covers the plan section 15 baseline schedule only; the exclusions listed
    in the config (such as S1 and the chunked arm) are reported with it, not costed."""
    settings = config["f0_report"]
    runs_dir = Path(config["run"]["runs_dir"])
    precision_for = set(settings.get("report_precision_for", []))
    models, gate, corrected_gate, complete = {}, 0.0, 0.0, True
    corrections: dict[str, dict[str, float]] = {}
    loaded: dict[str, dict[str, dict]] = {}
    for model in settings["models"]:
        conditions: dict[str, dict] = {}
        for path in model["results"]:
            for condition in io.read_json(path)["conditions"]:
                conditions[condition["condition"]] = condition
        loaded[model["name"]] = conditions
        comparisons = []
        for entry in model["comparisons"]:
            # [first, second, run] within one run, or [run, first, other run, second].
            if len(entry) == 3:
                first, second, run_id = entry
                comparisons.append({"run": run_id, **matched(runs_dir / run_id, first, second,
                                                             precision_for)})
            else:
                run_id, first, other, second = entry
                comparisons.append({"run": run_id, **matched(
                    runs_dir / run_id, first, second, precision_for, runs_dir / other)})
        validations = []
        for condition, run_id, target in model.get("validations", []):
            fit = conditions[condition]["seconds_model"]
            reuse = conditions[condition].get("cache_prompt", True)
            validations.append({"fitted_on": condition, "checked_on": f"{run_id}/{target}",
                                **validate(fit, _calls(runs_dir / run_id, target), reuse)})
        correction = {item["fitted_on"]: item["predicted_over_measured"]
                      for item in validations if item.get("predicted_over_measured")}
        corrections[model["name"]] = correction
        hours = schedule_hours(conditions, settings["plan"], settings["margin"], correction)
        gate += hours["hours_with_margin"]
        corrected_gate += hours.get("corrected_by_independent_sample",
                                    hours)["hours_with_margin"]
        complete &= hours["complete"]
        models[model["name"]] = {"comparisons": comparisons, "validations": validations,
                                 "schedule": hours}
    # Items that run once (such as the 100-paper run) are charged at the slower model
    # while the model choice is open.
    once = {name: schedule_hours(conditions, settings.get("once_plan", []), settings["margin"],
                                 corrections.get(name))
            for name, conditions in loaded.items()}
    slower = max(once, key=lambda name: once[name]["hours_with_margin"]) if once else None
    if slower is not None:
        gate += once[slower]["hours_with_margin"]
        corrected_gate += max(item.get("corrected_by_independent_sample", item)["hours_with_margin"]
                              for item in once.values())
        complete &= all(item["complete"] for item in once.values())
    return {"models": models, "run_once": {"charged_model": slower, "by_model": once},
            "section15_baseline_hours_all_models_with_margin": gate,
            "section15_baseline_hours_corrected_by_independent_sample": corrected_gate,
            "schedule_complete": complete, "gate_hours": settings["gate_hours"],
            "section15_baseline_gate_met": (complete and gate <= settings["gate_hours"]
                                            and corrected_gate <= settings["gate_hours"]),
            "not_costed": settings.get("not_costed", []),
            "note": ("Projections are fits on the stratified F0 sample, applied to workload "
                     "totals, and also corrected by their predicted/measured ratio on an "
                     "independent random sample where one exists; both are provisional. The "
                     "gate requires both and covers the section 15 baseline schedule only.")}
