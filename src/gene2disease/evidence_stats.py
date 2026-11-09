"""Estimates for the evidence assessment, as fixed in annotation/EVIDENCE_PROTOCOL.md §6.

Implementation details the protocol leaves open are listed in IMPLEMENTATION_NOTES and
reported with the results.
"""
from __future__ import annotations

import math
import random
from collections import Counter

from . import stats

Z = 1.959963984540054
IMPLEMENTATION_NOTES = [
    "Finite population correction in Wilson intervals: the sample size is replaced by the "
    "effective size n(N-1)/(N-n); when the whole stratum is sampled (n = N) the interval "
    "collapses to the point estimate.",
    "Newcombe difference intervals use the hybrid score method on the corrected Wilson limits "
    "of the two strata.",
    "Bootstrap percentile intervals use linear interpolation between order statistics; a "
    "resample in which a run has no item in a stratum where it has edges is undefined for that "
    "run and is dropped; the number dropped is reported.",
    "H-E2 pooled proportion: each item weighs one; its value is the mean over its runs; the "
    "Wilson interval for the pooled value uses the number of items.",
]


def wilson(successes: float, n: int, population: int | None = None) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = successes / n
    if population is not None and population <= n:
        return (p, p)
    effective = n * (population - 1) / (population - n) if population is not None else n
    centre = (p + Z * Z / (2 * effective)) / (1 + Z * Z / effective)
    half = Z * math.sqrt(p * (1 - p) / effective + Z * Z / (4 * effective ** 2)) / (1 + Z * Z / effective)
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe(s1: float, n1: int, N1: int | None, s2: float, n2: int, N2: int | None) -> dict:
    """Pass None for N1 and N2 to drop the finite population correction."""
    p1, p2 = s1 / n1, s2 / n2
    l1, u1 = wilson(s1, n1, N1)
    l2, u2 = wilson(s2, n2, N2)
    d = p1 - p2
    return {"difference": d,
            "low": d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            "high": d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)}


def stratum_proportions(items: list[dict], q1: dict[str, str], population: int) -> dict:
    """Q1 support in one stratum: undecided in the denominator (primary), plus the two
    reference values."""
    codes = Counter(q1[item["item_id"]] for item in items)
    n, supported, undecided = len(items), codes["supported"], codes["undecided"]
    out = {"N": population, "n": n, "codes": dict(codes), "supported": supported}
    if n:
        out["primary"] = {"proportion": supported / n, "wilson95": wilson(supported, n, population),
                          "wilson95_without_fpc": wilson(supported, n)}
        out["undecided_as_supported"] = (supported + undecided) / n
        decidable = n - undecided
        out["undecided_excluded"] = supported / decidable if decidable else None
    return out


def run_rates(items_by_stratum: dict[str, list[dict]], q1: dict[str, str],
              n_by_run: dict[str, dict[str, int]], runs: list[str]) -> dict[str, float | None]:
    """Domain estimate per run over the gold-outside strata (S1, S2):
    sum_h N_rh p_rh / sum_h N_rh. None when a stratum with N_rh > 0 has fewer than 3
    sampled items of the run (protocol: not estimated)."""
    rates: dict[str, float | None] = {}
    for run in runs:
        numerator = denominator = 0.0
        estimable = True
        for stratum in ("S1", "S2"):
            population = n_by_run[stratum].get(run, 0)
            if population == 0:
                continue
            mine = [item for item in items_by_stratum[stratum] if run in item["runs"]]
            if len(mine) < 3:
                estimable = False
                break
            numerator += population * sum(q1[item["item_id"]] == "supported" for item in mine) / len(mine)
            denominator += population
        rates[run] = numerator / denominator if estimable and denominator else None
    return rates


def bootstrap_runs(items_by_stratum: dict[str, list[dict]], q1: dict[str, str],
                   n_by_run: dict[str, dict[str, int]], runs: list[str], extra,
                   n_resamples: int = 10000, seed: int = 20261108) -> dict:
    """Stratified bootstrap within S1 and S2, one draw shared by every run, so that
    differences between runs and derived quantities (`extra`, a function of the rates)
    get intervals too."""
    generator = random.Random(seed)
    point = run_rates(items_by_stratum, q1, n_by_run, runs)
    draws: dict[str, list[float]] = {}
    undefined: Counter = Counter()
    for _ in range(n_resamples):
        sample = {stratum: generator.choices(items, k=len(items)) if items else []
                  for stratum, items in items_by_stratum.items()}
        rates = _rates_or_undefined(sample, q1, n_by_run, runs)
        values = {**{f"rate:{run}": rate for run, rate in rates.items()}, **extra(rates)}
        for name, value in values.items():
            if value is None:
                undefined[name] += 1
            else:
                draws.setdefault(name, []).append(value)
    point_values = {**{f"rate:{run}": rate for run, rate in point.items()}, **extra(point)}
    out = {}
    for name, value in point_values.items():
        low_high = stats.interval(draws[name], 0.95) if value is not None and draws.get(name) else None
        out[name] = {"point": value, "percentile95": low_high, "undefined_resamples": undefined[name]}
    return out


def _rates_or_undefined(sample, q1, n_by_run, runs) -> dict[str, float | None]:
    """Like run_rates, but in a resample the only failure is an empty domain."""
    rates = {}
    for run in runs:
        numerator = denominator = 0.0
        defined = True
        for stratum in ("S1", "S2"):
            population = n_by_run[stratum].get(run, 0)
            if population == 0:
                continue
            mine = [item for item in sample[stratum] if run in item["runs"]]
            if not mine:
                defined = False
                break
            numerator += population * sum(q1[item["item_id"]] == "supported" for item in mine) / len(mine)
            denominator += population
        rates[run] = numerator / denominator if defined and denominator else None
    return rates


def citation_gap(items: list[dict], q1: dict[str, str], q2: dict[str, str], rule: str) -> dict:
    """H-E2. Unit: (item, run) with citations of that run in the displayed passage. A run's
    Q2 is supported if any (rule "any", primary) or every (rule "all") of its citation sets
    is judged supported. Reported: among units with Q1 supported, the share whose Q2 is not
    supported. Units with Q1 or Q2 undecided are left out and counted."""
    per_run: dict[str, list[int]] = {}
    per_item: list[float] = []
    left_out = 0
    for item in items:
        if q1[item["item_id"]] == "undecided":
            left_out += len(item["citations"])
            continue
        if q1[item["item_id"]] != "supported":
            continue
        values = []
        for run in sorted(item["citations"]):
            codes = [q2[set_id] for set_id, data in item["citation_sets"].items() if run in data["runs"]]
            supported_sets = [code == "supported" for code in codes if code != "undecided"]
            # A decided set settles the result when it can: any rule, one supported set;
            # all rule, one unsupported set. Otherwise an undecided set leaves the unit out.
            if rule == "any" and any(supported_sets):
                supported = True
            elif rule == "all" and not all(supported_sets):
                supported = False
            elif "undecided" in codes:
                left_out += 1
                continue
            else:
                supported = all(supported_sets) if rule == "all" else False
            gap = int(not supported)
            per_run.setdefault(run, []).append(gap)
            values.append(gap)
        if values:
            per_item.append(sum(values) / len(values))
    runs = {run: {"units": len(gaps), "gap": sum(gaps) / len(gaps), "wilson95": wilson(sum(gaps), len(gaps))}
            for run, gaps in sorted(per_run.items())}
    pooled = sum(per_item) / len(per_item) if per_item else None
    bootstrap = None
    if per_item:
        # Alternative to the heuristic Wilson interval on fractional item values.
        generator = random.Random(20261108)
        draws = [sum(sample) / len(sample) for sample in
                 (generator.choices(per_item, k=len(per_item)) for _ in range(10000))]
        bootstrap = stats.interval(draws, 0.95)
    return {"rule": rule, "by_run": runs,
            "pooled_item_weighted": {"items": len(per_item), "gap": pooled,
                                     "wilson95": wilson(sum(per_item), len(per_item)) if per_item else None,
                                     "item_bootstrap95": bootstrap},
            "left_out_undecided": left_out}
