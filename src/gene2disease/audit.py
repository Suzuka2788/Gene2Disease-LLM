"""Design-based estimates for the two-stage author audit.

Stage 1 samples papers within a paper-type stratum. Stage 2 samples items within a
paper for each item stratum. All estimates are conditional on the fixed entity
inventory.

    T_hat = sum_h (N_h / n_h) sum_i sum_k (M_ik / m_ik) sum_j y_ijk

Intervals resample papers within paper-type strata and keep the same weights, without
finite-population correction. Resampling cannot express uncertainty in a stratum with
no observed positives, so such strata are reported separately and the missed-total and
recall intervals are withheld.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from . import stats

Label = bool | None  # None means the annotator could not decide.


@dataclass(frozen=True)
class Undecided:
    """How undecided labels count. Edges and non-edges move recall in opposite
    directions, so a valid bound assigns them differently."""
    edges: bool
    non_edges: bool


PESSIMISTIC = Undecided(edges=False, non_edges=True)
OPTIMISTIC = Undecided(edges=True, non_edges=False)


def allocate(target: int, sizes: Mapping[str, int]) -> dict[str, int]:
    """Proportional second-stage sample sizes for the selected papers.

    Each paper with items gets at least one, and a paper never gets more than it
    has, so the realised total can differ slightly from the target.
    """
    population = sum(sizes.values())
    if population == 0:
        return {paper: 0 for paper in sizes}
    return {paper: min(size, math.ceil(target * size / population))
            for paper, size in sizes.items()}


@dataclass(frozen=True)
class ItemStratum:
    population: int
    labels: tuple[Label, ...]

    def __post_init__(self) -> None:
        if len(self.labels) > self.population:
            raise ValueError("sample larger than item stratum")
        if self.population and not self.labels:
            raise ValueError("non-empty item stratum has no sampled items")

    def expanded_true(self, undecided_as: bool) -> float:
        if not self.labels:
            return 0.0
        hits = sum(undecided_as if label is None else label for label in self.labels)
        return self.population * hits / len(self.labels)


@dataclass(frozen=True)
class AuditedPaper:
    edges: ItemStratum
    non_edges: Mapping[str, ItemStratum]


@dataclass(frozen=True)
class PaperStratum:
    population: int
    sampled: tuple[AuditedPaper, ...]

    def __post_init__(self) -> None:
        if not self.sampled or len(self.sampled) > self.population:
            raise ValueError("each paper stratum needs 1..population sampled papers")


@dataclass(frozen=True)
class ZeroPositiveStratum:
    sampled: int
    prevalence_upper_bound: float


@dataclass(frozen=True)
class AuditEstimates:
    edge_precision: stats.Estimate
    missed_positive_pairs: float
    conditional_recall: float | None
    missed_interval: stats.Estimate | None
    recall_interval: stats.Estimate | None
    zero_positive_strata: Mapping[str, ZeroPositiveStratum]


def _totals(strata: Mapping[str, Sequence[AuditedPaper]], populations: Mapping[str, int],
            undecided: Undecided) -> tuple[float, float, float]:
    """Estimated emitted edges, true emitted edges, missed positive pairs."""
    emitted = true_edges = missed = 0.0
    for name, papers in strata.items():
        weight = populations[name] / len(papers)
        for paper in papers:
            emitted += weight * paper.edges.population
            true_edges += weight * paper.edges.expanded_true(undecided.edges)
            missed += weight * sum(item.expanded_true(undecided.non_edges)
                                   for item in paper.non_edges.values())
    return emitted, true_edges, missed


def edge_precision(strata, populations, undecided: Undecided) -> float | None:
    emitted, true_edges, _ = _totals(strata, populations, undecided)
    return true_edges / emitted if emitted else None


def missed_total(strata, populations, undecided: Undecided) -> float:
    return _totals(strata, populations, undecided)[2]


def conditional_recall(strata, populations, undecided: Undecided) -> float | None:
    """True emitted edges over true emitted plus missed, given the entity inventory."""
    _, true_edges, missed = _totals(strata, populations, undecided)
    denominator = true_edges + missed
    return true_edges / denominator if denominator else None


def _zero_positive(design: Mapping[str, PaperStratum],
                   required: frozenset[str]) -> dict[str, ZeroPositiveStratum]:
    result = {}
    for name in sorted(required):
        labels = [label for stratum in design.values() for paper in stratum.sampled
                  for label in paper.non_edges[name].labels]
        if labels and not any(label is True for label in labels):
            result[name] = ZeroPositiveStratum(len(labels), min(1.0, 3 / len(labels)))
    return result


def _check_complete(design: Mapping[str, PaperStratum], required: frozenset[str]) -> None:
    for stratum in design.values():
        for paper in stratum.sampled:
            if set(paper.non_edges) != required:
                raise ValueError(f"audited paper must declare exactly {sorted(required)}")


def estimate(design: Mapping[str, PaperStratum], required_strata: frozenset[str],
             settings: stats.BootstrapSettings, undecided: Undecided) -> AuditEstimates:
    """Run with PESSIMISTIC and OPTIMISTIC to bound all three quantities. Every paper
    must declare every non-edge stratum, using size 0 where it has none."""
    _check_complete(design, required_strata)
    strata = {name: stratum.sampled for name, stratum in design.items()}
    populations = {name: stratum.population for name, stratum in design.items()}
    zero_positive = _zero_positive(design, required_strata)

    def interval(function):
        return stats.stratified_bootstrap(
            strata, lambda sample: function(sample, populations, undecided), settings)

    return AuditEstimates(
        edge_precision=interval(edge_precision),
        missed_positive_pairs=missed_total(strata, populations, undecided),
        conditional_recall=conditional_recall(strata, populations, undecided),
        missed_interval=None if zero_positive else interval(missed_total),
        recall_interval=None if zero_positive else interval(conditional_recall),
        zero_positive_strata=zero_positive,
    )
