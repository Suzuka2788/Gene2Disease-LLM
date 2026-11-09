"""Set scoring. An absent paper is an error; a failed paper is a declared state
that the caller must resolve with a policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Literal, Mapping, Sequence

FAILED = None
FailurePolicy = Literal["as_empty", "exclude"]


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def compare_sets(gold: Iterable[Hashable], predicted: Iterable[Hashable]) -> Counts:
    gold_set, predicted_set = set(gold), set(predicted)
    return Counts(len(gold_set & predicted_set), len(predicted_set - gold_set),
                  len(gold_set - predicted_set))


def total(counts: Iterable[Counts]) -> Counts:
    result = Counts(0, 0, 0)
    for item in counts:
        result = result + item
    return result


def precision(counts: Counts) -> float | None:
    denominator = counts.tp + counts.fp
    return counts.tp / denominator if denominator else None


def recall(counts: Counts) -> float | None:
    denominator = counts.tp + counts.fn
    return counts.tp / denominator if denominator else None


def f1(counts: Counts) -> float | None:
    denominator = 2 * counts.tp + counts.fp + counts.fn
    return 2 * counts.tp / denominator if denominator else None


def micro(metric, sample: Sequence[Counts]) -> float | None:
    """Pooled metric over a sample of per-paper counts."""
    return metric(total(sample))


def score_papers(gold: Mapping[str, Iterable[Hashable]],
                 predicted: Mapping[str, Iterable[Hashable] | None],
                 policy: FailurePolicy) -> dict[str, Counts]:
    """Per-paper counts. `predicted[paper] = FAILED` marks a failed paper."""
    missing = sorted(set(gold) - set(predicted))
    extra = sorted(set(predicted) - set(gold))
    if missing or extra:
        raise ValueError(f"paper sets differ: missing={missing[:5]} extra={extra[:5]}")
    scored = {}
    for paper, gold_items in gold.items():
        items = predicted[paper]
        if items is FAILED:
            if policy == "exclude":
                continue
            items = ()
        scored[paper] = compare_sets(gold_items, items)
    return scored
