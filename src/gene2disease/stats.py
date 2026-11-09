"""The only resampling code in the project.

Seeds and resample counts are required arguments and come from configuration.
Intervals use linear interpolation between order statistics (Hyndman-Fan type 7).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, TypeVar

Unit = TypeVar("Unit")
Statistic = Callable[[Sequence[Unit]], float | None]


@dataclass(frozen=True)
class BootstrapSettings:
    n_resamples: int
    seed: int
    level: float

    def __post_init__(self) -> None:
        if self.n_resamples < 1:
            raise ValueError("n_resamples must be positive")
        if not 0 < self.level < 1:
            raise ValueError("level must be between 0 and 1")


@dataclass(frozen=True)
class Estimate:
    point: float | None
    low: float | None
    high: float | None
    n_units: int
    n_resamples: int
    n_undefined: int


def quantile(values: Sequence[float], probability: float) -> float:
    """Type 7 quantile."""
    if not values:
        raise ValueError("quantile of an empty sequence")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def interval(values: Sequence[float], level: float) -> tuple[float, float]:
    tail = (1 - level) / 2
    return quantile(values, tail), quantile(values, 1 - tail)


def _summarize(point: float | None, draws: list[float | None], n_units: int,
               settings: BootstrapSettings) -> Estimate:
    defined = [draw for draw in draws if draw is not None]
    low, high = interval(defined, settings.level) if defined else (None, None)
    return Estimate(point, low, high, n_units, settings.n_resamples,
                    len(draws) - len(defined))


def bootstrap(units: Sequence[Unit], statistic: Statistic,
              settings: BootstrapSettings) -> Estimate:
    """Resample whole units, normally papers, with replacement."""
    if not units:
        raise ValueError("bootstrap needs at least one unit")
    generator = random.Random(settings.seed)
    draws = [statistic(generator.choices(units, k=len(units)))
             for _ in range(settings.n_resamples)]
    return _summarize(statistic(units), draws, len(units), settings)


def paired_difference(units: Sequence[Unit], statistic_a: Statistic,
                      statistic_b: Statistic,
                      settings: BootstrapSettings) -> Estimate:
    """Interval for statistic_a - statistic_b on the same resampled units."""
    def difference(sample: Sequence[Unit]) -> float | None:
        first, second = statistic_a(sample), statistic_b(sample)
        return None if first is None or second is None else first - second

    return bootstrap(units, difference, settings)


def stratified_bootstrap(strata: Mapping[str, Sequence[Unit]],
                         statistic: Callable[[Mapping[str, Sequence[Unit]]], float | None],
                         settings: BootstrapSettings) -> Estimate:
    """Resample units within each stratum, preserving stratum sizes."""
    if not strata or any(not units for units in strata.values()):
        raise ValueError("every stratum needs at least one unit")
    generator = random.Random(settings.seed)
    names = sorted(strata)
    draws = []
    for _ in range(settings.n_resamples):
        sample = {name: generator.choices(strata[name], k=len(strata[name]))
                  for name in names}
        draws.append(statistic(sample))
    n_units = sum(len(units) for units in strata.values())
    return _summarize(statistic(strata), draws, n_units, settings)


def classify_difference(estimate: Estimate, delta: float) -> str:
    """Mutually exclusive reading of a difference (first method minus second)."""
    if estimate.low is None or estimate.high is None:
        return "inconclusive"
    if estimate.low > delta:
        return "first_meaningfully_higher"
    if estimate.high < -delta:
        return "second_meaningfully_higher"
    if -delta <= estimate.low and estimate.high <= delta:
        return "equivalent_within_delta"
    if estimate.low > 0:
        return "first_higher_magnitude_unresolved"
    if estimate.high < 0:
        return "second_higher_magnitude_unresolved"
    return "inconclusive"


def sample_without_replacement(items: Sequence[Unit], size: int, seed: int) -> list[Unit]:
    """Seeded simple random sample, for pilot and audit selection."""
    if size > len(items):
        raise ValueError("sample size exceeds population")
    return random.Random(seed).sample(list(items), size)
