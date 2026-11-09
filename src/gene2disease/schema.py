"""Output vocabulary. Assertions are labelled first; pair decisions derive from them."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Certainty(StrEnum):
    STATED = "stated"
    HEDGED = "hedged"


class RelationClass(StrEnum):
    CAUSAL = "causal"
    ASSOCIATION = "association"


class Provenance(StrEnum):
    THIS_STUDY = "this_study"
    CITED_BACKGROUND = "cited_background"
    UNCLEAR = "unclear"


class Disposition(StrEnum):
    """Every gene-disease pair ends in exactly one of these."""
    EDGE = "edge"
    NEGATIVE_ONLY = "negative_only"
    NO_ASSERTION = "no_assertion"
    NOT_ASSESSED_NO_COOCCURRENCE = "not_assessed_no_cooccurrence"
    OMITTED_BY_MODEL = "omitted_by_model"
    FAILED = "failed"
    BLOCKED = "blocked"
    INELIGIBLE_ENTITY = "ineligible_entity"


class EntityStatus(StrEnum):
    NORMALIZED = "normalized"
    UNNORMALIZED = "unnormalized"
    AMBIGUOUS = "ambiguous"
    GENE_GROUP = "gene_group"


GRAPH_ELIGIBLE = frozenset({EntityStatus.NORMALIZED, EntityStatus.UNNORMALIZED})


@dataclass(frozen=True)
class Assertion:
    gene: str
    disease: str
    polarity: Polarity
    certainty: Certainty
    relation_class: RelationClass
    provenance: Provenance
    sentence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Entity:
    entity_id: str
    category: str
    status: EntityStatus
    name: str
    candidate_ids: tuple[str, ...] = ()
    taxon: str | None = None


@dataclass
class PairRecord:
    gene: str
    disease: str
    disposition: Disposition
    assertions: list[Assertion] = field(default_factory=list)
    reason: str = ""
