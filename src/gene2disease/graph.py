"""From assertions to pair decisions, graph views and GraphML."""
from __future__ import annotations

from typing import Callable, Iterable
from xml.sax.saxutils import quoteattr

from .schema import Assertion, Certainty, Disposition, PairRecord, Polarity, Provenance, RelationClass

ViewRule = Callable[[Assertion], bool]


def _positive(item: Assertion) -> bool:
    return item.polarity is Polarity.POSITIVE


VIEWS: dict[str, ViewRule] = {
    "main": _positive,
    "stated_only": lambda item: _positive(item) and item.certainty is Certainty.STATED,
    "this_study_only": lambda item: _positive(item) and item.provenance is Provenance.THIS_STUDY,
    "causal_only": lambda item: _positive(item) and item.relation_class is RelationClass.CAUSAL,
}


def decide_pair(gene: str, disease: str, assertions: Iterable[Assertion]) -> PairRecord:
    """An assessed pair is an edge if any assertion passes the main view. Negative
    and positive assertions stay attached, so conflicts remain visible."""
    items = list(assertions)
    if any(VIEWS["main"](item) for item in items):
        disposition = Disposition.EDGE
    elif items:
        disposition = Disposition.NEGATIVE_ONLY
    else:
        disposition = Disposition.NO_ASSERTION
    return PairRecord(gene, disease, disposition, items)


def edges(records: Iterable[PairRecord], view: str = "main") -> list[PairRecord]:
    rule = VIEWS[view]
    return [record for record in records if any(rule(item) for item in record.assertions)]


def to_graphml(records: Iterable[PairRecord], view: str = "main") -> str:
    kept = edges(records, view)
    genes = sorted({record.gene for record in kept})
    diseases = sorted({record.disease for record in kept})
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
             '<key id="category" for="node" attr.name="category" attr.type="string"/>',
             '<key id="assertions" for="edge" attr.name="assertions" attr.type="int"/>',
             '<graph edgedefault="undirected">']
    for category, nodes in (("gene", genes), ("disease", diseases)):
        for node in nodes:
            lines.append(f'<node id={quoteattr(node)}><data key="category">{category}</data></node>')
    for record in kept:
        lines.append(f'<edge source={quoteattr(record.gene)} target={quoteattr(record.disease)}>'
                     f'<data key="assertions">{len(record.assertions)}</data></edge>')
    lines += ["</graph>", "</graphml>"]
    return "\n".join(lines) + "\n"
