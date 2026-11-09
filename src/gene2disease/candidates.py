"""Evidence windows and the record of every gene-disease pair."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .corpus import Block, Document, Mention

LEVELS = ("sentence", "adjacent", "block", "section", "document")


@dataclass(frozen=True)
class Segment:
    block_id: str
    start: int
    end: int


@dataclass(frozen=True)
class Window:
    window_id: str
    level: str
    segments: tuple[Segment, ...]

    def covers(self, block_id: str, start: int, end: int) -> bool:
        return any(segment.block_id == block_id and start < segment.end
                   and end > segment.start for segment in self.segments)

    def contains(self, block_id: str, start: int, end: int) -> bool:
        return any(segment.block_id == block_id and segment.start <= start
                   and end <= segment.end for segment in self.segments)


@dataclass(frozen=True)
class Candidate:
    window: Window
    genes: frozenset[str]
    diseases: frozenset[str]

    def pairs(self) -> set[tuple[str, str]]:
        return {(gene, disease) for gene in self.genes for disease in self.diseases}


def _whole(block: Block) -> Segment:
    return Segment(block.block_id, 0, len(block.text))


def _sentence_segments(block: Block) -> list[Segment]:
    return [Segment(block.block_id, sentence.start, sentence.end)
            for sentence in block.sentences] or [_whole(block)]


def build_windows(document: Document, level: str) -> list[Window]:
    if level not in LEVELS:
        raise ValueError(f"unknown window level: {level}")
    groups: list[tuple[Segment, ...]] = []
    if level == "sentence":
        for block in document.blocks:
            groups.extend((segment,) for segment in _sentence_segments(block))
    elif level == "adjacent":
        for block in document.blocks:
            segments = _sentence_segments(block)
            if len(segments) == 1:
                groups.append((segments[0],))
            for first, second in zip(segments, segments[1:]):
                groups.append((Segment(block.block_id, first.start, second.end),))
    elif level == "block":
        groups = [(_whole(block),) for block in document.blocks]
    elif level == "section":
        sections: dict[str, list[Segment]] = {}
        for block in document.blocks:
            sections.setdefault(block.section, []).append(_whole(block))
        groups = [tuple(segments) for segments in sections.values()]
    else:
        groups = [tuple(_whole(block) for block in document.blocks)]
    return [Window(f"{document.doc_id}.{level}{index}", level, segments)
            for index, segments in enumerate(groups)]


def build_candidates(document: Document, mentions: Iterable[Mention],
                     level: str) -> list[Candidate]:
    """Windows that hold at least one gene and one disease."""
    by_block: dict[str, list[Mention]] = {}
    for mention in mentions:
        by_block.setdefault(mention.block_id, []).append(mention)
    candidates = []
    for window in build_windows(document, level):
        genes, diseases = set(), set()
        for segment in window.segments:
            for mention in by_block.get(segment.block_id, ()):
                if mention.start < segment.end and mention.end > segment.start:
                    (genes if mention.category == "gene" else diseases).add(mention.entity)
        if genes and diseases:
            candidates.append(Candidate(window, frozenset(genes), frozenset(diseases)))
    return candidates


def all_pairs(mentions: Iterable[Mention]) -> set[tuple[str, str]]:
    items = list(mentions)
    genes = {item.entity for item in items if item.category == "gene"}
    diseases = {item.entity for item in items if item.category == "disease"}
    return {(gene, disease) for gene in genes for disease in diseases}


def covered_pairs(candidates: Iterable[Candidate]) -> set[tuple[str, str]]:
    covered: set[tuple[str, str]] = set()
    for candidate in candidates:
        covered |= candidate.pairs()
    return covered


def window_characters(document: Document, window: Window) -> int:
    texts = {block.block_id: block for block in document.blocks}
    return sum(segment.end - segment.start for segment in window.segments
               if segment.block_id in texts)
