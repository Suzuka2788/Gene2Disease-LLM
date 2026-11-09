"""Fixed entity layer: PubTator BioC annotations aligned to the parsed document."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import io
from .corpus import Block, Document, Mention, canonical_surface
from .schema import EntityStatus

EXCLUDED_SECTIONS = frozenset({"REF"})
SECTION_LABELS = frozenset({"TITLE", "ABSTRACT", "INTRO", "METHODS", "RESULTS", "DISCUSS",
                            "CONCL", "CASE", "FIG", "TABLE", "SUPPL", "ABBR", "ACK_FUND",
                            "AUTH_CONT", "COMP_INT", "REVIEW_INFO"})
MIN_PASSAGE_OVERLAP = 0.6
TIE_MARGIN = 0.01
CONTEXT_WORDS = 3
_CATEGORIES = {"Gene": "gene", "Disease": "disease"}
_FOLD = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-",
                       "—": "-", "−": "-", "‘": "'", "’": "'",
                       "“": '"', "”": '"', " ": " "})


@dataclass(frozen=True)
class RawAnnotation:
    category: str
    identifier: str
    text: str
    section_type: str
    passage_text: str
    local_start: int


@dataclass
class AlignmentReport:
    aligned: int = 0
    unaligned: int = 0
    unaligned_by_section: dict[str, int] = field(default_factory=dict)

    def miss(self, section_type: str) -> None:
        self.unaligned += 1
        label = section_type if section_type in SECTION_LABELS else "other"
        self.unaligned_by_section[label] = self.unaligned_by_section.get(label, 0) + 1


def entity_status(identifier: str) -> EntityStatus:
    if not identifier or identifier in {"-", "None"}:
        return EntityStatus.UNNORMALIZED
    if re.search(r"[;,|]", identifier):
        return EntityStatus.AMBIGUOUS
    return EntityStatus.NORMALIZED


def read_pubtator(path: str | Path) -> list[RawAnnotation]:
    """Gene and disease annotations outside the reference list."""
    payload = io.read_json(path)
    documents = payload["PubTator3"] if isinstance(payload, dict) else payload
    annotations = []
    for document in documents:
        for passage in document["passages"]:
            section_type = passage.get("infons", {}).get("section_type", "")
            if section_type in EXCLUDED_SECTIONS:
                continue
            for item in passage.get("annotations", []):
                category = _CATEGORIES.get(item["infons"].get("type"))
                if category is None:
                    continue
                identifier = str(item["infons"].get("identifier") or "")
                for location in item.get("locations", []):
                    annotations.append(RawAnnotation(
                        category, identifier, item.get("text", ""), section_type,
                        passage.get("text", ""), location["offset"] - passage["offset"]))
    return annotations


def fold(text: str) -> str:
    """Length-preserving character folding, so offsets stay valid."""
    return text.translate(_FOLD)


def _word_pattern(surface: str) -> re.Pattern[str]:
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(fold(surface)) + r"(?![A-Za-z0-9])")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", fold(text).lower()))


def entity_key(annotation: RawAnnotation) -> str:
    if entity_status(annotation.identifier) is EntityStatus.UNNORMALIZED:
        return f"{annotation.category}:text:{canonical_surface(annotation.text)}"
    return f"{annotation.category}:{annotation.identifier}"


def ineligible_entities(annotations: list[RawAnnotation]) -> frozenset[str]:
    """Entities that stay on record but never enter the graph or a model request."""
    return frozenset(entity_key(item) for item in annotations
                     if entity_status(item.identifier) is EntityStatus.AMBIGUOUS)


def _ordered_similarity(first: str, second: str) -> float:
    return difflib.SequenceMatcher(None, fold(first).lower(), fold(second).lower(),
                                   autojunk=False).ratio()


def _best_block(passage: str, indexed: list[tuple[Block, set[str]]]) -> Block | None:
    """The block sharing most words. Word sets ignore order, so near-ties are settled
    by ordered similarity. A tie that remains means no block is trusted."""
    wanted = _tokens(passage)
    if not wanted:
        return None
    scored = [(len(wanted & tokens) / len(wanted), block) for block, tokens in indexed]
    top = max((score for score, _ in scored), default=0.0)
    if top < MIN_PASSAGE_OVERLAP:
        return None
    tied = [block for score, block in scored if score >= top - TIE_MARGIN]
    if len(tied) == 1:
        return tied[0]
    ordered = sorted(((_ordered_similarity(passage, block.text), index)
                      for index, block in enumerate(tied)), reverse=True)
    if ordered[0][0] - ordered[1][0] < TIE_MARGIN:
        return None
    return tied[ordered[0][1]]


def _neighbours(text: str, start: int, end: int) -> tuple[list[str], list[str]]:
    """Nearest words first on both sides."""
    before = re.findall(r"[a-z0-9]+", text[:start].lower())[-CONTEXT_WORDS:]
    after = re.findall(r"[a-z0-9]+", text[end:].lower())[:CONTEXT_WORDS]
    return before[::-1], after


def _context_score(first: tuple[list[str], list[str]],
                   second: tuple[list[str], list[str]]) -> int:
    """Words that match at the same distance from the mention."""
    return sum(left == right for side_a, side_b in zip(first, second)
               for left, right in zip(side_a, side_b))


def _choose_occurrence(annotation: RawAnnotation, block: Block,
                       found: list[re.Match[str]]) -> re.Match[str] | None:
    """Exact transfer when the block holds the passage; otherwise the single occurrence
    whose neighbouring words best match. No shared context means no placement."""
    passage = fold(annotation.passage_text)
    text = fold(block.text)
    offset = text.find(passage.strip())
    if offset >= 0:
        wanted = offset + annotation.local_start - (len(passage) - len(passage.lstrip()))
        return next((match for match in found if match.start() == wanted), None)
    end = annotation.local_start + len(annotation.text)
    context = _neighbours(passage, annotation.local_start, end)
    if not any(context):
        return found[0] if len(found) == 1 else None
    scored = sorted(((_context_score(context, _neighbours(text, match.start(), match.end())),
                      index) for index, match in enumerate(found)), reverse=True)
    if scored[0][0] == 0 or (len(scored) > 1 and scored[0][0] == scored[1][0]):
        return None
    return found[scored[0][1]]


def align(document: Document,
          annotations: list[RawAnnotation]) -> tuple[list[Mention], AlignmentReport]:
    """Place annotations in our text.

    BioC passages drop citation markers and fold punctuation, so offsets do not always
    transfer. A passage is matched to one block, or to one whole table for table
    passages, and the occurrence is chosen by exact offset or by local context.
    Anything without a unique match is reported as unaligned.
    """
    report = AlignmentReport()
    mentions: list[Mention] = []
    indexed = [(block, _tokens(block.text)) for block in document.blocks]
    tables: dict[str, list[Block]] = {}
    for block in document.blocks:
        if block.group:
            tables.setdefault(block.group, []).append(block)
    table_tokens = [(Block(name, "", "table", " ".join(b.text for b in rows)),
                     _tokens(" ".join(b.text for b in rows))) for name, rows in tables.items()]
    matched: dict[str, Block | None] = {}
    matched_tables: dict[str, Block | None] = {}

    for annotation in annotations:
        passage = annotation.passage_text
        pattern = _word_pattern(annotation.text)
        if passage not in matched:
            matched[passage] = _best_block(passage, indexed)
        rows = [matched[passage]] if matched[passage] is not None else []
        if not rows and annotation.section_type.startswith("TABLE"):
            if passage not in matched_tables:
                matched_tables[passage] = _best_block(passage, table_tokens)
            table = matched_tables[passage]
            rows = tables[table.block_id] if table is not None else []
        placed = None
        options = [(block, match) for block in rows for match in pattern.finditer(fold(block.text))]
        if len(rows) == 1 and options:
            match = _choose_occurrence(annotation, rows[0], [item[1] for item in options])
            placed = (rows[0], match) if match is not None else None
        elif len(options) == 1:
            placed = options[0]
        elif options:
            context = _neighbours(fold(passage), annotation.local_start,
                                  annotation.local_start + len(annotation.text))
            scored = sorted(((_context_score(context, _neighbours(fold(block.text), match.start(),
                                                                  match.end())), index)
                             for index, (block, match) in enumerate(options)), reverse=True)
            if scored[0][0] > 0 and scored[0][0] > scored[1][0]:
                placed = options[scored[0][1]]
        if placed is None:
            report.miss(annotation.section_type)
            continue
        block, match = placed
        mentions.append(Mention(entity_key(annotation), annotation.category, block.block_id,
                                match.start(), match.end(), annotation.text))
        report.aligned += 1
    return sorted(set(mentions), key=lambda m: (m.block_id, m.start, m.entity)), report


def surfaces(annotations: list[RawAnnotation], category: str) -> set[str]:
    return {canonical_surface(item.text) for item in annotations if item.category == category}
