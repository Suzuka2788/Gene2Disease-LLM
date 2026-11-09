"""Document model plus readers for JATS XML, BioRED and the BioTriplex release."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path

from . import io

BIOTRIPLEX_SPLITS = frozenset({"train", "val", "test"})
LOCKED_BIOTRIPLEX_SPLITS = frozenset({"val", "test"})
BIORED_SPLITS = frozenset({"Train", "Dev", "Test"})
LOCKED_BIORED_SPLITS = frozenset({"Test"})
BIORED_TYPES = {"GeneOrGeneProduct": "gene", "DiseaseOrPhenotypicFeature": "disease"}
NO_RELATION = "no relation"

_ABBREVIATIONS = ("et al.", "e.g.", "i.e.", "vs.", "Fig.", "Figs.", "Dr.", "No.", "cf.",
                  "approx.", "ca.", "Ref.", "Refs.", "Eq.", "St.", "sp.", "spp.", "var.",
                  "subsp.", "ssp.", "p.", "pp.", "vol.", "i.v.", "i.p.", "s.c.")
_BOUNDARY = re.compile(r"[.!?][\"')\]]*\s+(?=[A-Za-z0-9(\[\"'])")
_INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")
_NOT_INLINE = {"fig", "table-wrap", "supplementary-material", "boxed-text", "list",
               "p", "sec", "disp-quote", "table-wrap-foot"}


class LockedSplitError(RuntimeError):
    """A locked evaluation split was requested before the protocol is frozen."""


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    start: int
    end: int


@dataclass
class Block:
    block_id: str
    section: str
    source_type: str
    text: str
    sentences: list[Sentence] = field(default_factory=list)
    context: str = ""
    group: str = ""


@dataclass
class Document:
    doc_id: str
    blocks: list[Block]


@dataclass(frozen=True)
class Mention:
    entity: str
    category: str
    block_id: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class GoldLink:
    """`block_id` is None when the relation trigger has no single-block position."""
    gene: str
    disease: str
    block_id: str | None
    trigger_start: int
    trigger_end: int
    relation: str
    location: str


@dataclass
class AnnotatedDocument:
    document: Document
    mentions: list[Mention]
    links: list[GoldLink]
    negatives: list[GoldLink]
    split: str
    unplaced_spans: int = 0

    def gold_pairs(self) -> set[tuple[str, str]]:
        return {(link.gene, link.disease) for link in self.links}


def collapse_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Rule-based splitter. Lowercase starts are allowed for names such as p53."""
    ranges, start = [], 0
    for match in _BOUNDARY.finditer(text):
        candidate = text[start:match.start() + 1]
        if candidate.endswith(_ABBREVIATIONS) or _INITIAL.search(candidate):
            continue
        ranges.append((start, match.end()))
        start = match.end()
    if start < len(text):
        ranges.append((start, len(text)))
    return [(begin, _rstrip_end(text, begin, end)) for begin, end in ranges]


def _rstrip_end(text: str, begin: int, end: int) -> int:
    while end > begin and text[end - 1].isspace():
        end -= 1
    return end


def _make_block(block_id: str, section: str, source_type: str, text: str,
                context: str = "", group: str = "") -> Block:
    sentences = [Sentence(f"{block_id}.s{index}", start, end)
                 for index, (start, end) in enumerate(split_sentences(text))]
    return Block(block_id, section, source_type, text, sentences, context, group)


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _inline_text(element: ElementTree.Element) -> str:
    parts = [element.text or ""]
    for child in element:
        if _local(child.tag) not in _NOT_INLINE:
            parts.append(_inline_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _cell_text(cell: ElementTree.Element) -> str:
    """Cell text, including paragraphs nested inside the cell."""
    parts = [_inline_text(cell)]
    parts += [_inline_text(item) for item in cell.iter() if _local(item.tag) == "p"]
    return collapse_space(" ".join(parts))


class _JatsReader:
    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        self.blocks: list[Block] = []
        self.tables = 0

    def add(self, section: str, source_type: str, text: str, context: str = "",
            group: str = "") -> None:
        cleaned = collapse_space(text)
        if cleaned:
            block_id = f"{self.doc_id}.b{len(self.blocks)}"
            self.blocks.append(_make_block(block_id, section, source_type, cleaned,
                                           context, group))

    def caption(self, element: ElementTree.Element, section: str, source_type: str,
                group: str = "") -> None:
        for child in element:
            if _local(child.tag) in {"label", "caption"}:
                text = " ".join(_inline_text(item) for item in child
                                if _local(item.tag) in {"p", "title"}) or _inline_text(child)
                self.add(section, source_type, text, group=group)

    def table(self, element: ElementTree.Element, section: str) -> None:
        group = f"table{self.tables}"
        self.tables += 1
        self.caption(element, section, "table_caption", group)
        header = ""
        for row in element.iter():
            if _local(row.tag) != "tr":
                continue
            cells = [_cell_text(cell) for cell in row if _local(cell.tag) in {"td", "th"}]
            if not any(cells):
                continue
            joined = " | ".join(cells)
            if all(_local(cell.tag) == "th" for cell in row) or not header:
                header = joined
            self.add(section, "table_cell", joined, context=header, group=group)
        for child in element.iter():
            if _local(child.tag) == "table-wrap-foot":
                for note in child.iter():
                    if _local(note.tag) == "p":
                        self.add(section, "table_caption", _inline_text(note), group=group)

    def walk(self, element: ElementTree.Element, section: str) -> None:
        for child in element:
            tag = _local(child.tag)
            if tag == "sec":
                title = next((collapse_space(_inline_text(item)) for item in child
                              if _local(item.tag) == "title"), section)
                self.walk(child, title)
            elif tag == "p":
                self.add(section, "paragraph", _inline_text(child))
                self.walk(child, section)
            elif tag == "fig":
                self.caption(child, section, "figure_caption")
            elif tag == "table-wrap":
                self.table(child, section)
            elif tag in {"ref-list", "supplementary-material", "title", "caption", "label"}:
                continue
            else:
                self.walk(child, section)


def parse_jats(path: str | Path) -> Document:
    """Title, abstract, body text, captions, table rows and table notes. Excludes the
    reference list and supplementary files."""
    source = io.require_file(path)
    root = ElementTree.parse(source).getroot()
    reader = _JatsReader(source.stem)
    for element in root.iter():
        if _local(element.tag) == "article-title":
            reader.add("Title", "title", _inline_text(element))
            break
    for element in root.iter():
        if _local(element.tag) == "abstract":
            before = len(reader.blocks)
            reader.walk(element, "Abstract")
            if len(reader.blocks) == before:
                reader.add("Abstract", "paragraph", _inline_text(element))
    for element in root.iter():
        if _local(element.tag) == "body":
            reader.walk(element, "Body")
            break
    return Document(source.stem, reader.blocks)


def read_pmid(path: str | Path) -> str | None:
    root = ElementTree.parse(io.require_file(path)).getroot()
    for element in root.iter():
        if _local(element.tag) == "article-id" and element.get("pub-id-type") == "pmid":
            return (element.text or "").strip() or None
    return None


def canonical_surface(text: str) -> str:
    lowered = collapse_space(text.lower().replace("...", " "))
    return re.sub(r"[‐-―-]+", "-", lowered)


def is_abbreviation(short: str, long: str) -> bool:
    """A short form has at least two capitals or digits, and its letters appear in
    order in the long form, starting at the first letter of a word. Ordinary words
    such as a disease name in parentheses do not qualify."""
    if sum(character.isupper() or character.isdigit() for character in short) < 2:
        return False
    letters = re.sub(r"[^a-z0-9]", "", short.lower())
    target = long.lower()
    if len(letters) < 2 or len(letters) >= len(re.sub(r"[^a-z0-9]", "", target)):
        return False
    if not any(word.startswith(letters[0]) for word in re.findall(r"[a-z0-9]+", target)):
        return False
    position = 0
    for letter in letters:
        position = target.find(letter, position)
        if position < 0:
            return False
        position += 1
    return True


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: str, second: str) -> None:
        self.parent[self.find(second)] = self.find(first)


def parse_spans(spans: str) -> list[tuple[int, int]]:
    """`"10~14,20~24"` gives one range per component of a discontinuous span."""
    return [(int(start), int(end)) for start, end in re.findall(r"(\d+)~(\d+)", spans)]


def _paragraph_blocks(doc_id: str, section_index: int, section: str,
                      text: str) -> list[tuple[int, Block]]:
    blocks, offset = [], 0
    for index, line in enumerate(text.split("\n")):
        if line.strip():
            block_id = f"{doc_id}.x{section_index}.p{index}"
            blocks.append((offset, _make_block(block_id, section, "paragraph", line)))
        offset += len(line) + 1
    return blocks


def _locate(blocks: list[tuple[int, Block]], start: int, end: int) -> tuple[Block, int, int] | None:
    for offset, block in blocks:
        if offset <= start and end <= offset + len(block.text):
            return block, start - offset, end - offset
    return None


def _check_splits(splits: set[str], known: frozenset[str]) -> None:
    """Exact names only: Windows paths ignore case, so "test" would open "Test"."""
    unknown = splits - known
    if unknown:
        raise ValueError(f"unknown split names: {sorted(unknown)}")


def load_biored(folder: str | Path, splits: set[str],
                allow_locked: bool = False) -> dict[str, tuple[Document, list[Mention], set]]:
    """Title and abstract with identifier-level gene-disease pairs. Relations in
    BioRED are document-level, so there is no trigger position."""
    _check_splits(splits, BIORED_SPLITS)
    locked = splits & LOCKED_BIORED_SPLITS
    if locked and not allow_locked:
        raise LockedSplitError(f"locked BioRED splits requested: {sorted(locked)}")
    papers = {}
    for split in sorted(splits):
        release = io.read_json(Path(folder) / f"{split}.BioC.JSON")
        for item in release["documents"]:
            papers[f"BioRED{item['id']}"] = _read_biored_paper(f"BioRED{item['id']}", item)
    return papers


def _read_biored_paper(doc_id: str, item: dict) -> tuple[Document, list[Mention], set]:
    blocks, mentions, categories = [], [], {}
    for index, passage in enumerate(item["passages"]):
        block_id = f"{doc_id}.b{index}"
        blocks.append(_make_block(block_id, passage.get("infons", {}).get("type", "passage"),
                                  "paragraph", passage["text"]))
        for annotation in passage["annotations"]:
            category = BIORED_TYPES.get(annotation["infons"].get("type"))
            if category is None:
                continue
            for identifier in str(annotation["infons"].get("identifier", "")).split(","):
                categories[identifier] = category
                for location in annotation["locations"]:
                    start = location["offset"] - passage["offset"]
                    mentions.append(Mention(f"{category}:{identifier}", category, block_id,
                                            start, start + location["length"],
                                            annotation["text"]))
    pairs = set()
    for relation in item["relations"]:
        first, second = relation["infons"]["entity1"], relation["infons"]["entity2"]
        kinds = {first: categories.get(first), second: categories.get(second)}
        if set(kinds.values()) == {"gene", "disease"}:
            gene = next(key for key, value in kinds.items() if value == "gene")
            disease = next(key for key, value in kinds.items() if value == "disease")
            pairs.add((f"gene:{gene}", f"disease:{disease}"))
    return Document(doc_id, blocks), mentions, pairs


def load_biotriplex(path: str | Path, splits: set[str],
                    allow_locked: bool = False) -> dict[str, AnnotatedDocument]:
    """Read the release file. Entity clusters are per paper because the corpus has
    no identifiers: same-type mentions merge on canonical text and on verified
    "long form (short form)" definitions."""
    _check_splits(splits, BIOTRIPLEX_SPLITS)
    locked = splits & LOCKED_BIOTRIPLEX_SPLITS
    if locked and not allow_locked:
        raise LockedSplitError(f"locked BioTriplex splits requested: {sorted(locked)}")
    release = io.read_json(path)
    return {f"BTX{paper_id}": _read_biotriplex_paper(f"BTX{paper_id}", paper)
            for paper_id, paper in release.items() if paper["split"] in splits}


def _read_biotriplex_paper(doc_id: str, paper: dict) -> AnnotatedDocument:
    blocks: list[Block] = []
    keys: dict[tuple[int, str], str] = {}
    raw_mentions: list[Mention] = []
    pending_links: list[tuple[int, dict, dict]] = []
    located: dict[int, list[tuple[int, Block]]] = {}
    unplaced = 0
    clusters = _UnionFind()

    for index, (section, data) in enumerate(paper["section_data"].items()):
        section_blocks = _paragraph_blocks(doc_id, index, section, data["text"])
        located[index] = section_blocks
        blocks.extend(block for _, block in section_blocks)
        for category, table in (("gene", data["genes"]), ("disease", data["diseases"])):
            for local_id, item in table.items():
                key = f"{category}:{canonical_surface(item['text'])}"
                clusters.find(key)
                keys[(index, local_id)] = key
                for start, end in parse_spans(item["spans"]):
                    place = _locate(section_blocks, start, end)
                    if place is None:
                        unplaced += 1
                        continue
                    block, local_start, local_end = place
                    raw_mentions.append(Mention(key, category, block.block_id, local_start,
                                                local_end, block.text[local_start:local_end]))
        for link in data["entity_linkings"].values():
            pending_links.append((index, link, data["relations"].get(link["relationID"], {})))

    by_block: dict[str, list[Mention]] = {}
    for mention in raw_mentions:
        by_block.setdefault(mention.block_id, []).append(mention)
    texts = {block.block_id: block.text for block in blocks}
    for block_id, items in by_block.items():
        ordered = sorted(items, key=lambda item: item.start)
        for first, second in zip(ordered, ordered[1:]):
            between = texts[block_id][first.end:second.start].strip()
            closes = texts[block_id][second.end:second.end + 1] == ")"
            if first.category != second.category or between != "(" or not closes:
                continue
            shorter, longer = sorted((first.text, second.text), key=len)
            if is_abbreviation(shorter, longer):
                clusters.union(first.entity, second.entity)

    merged = [Mention(clusters.find(item.entity), item.category, item.block_id,
                      item.start, item.end, item.text) for item in raw_mentions]

    links, negatives = [], []
    for index, link, relation in pending_links:
        gene_key = keys.get((index, link["geneID"]))
        disease_key = keys.get((index, link["diseaseID"]))
        if gene_key is None or disease_key is None or "relation" not in relation:
            unplaced += 1
            continue
        components = parse_spans(relation.get("spans", ""))
        whole = (components[0][0], components[-1][1]) if components else None
        place = _locate(located[index], *whole) if whole else None
        block_id, start, end = (place[0].block_id, place[1], place[2]) if place else (None, 0, 0)
        record = GoldLink(clusters.find(gene_key), clusters.find(disease_key), block_id,
                          start, end, relation["relation"], relation.get("location", ""))
        (negatives if relation["relation"] == NO_RELATION else links).append(record)

    return AnnotatedDocument(Document(doc_id, blocks), merged, links, negatives,
                             paper["split"], unplaced)
