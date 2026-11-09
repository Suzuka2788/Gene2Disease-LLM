"""LLM judgment over candidate windows.

All forms share one instruction text, one window rendering, one output format and one
output-budget rule. They differ only in how many pairs a single call lists: one
(pairwise), up to chunk_size (chunked), or every pair in the window (joint). The output
format (JSON or compact lines) is a run setting shared by every form of that run.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Mapping, Protocol
from urllib.parse import urlparse

from . import candidates, graph, io
from .candidates import Candidate
from .corpus import Document, Mention
from .schema import (Assertion, Certainty, Disposition, PairRecord, Polarity, Provenance,
                     RelationClass)

Form = Literal["pairwise", "chunked", "joint"]
OutputFormat = Literal["json", "lines"]
OUTPUT_FORMATS = ("json", "lines")
Pair = tuple[str, str]
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

TASK = """You read a passage from a biomedical paper and judge gene-disease pairs.
Use only the passage. For each listed pair, report every assertion the passage makes
about that gene and that disease. Report no assertion if the passage makes none.

Fields of an assertion:
- polarity: positive if the passage asserts a relation, including an inverse
  association. negative only if it explicitly denies a relation in the tested comparison.
- certainty: stated, or hedged for may, might, suggest and similar wording.
- relation_class: causal only with intervention, perturbation, pathogenic variant or
  Mendelian randomization evidence for this relation, or an explicit justified causal
  claim. Otherwise association.
- provenance: this_study, cited_background, or unclear.
- sentence_ids: the bracketed ids of the sentences that carry the assertion."""

JSON_FORMAT = """Return one JSON object and nothing else:
{"pairs": [{"gene": "<id>", "disease": "<id>", "assertions": [{"polarity": "...",
"certainty": "...", "relation_class": "...", "provenance": "...",
"sentence_ids": ["..."]}]}]}
The "pairs" list must contain every listed pair exactly once. Use "assertions": [] when
the passage makes no assertion about a pair."""

LINES_FORMAT = """Write one line per assertion and nothing else:
<gene id>|<disease id>|<polarity>|<certainty>|<relation_class>|<provenance>|<sentence ids>
Codes: polarity pos or neg; certainty stated or hedged; relation_class causal or assoc;
provenance this, cited or unclear. Separate several sentence ids with commas.
For a pair with no assertion write exactly one line: <gene id>|<disease id>|none
Every listed pair must appear, either in assertion lines or in one none line."""

# The gene-disease relation rules of the official BioRED annotation guideline
# (ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BioRED_Annotation_Guideline.pdf), restated for the
# fields above. Its entity rules do not apply, because the entities are given. Used only
# for the exploratory comparison of the minimal instruction with guideline rules.
GUIDELINE_RULES = """Rules for when a relation holds (from the BioRED annotation guideline):
- positive only when the passage explicitly states the relation. The gene and the
  disease need not be in the same sentence. If the statement is unclear, report no
  assertion.
- Correlations count, in either direction: more or less of the gene product causes,
  increases, prevents or treats the disease, or knockdown or knockout of the gene changes
  the disease.
- Genetic associations count: a variant of the gene observed in patients with the
  disease, a variant that raises or lowers disease risk, or a disease caused by
  deficiency of the gene product. A confirmed variant-disease association also links the
  variant's gene to the disease.
- Minor associations count, for example an association seen in only a few patients.
- A disease that belongs to the target disease is linked to the same gene.
- Do not report a gene that is only a prognostic factor of the disease, for example
  expression predicting survival, unless the passage states a relation with the disease
  itself.
- Do not link a symptom of the target disease to the gene unless the passage clearly
  relates the symptom to the gene.
- A negative conclusion such as "not significant" or "no correlation was observed" is
  negative.
- provenance: this_study when the relation belongs to the paper's own results or
  conclusions; cited_background when it is context, such as earlier studies."""

INSTRUCTION_SETS = {"minimal": TASK, "biored_guideline": f"{TASK}\n\n{GUIDELINE_RULES}"}
FORMAT_TEXT = {"json": JSON_FORMAT, "lines": LINES_FORMAT}
INSTRUCTIONS = f"{TASK}\n\n{JSON_FORMAT}"

LINE_CODES = {
    "polarity": {"pos": Polarity.POSITIVE, "neg": Polarity.NEGATIVE},
    "certainty": {"stated": Certainty.STATED, "hedged": Certainty.HEDGED},
    "relation_class": {"causal": RelationClass.CAUSAL, "assoc": RelationClass.ASSOCIATION},
    "provenance": {"this": Provenance.THIS_STUDY, "cited": Provenance.CITED_BACKGROUND,
                   "unclear": Provenance.UNCLEAR},
}


class CallFailure(Exception):
    """A model call produced no usable answer."""


class ParseFailure(CallFailure):
    """The response is not the required structure."""


class TruncatedOutput(CallFailure):
    """The model stopped at the token limit."""

    def __init__(self, message: str, characters: int = 0, text: str = "") -> None:
        super().__init__(message)
        self.characters = characters or len(text)
        self.text = text


class BlockedOutput(CallFailure):
    """The provider refused the request."""


class ContextRefused(CallFailure):
    """The server rejected the request because it does not fit the context. Under the
    frozen protocol this is a terminal outcome of the request (no prediction), stored
    in the cache like a truncation, so a resumed run does not send it again."""


CONTEXT_ERROR_TYPE = "exceed_context_size_error"   # llama.cpp's structured error type


def http_failure(code: int, body: str) -> CallFailure:
    """A context refusal is recognized by the structured error type only; anything
    else, including an unparseable body, is an ordinary (operational) call failure."""
    try:
        kind = json.loads(body).get("error", {}).get("type")
    except (ValueError, AttributeError):
        kind = None
    if kind == CONTEXT_ERROR_TYPE:
        return ContextRefused(f"HTTP {code}: {body[:300]}")
    return CallFailure(f"HTTP {code}: {body[:300]}")


@dataclass(frozen=True)
class OutputBudget:
    base: int
    per_pair: int
    extra: int = 0

    def tokens(self, pair_count: int) -> int:
        return self.base + self.per_pair * pair_count + self.extra


class Backend(Protocol):
    """`last` describes the most recent call: usage, timing and cache status, as far as
    the backend knows them. Backends that know nothing leave it empty."""
    model_id: str

    def complete(self, prompt: str, max_tokens: int) -> str: ...


def call_info(backend: Backend) -> dict:
    return dict(getattr(backend, "last", None) or {})


class LocalOpenAIBackend:
    """An OpenAI-compatible server on this machine, such as llama.cpp's server."""

    def __init__(self, base_url: str, model_id: str, timeout: float,
                 api_key: str | None = None, thinking: bool = False,
                 cache_prompt: bool = True) -> None:
        if urlparse(base_url).hostname not in LOCAL_HOSTS:
            raise ValueError(f"judge backend must be local: {base_url}")
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout
        self.api_key = api_key
        self.thinking = thinking
        self.cache_prompt = cache_prompt
        self.last: dict = {}

    def complete(self, prompt: str, max_tokens: int) -> str:
        # With thinking off the output budget holds the answer alone. read_choice rejects
        # reasoning text unless thinking was requested, so a template that ignores the
        # switch shows up as failed calls instead of silently eating the budget.
        body = json.dumps({"model": self.model_id, "temperature": 0, "max_tokens": max_tokens,
                           "cache_prompt": self.cache_prompt,
                           "chat_template_kwargs": {"enable_thinking": self.thinking},
                           "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.base_url}/v1/chat/completions", body, headers)
        started = time.perf_counter()
        self.last = {}
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")[:2000]
            self.last = {"wall_seconds": time.perf_counter() - started}
            raise http_failure(error.code, body) from error
        except (OSError, http.client.HTTPException, ValueError) as error:
            self.last = {"wall_seconds": time.perf_counter() - started}
            raise CallFailure(str(error)) from error
        self.last = {"wall_seconds": time.perf_counter() - started, **usage(payload)}
        return read_choice(payload, allow_reasoning=self.thinking)


def usage(payload: dict) -> dict:
    """Token counts and server timings, when the server reports them."""
    info: dict = {}
    counts = payload.get("usage") if isinstance(payload, dict) else None
    if isinstance(counts, dict):
        info["prompt_tokens"] = counts.get("prompt_tokens")
        info["completion_tokens"] = counts.get("completion_tokens")
    timings = payload.get("timings") if isinstance(payload, dict) else None
    if isinstance(timings, dict):
        info["cached_prompt_tokens"] = timings.get("cache_n")
        info["evaluated_prompt_tokens"] = timings.get("prompt_n")
        info["prompt_ms"] = timings.get("prompt_ms")
        info["generation_ms"] = timings.get("predicted_ms")
    try:
        choice = payload["choices"][0]
        info["finish_reason"] = choice.get("finish_reason")
        info["reasoning_characters"] = len(choice["message"].get("reasoning_content") or "")
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    return info


def read_choice(payload: dict, allow_reasoning: bool = False) -> str:
    try:
        choice = payload["choices"][0]
        reason = choice.get("finish_reason")
        content = choice["message"]["content"]
        reasoning = choice["message"].get("reasoning_content")
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise CallFailure("malformed response envelope") from error
    if reason == "length":
        raise TruncatedOutput("output reached max_tokens",
                              text=content if isinstance(content, str) else "")
    if reason == "content_filter":
        raise BlockedOutput("provider refused the request")
    if reasoning and not allow_reasoning:
        raise CallFailure("model returned reasoning text although thinking is off")
    if not isinstance(content, str):
        raise CallFailure("response has no text")
    return content


class MeteredBackend:
    """Measures real model calls only, so feasibility figures exclude cache hits."""

    def __init__(self, inner: Backend) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.attempts = 0
        self.seconds = 0.0
        self.prompt_characters = 0
        self.response_characters = 0
        self.last: dict = {}

    def complete(self, prompt: str, max_tokens: int) -> str:
        started = time.perf_counter()
        self.attempts += 1
        self.prompt_characters += len(prompt)
        try:
            response = self.inner.complete(prompt, max_tokens)
        except TruncatedOutput as error:
            self.response_characters += error.characters
            raise
        finally:
            self.seconds += time.perf_counter() - started
            self.last = call_info(self.inner)
        self.response_characters += len(response)
        return response


class CachedBackend:
    """Stores each response on disk, so scoring can rerun at no cost. A truncated answer
    is a property of the model and budget, so it is stored too and replayed as a
    truncation; so is a context refusal, and a provider block once its one retry is
    spent. Transport failures are not stored. The original
    call's measurements are stored beside the answer and returned on a hit, marked
    `cache_hit`, so a replay keeps the timing evidence of the call it replays."""

    def __init__(self, inner: Backend, folder: str | Path) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.folder = Path(folder)
        self.hits = 0
        self.last: dict = {}

    def complete(self, prompt: str, max_tokens: int) -> str:
        key = hashlib.sha256(f"{self.model_id}\n{max_tokens}\n{prompt}".encode("utf-8")).hexdigest()
        path = self.folder / f"{key}.txt"
        truncated = self.folder / f"{key}.truncated.txt"
        refused = self.folder / f"{key}.context_refused.txt"
        first_block = self.folder / f"{key}.blocked_once.txt"
        final_block = self.folder / f"{key}.blocked.txt"
        measured = self.folder / f"{key}.meta.json"

        def original() -> dict:
            return io.read_json(measured) if measured.is_file() else {}
        if path.is_file():
            self.hits += 1
            self.last = {**original(), "cache_hit": True}
            return io.read_text(path)
        if truncated.is_file():
            self.hits += 1
            self.last = {**original(), "cache_hit": True}
            raise TruncatedOutput("output reached max_tokens (cached)",
                                  text=io.read_text(truncated))
        if refused.is_file():
            self.hits += 1
            self.last = {**original(), "cache_hit": True}
            raise ContextRefused(f"{io.read_text(refused)} (cached)")
        if final_block.is_file():
            # The one retry was already spent: the block is the request's final outcome.
            self.hits += 1
            self.last = {**original(), "cache_hit": True}
            raise BlockedOutput(f"{io.read_text(final_block)} (cached)")
        try:
            response = self.inner.complete(prompt, max_tokens)
        except TruncatedOutput as error:
            self.last = {**call_info(self.inner), "cache_hit": False}
            io.write_json(measured, call_info(self.inner))
            io.write_text(truncated, error.text or "x" * error.characters)
            raise
        except ContextRefused as error:
            self.last = {**call_info(self.inner), "cache_hit": False}
            io.write_json(measured, call_info(self.inner))
            io.write_text(refused, str(error))
            raise
        except BlockedOutput as error:
            # First block: a marker, so the next call of this request is its one retry,
            # also after a resume. Second block: stored as the final outcome.
            self.last = {**call_info(self.inner), "cache_hit": False}
            if first_block.is_file():
                io.write_json(measured, call_info(self.inner))
                io.write_text(final_block, str(error))
            else:
                io.write_text(first_block, str(error))
            raise
        except CallFailure:
            self.last = {**call_info(self.inner), "cache_hit": False}
            raise
        self.last = {**call_info(self.inner), "cache_hit": False}
        io.write_json(measured, call_info(self.inner))
        io.write_text(path, response)
        return response


def render_window(document: Document, window) -> tuple[str, set[str]]:
    """Sentences with ids. Table rows carry their header line as context."""
    blocks = {block.block_id: block for block in document.blocks}
    lines, sentence_ids = [], set()
    for segment in window.segments:
        block = blocks[segment.block_id]
        if block.context and block.context != block.text:
            lines.append(f"(table header: {block.context})")
        for sentence in block.sentences:
            if sentence.start < segment.end and sentence.end > segment.start:
                lines.append(f"[{sentence.sentence_id}] {block.text[sentence.start:sentence.end]}")
                sentence_ids.add(sentence.sentence_id)
    return "\n".join(lines), sentence_ids


def build_prompt(passage: str, pairs: Iterable[Pair], names: Mapping[str, str],
                 output_format: OutputFormat = "json", instructions: str = "minimal") -> str:
    listed = "\n".join(f'- gene "{gene}" ({names.get(gene, gene)}), '
                       f'disease "{disease}" ({names.get(disease, disease)})'
                       for gene, disease in sorted(pairs))
    return (f"{INSTRUCTION_SETS[instructions]}\n\n{FORMAT_TEXT[output_format]}\n\n"
            f"Passage:\n{passage}\n\n"
            f"Pairs:\n{listed}\n")


@dataclass
class ParsedResponse:
    assertions: list[Assertion] = field(default_factory=list)
    returned: set[Pair] = field(default_factory=set)
    all_rejected: set[Pair] = field(default_factory=set)
    rejected_assertions: int = 0


def _json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ParseFailure("no JSON object in response")
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise ParseFailure(str(error)) from error
    if not isinstance(value, dict) or not isinstance(value.get("pairs"), list):
        raise ParseFailure('response has no "pairs" list')
    return value


def _check_evidence(cited: tuple[str, ...], sentence_ids: set[str]) -> None:
    if not cited or not set(cited) <= sentence_ids:
        raise ValueError("evidence outside the window")


def _sentence_id(value: str, canonical: bool) -> str:
    """Canonical form drops the brackets the passage shows around ids: the F0 smoke run
    showed models copying them. Nothing else about the id changes."""
    if not isinstance(value, str):
        raise TypeError("sentence id must be a string")
    return value.strip().removeprefix("[").removesuffix("]").strip() if canonical else value


def entity_aliases(pairs: Iterable[Pair], names: Mapping[str, str]) -> dict[str, str]:
    """Spellings that name exactly one entity of these pairs: the id, the id without
    its "gene:" or "disease:" prefix, and the displayed label (first mention text),
    compared without case. The F0 run showed models dropping the prefix or echoing the
    label. A spelling shared by two entities is dropped, so it never resolves. Callers
    pass every pair of the window, so the table is the same for every form and request
    size; only pairs of the call itself are then accepted."""
    seen: dict[str, set[str]] = {}
    for entity in {value for pair in pairs for value in pair}:
        spellings = {entity, entity.split(":", 1)[1] if ":" in entity else entity,
                     names.get(entity, entity)}
        for spelling in spellings:
            seen.setdefault(spelling.strip().lower(), set()).add(entity)
    return {spelling: next(iter(ids)) for spelling, ids in seen.items() if len(ids) == 1}


def _resolve(pair: Pair, expected: set[Pair], aliases: Mapping[str, str] | None) -> Pair:
    if pair in expected or not aliases:
        return pair
    return tuple(aliases.get(value.strip().lower(), value) for value in pair)


def _parse_json(text: str, expected: set[Pair], sentence_ids: set[str],
                canonical: bool = True, aliases: Mapping[str, str] | None = None) -> ParsedResponse:
    parsed = ParsedResponse()
    seen: set[Pair] = set()
    for entry in _json_object(text)["pairs"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("assertions"), list):
            raise ParseFailure("pair entry has no assertions list")
        pair = (entry.get("gene"), entry.get("disease"))
        if not all(isinstance(value, str) for value in pair):
            raise ParseFailure("pair ids must be strings")
        if canonical:
            pair = _resolve(pair, expected, aliases)
        if pair not in expected:
            parsed.rejected_assertions += len(entry["assertions"]) or 1
            continue
        if pair in seen:
            raise ParseFailure("pair returned twice")
        seen.add(pair)
        kept = 0
        for item in entry["assertions"]:
            try:
                cited = tuple(_sentence_id(value, canonical) for value in item["sentence_ids"])
                _check_evidence(cited, sentence_ids)
                parsed.assertions.append(Assertion(
                    pair[0], pair[1], Polarity(item["polarity"]), Certainty(item["certainty"]),
                    RelationClass(item["relation_class"]), Provenance(item["provenance"]), cited))
                kept += 1
            except (KeyError, TypeError, ValueError):
                parsed.rejected_assertions += 1
        if kept or not entry["assertions"]:
            parsed.returned.add(pair)
        else:
            parsed.all_rejected.add(pair)
    return parsed


def _parse_lines(text: str, expected: set[Pair], sentence_ids: set[str],
                 canonical: bool = True, aliases: Mapping[str, str] | None = None) -> ParsedResponse:
    parsed = ParsedResponse()
    said_none: set[Pair] = set()
    with_lines: set[Pair] = set()
    kept: set[Pair] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        fields = [value.strip() for value in line.split("|")]
        if len(fields) < 3:
            raise ParseFailure("line is not pipe-separated")
        pair = (fields[0], fields[1])
        if canonical:
            pair = _resolve(pair, expected, aliases)
        if pair not in expected:
            parsed.rejected_assertions += 1
            continue
        # Canonical form reads a none line padded to the full width ("g|d|none|none|...",
        # seen in the F0 smoke run) as the none line it is. A partly filled line stays
        # an assertion line and is judged as one.
        is_none = fields[2:] == ["none"] or (canonical and len(fields) == 7
                                             and all(value == "none" for value in fields[2:]))
        if is_none:
            if pair in said_none or pair in with_lines:
                raise ParseFailure("pair has a none line and another line")
            said_none.add(pair)
            continue
        if pair in said_none:
            raise ParseFailure("pair has a none line and another line")
        with_lines.add(pair)
        try:
            if len(fields) != 7:
                raise ValueError("wrong number of fields")
            cited = tuple(_sentence_id(value, canonical) for value in fields[6].split(",")
                          if value.strip())
            _check_evidence(cited, sentence_ids)
            parsed.assertions.append(Assertion(
                pair[0], pair[1], LINE_CODES["polarity"][fields[2]],
                LINE_CODES["certainty"][fields[3]], LINE_CODES["relation_class"][fields[4]],
                LINE_CODES["provenance"][fields[5]], cited))
            kept.add(pair)
        except (KeyError, ValueError):
            parsed.rejected_assertions += 1
    parsed.returned = said_none | kept
    parsed.all_rejected = with_lines - kept
    return parsed


def parse_response(text: str, expected: set[Pair], sentence_ids: set[str],
                   output_format: OutputFormat = "json", canonical: bool = True,
                   aliases: Mapping[str, str] | None = None) -> ParsedResponse:
    """A structural violation fails the whole call, so `no_assertion` only ever comes
    from an explicit empty answer. A single bad assertion is rejected and counted: it
    must use known labels and cite at least one sentence from this window. A pair whose
    every assertion was rejected is not an answer for that pair. `canonical` accepts
    only the harmless variants listed in _sentence_id, _parse_lines and entity_aliases."""
    if output_format == "json":
        return _parse_json(text, expected, sentence_ids, canonical, aliases)
    if output_format == "lines":
        return _parse_lines(text, expected, sentence_ids, canonical, aliases)
    raise ValueError(f"unknown output format: {output_format}")


def _raw_reading(text: str, pairs: set[Pair], sentence_ids: set[str],
                 output_format: OutputFormat) -> dict:
    """How the same text fares without canonicalization, pair by pair, so strict and
    canonical results can be reported side by side."""
    try:
        raw = parse_response(text, pairs, sentence_ids, output_format, canonical=False)
    except ParseFailure:
        return {"raw_outcome": "parse_failure",
                "raw_pair_outcomes": {_pair_key(pair): "failed" for pair in sorted(pairs)}}
    found: dict[Pair, list[Assertion]] = {}
    for assertion in raw.assertions:
        found.setdefault((assertion.gene, assertion.disease), []).append(assertion)

    def strict(pair: Pair) -> str:
        if pair in raw.returned:
            return graph.decide_pair(*pair, found.get(pair, [])).disposition.value
        return "all_assertions_rejected" if pair in raw.all_rejected else "omitted_by_model"
    return {"raw_outcome": "ok", "raw_returned": len(raw.returned),
            "raw_rejected_assertions": raw.rejected_assertions,
            "raw_pair_outcomes": {_pair_key(pair): strict(pair) for pair in sorted(pairs)}}


@dataclass
class PaperJudgment:
    records: list[PairRecord]
    calls: int
    failed_calls: int
    truncated_calls: int
    blocked_calls: int
    rejected_assertions: int
    call_log: list[dict] = field(default_factory=list)


def _calls(form: Form, candidate: Candidate, chunk_size: int | None) -> list[set[Pair]]:
    pairs = sorted(candidate.pairs())
    size = {"pairwise": 1, "chunked": chunk_size, "joint": len(pairs)}[form]
    if not size or size < 1:
        raise ValueError("chunked form needs a positive chunk_size")
    return [set(pairs[index:index + size]) for index in range(0, len(pairs), size)]


def _complete_with_block_retry(backend: Backend, prompt: str, max_tokens: int,
                               entry: dict) -> str:
    """The project rule: a provider block gets one identical retry and nothing else.
    The entry records how many attempts were made and the first attempt's measurements
    when it was retried, whatever the second attempt does."""
    entry["attempts"] = 1
    try:
        return backend.complete(prompt, max_tokens)
    except BlockedOutput:
        entry["attempts"] = 2
        entry["first_attempt"] = call_info(backend)
        return backend.complete(prompt, max_tokens)


def _pair_key(pair: Pair) -> str:
    return f"{pair[0]}|{pair[1]}"


@dataclass
class WindowOutcome:
    """What one window's calls returned, pair by pair."""
    assertions: dict[Pair, list[Assertion]] = field(default_factory=dict)
    returned: set[Pair] = field(default_factory=set)
    rejected: set[Pair] = field(default_factory=set)
    failed: set[Pair] = field(default_factory=set)
    blocked: set[Pair] = field(default_factory=set)
    calls: int = 0
    failed_calls: int = 0
    truncated_calls: int = 0
    blocked_calls: int = 0
    rejected_assertions: int = 0
    call_log: list[dict] = field(default_factory=list)

    def pair_outcome(self, pair: Pair) -> str:
        if pair in self.returned:
            return graph.decide_pair(*pair, self.assertions.get(pair, [])).disposition.value
        if pair in self.blocked:
            return "blocked"
        if pair in self.failed:
            return "failed"
        if pair in self.rejected:
            return "all_assertions_rejected"
        return "omitted_by_model"


def judge_window(document: Document, candidate: Candidate, form: Form, backend: Backend,
                 names: Mapping[str, str], budget: OutputBudget, chunk_size: int | None = None,
                 output_format: OutputFormat = "json",
                 instructions: str = "minimal") -> WindowOutcome:
    passage, sentence_ids = render_window(document, candidate.window)
    aliases = entity_aliases(candidate.pairs(), names)
    outcome = WindowOutcome()
    for pairs in _calls(form, candidate, chunk_size):
        outcome.calls += 1
        prompt = build_prompt(passage, pairs, names, output_format, instructions)
        max_tokens = budget.tokens(len(pairs))
        # window_characters is the raw window text, as the workload totals count it;
        # passage_characters is the rendered passage with ids and table headers.
        entry = {"window_id": candidate.window.window_id, "form": form, "pairs": len(pairs),
                 "window_pairs": len(candidate.pairs()), "max_tokens": max_tokens,
                 "first_in_window": outcome.calls == 1,
                 "window_characters": candidates.window_characters(document, candidate.window),
                 "prompt_characters": len(prompt), "passage_characters": len(passage)}
        try:
            text = _complete_with_block_retry(backend, prompt, max_tokens, entry)
            parsed = parse_response(text, pairs, sentence_ids, output_format, aliases=aliases)
        except BlockedOutput as error:
            outcome.blocked_calls += 1
            outcome.blocked |= pairs
            entry.update(outcome="blocked", detail=str(error),
                         raw_pair_outcomes={_pair_key(pair): "blocked" for pair in sorted(pairs)})
        except CallFailure as error:
            outcome.failed_calls += 1
            truncated = isinstance(error, TruncatedOutput)
            outcome.truncated_calls += truncated
            outcome.failed |= pairs
            kind = ("truncated" if truncated else
                    "parse_failure" if isinstance(error, ParseFailure) else
                    "context_refused" if isinstance(error, ContextRefused) else "failed")
            entry.update(outcome=kind, detail=str(error)[:300])
            entry["raw_pair_outcomes"] = {_pair_key(pair): "failed" for pair in sorted(pairs)}
            if kind == "parse_failure":
                entry["raw_outcome"] = "parse_failure"
        else:
            entry.update(_raw_reading(text, pairs, sentence_ids, output_format))
            outcome.returned |= parsed.returned
            outcome.rejected |= parsed.all_rejected
            outcome.rejected_assertions += parsed.rejected_assertions
            for assertion in parsed.assertions:
                outcome.assertions.setdefault((assertion.gene, assertion.disease),
                                              []).append(assertion)
            entry.update(outcome="ok", response_characters=len(text),
                         returned=len(parsed.returned), all_rejected=len(parsed.all_rejected),
                         rejected_assertions=parsed.rejected_assertions)
        entry.update(call_info(backend))
        entry["pair_outcomes"] = {_pair_key(pair): outcome.pair_outcome(pair)
                                  for pair in sorted(pairs)}
        outcome.call_log.append(entry)
    return outcome


def judge_paper(document: Document, mentions: list[Mention], level: str, form: Form,
                backend: Backend, names: Mapping[str, str], budget: OutputBudget,
                ineligible: frozenset[str], chunk_size: int | None = None,
                output_format: OutputFormat = "json",
                instructions: str = "minimal") -> PaperJudgment:
    """Every pair ends with one disposition, including pairs never sent to the model.
    A pair returned by any window is decided by what was returned; otherwise a block,
    then a failed call, then an answer whose assertions were all rejected, then an
    omission explains it."""
    eligible = [mention for mention in mentions if mention.entity not in ineligible]
    assertions: dict[Pair, list[Assertion]] = {}
    returned: set[Pair] = set()
    attempted: set[Pair] = set()
    in_failed_call: set[Pair] = set()
    in_blocked_call: set[Pair] = set()
    rejected_pairs: set[Pair] = set()
    calls = failed_calls = truncated_calls = blocked_calls = rejected = 0
    log: list[dict] = []
    for candidate in candidates.build_candidates(document, eligible, level):
        attempted |= candidate.pairs()
        outcome = judge_window(document, candidate, form, backend, names, budget, chunk_size,
                               output_format, instructions)
        calls += outcome.calls
        failed_calls += outcome.failed_calls
        truncated_calls += outcome.truncated_calls
        blocked_calls += outcome.blocked_calls
        rejected += outcome.rejected_assertions
        returned |= outcome.returned
        in_failed_call |= outcome.failed
        in_blocked_call |= outcome.blocked
        rejected_pairs |= outcome.rejected
        for pair, items in outcome.assertions.items():
            assertions.setdefault(pair, []).extend(items)
        log.extend(outcome.call_log)

    records = []
    for pair in sorted(candidates.all_pairs(mentions)):
        gene, disease = pair
        if gene in ineligible or disease in ineligible:
            records.append(PairRecord(gene, disease, Disposition.INELIGIBLE_ENTITY))
        elif pair not in attempted:
            records.append(PairRecord(gene, disease, Disposition.NOT_ASSESSED_NO_COOCCURRENCE))
        elif pair in returned:
            record = graph.decide_pair(gene, disease, assertions.get(pair, []))
            if pair in in_failed_call | in_blocked_call | rejected_pairs:
                record.reason = "some windows failed"
            records.append(record)
        elif pair in in_blocked_call:
            records.append(PairRecord(gene, disease, Disposition.BLOCKED))
        elif pair in in_failed_call:
            records.append(PairRecord(gene, disease, Disposition.FAILED))
        elif pair in rejected_pairs:
            records.append(PairRecord(gene, disease, Disposition.FAILED, [],
                                      "all assertions rejected"))
        else:
            records.append(PairRecord(gene, disease, Disposition.OMITTED_BY_MODEL))
    return PaperJudgment(records, calls, failed_calls, truncated_calls, blocked_calls, rejected,
                         log)
