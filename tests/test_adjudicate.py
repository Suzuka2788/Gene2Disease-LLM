import io
import json
import tempfile
import unittest
import urllib.error
import xml.etree.ElementTree as ElementTree
from unittest import mock

from gene2disease import adjudicate, graph
from gene2disease.corpus import Block, Document, Mention, Sentence
from gene2disease.schema import Disposition

TEXT = "AAA causes BBB. CCC is unrelated."
DOCUMENT = Document("d", [
    Block("d.b0", "R", "paragraph", TEXT, [Sentence("s0", 0, 15), Sentence("s1", 16, 33)]),
    Block("d.b1", "R", "paragraph", "DDD alone.", [Sentence("s2", 0, 10)])])
MENTIONS = [Mention("gene:a", "gene", "d.b0", 0, 3, "AAA"),
            Mention("disease:b", "disease", "d.b0", 11, 14, "BBB"),
            Mention("disease:c", "disease", "d.b0", 16, 19, "CCC"),
            Mention("disease:d", "disease", "d.b1", 0, 3, "DDD")]
NAMES = {"gene:a": "AAA", "disease:b": "BBB", "disease:c": "CCC", "disease:d": "DDD"}
BUDGET = adjudicate.OutputBudget(200, 60)
NONE = frozenset()


def assertion(sentence="s0", polarity="positive"):
    return {"polarity": polarity, "certainty": "stated", "relation_class": "causal",
            "provenance": "this_study", "sentence_ids": [sentence]}


class Scripted:
    model_id = "scripted"

    def __init__(self, answer):
        self.answer = answer
        self.prompts = []
        self.limits = []

    def complete(self, prompt, max_tokens):
        self.prompts.append(prompt)
        self.limits.append(max_tokens)
        return self.answer(prompt)


def reply(prompt):
    pairs = []
    if '"disease:b"' in prompt:
        pairs.append({"gene": "gene:a", "disease": "disease:b", "assertions": [assertion()]})
    if '"disease:c"' in prompt:
        pairs.append({"gene": "gene:a", "disease": "disease:c", "assertions": []})
    return json.dumps({"pairs": pairs})


def by_pair(judgment):
    return {(r.gene, r.disease): r.disposition for r in judgment.records}


class FormTests(unittest.TestCase):
    def test_forms_differ_only_in_pair_list_and_budget(self):
        first, second = Scripted(reply), Scripted(reply)
        pairwise = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "pairwise",
                                          first, NAMES, BUDGET, NONE)
        joint = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                       second, NAMES, BUDGET, NONE)
        self.assertEqual((pairwise.calls, joint.calls), (2, 1))
        self.assertEqual(by_pair(pairwise), by_pair(joint))
        self.assertEqual(first.limits, [260, 260])
        self.assertEqual(second.limits, [320])

        def without_pair_list(prompt):
            return prompt.split("Pairs:")[0]
        self.assertEqual({without_pair_list(p) for p in first.prompts + second.prompts},
                         {without_pair_list(second.prompts[0])})

    def test_chunked_form_splits_the_pair_list(self):
        mentions = MENTIONS + [Mention("gene:z", "gene", "d.b0", 16, 19, "CCC")]
        backend = Scripted(lambda p: json.dumps({"pairs": []}))
        result = adjudicate.judge_paper(DOCUMENT, mentions, "block", "chunked", backend, NAMES,
                                        BUDGET, NONE, chunk_size=3)
        self.assertEqual(result.calls, 2)
        self.assertEqual(backend.limits, [380, 260])

    def test_chunked_form_needs_a_size(self):
        with self.assertRaises(ValueError):
            adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "chunked", Scripted(reply),
                                   NAMES, BUDGET, NONE)

    def test_every_pair_gets_one_disposition(self):
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                        Scripted(reply), NAMES, BUDGET, NONE)
        self.assertEqual(by_pair(result), {
            ("gene:a", "disease:b"): Disposition.EDGE,
            ("gene:a", "disease:c"): Disposition.NO_ASSERTION,
            ("gene:a", "disease:d"): Disposition.NOT_ASSESSED_NO_COOCCURRENCE})

    def test_joint_omission_is_recorded(self):
        def partial(prompt):
            return json.dumps({"pairs": [{"gene": "gene:a", "disease": "disease:b",
                                          "assertions": [assertion()]}]})
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                        Scripted(partial), NAMES, BUDGET, NONE)
        self.assertIs(by_pair(result)[("gene:a", "disease:c")], Disposition.OMITTED_BY_MODEL)

    def test_malformed_output_is_a_failure_not_a_negative(self):
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                        Scripted(lambda p: "sorry"), NAMES, BUDGET, NONE)
        self.assertEqual(result.failed_calls, 1)
        self.assertIs(by_pair(result)[("gene:a", "disease:b")], Disposition.FAILED)

    def test_truncated_output_is_counted_as_failed(self):
        def stop(prompt):
            raise adjudicate.TruncatedOutput("limit")
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                        Scripted(stop), NAMES, BUDGET, NONE)
        self.assertEqual((result.failed_calls, result.truncated_calls), (1, 1))

    def test_answer_from_one_window_beats_failure_in_another(self):
        two_windows = Document("d", [
            Block("d.b0", "R", "paragraph", "AAA BBB.", [Sentence("s0", 0, 8)]),
            Block("d.b1", "R", "paragraph", "AAA BBB.", [Sentence("s1", 0, 8)])])
        mentions = [Mention("gene:a", "gene", "d.b0", 0, 3, "AAA"),
                    Mention("disease:b", "disease", "d.b0", 4, 7, "BBB"),
                    Mention("gene:a", "gene", "d.b1", 0, 3, "AAA"),
                    Mention("disease:b", "disease", "d.b1", 4, 7, "BBB")]

        def first_only(prompt):
            if "[s0]" in prompt:
                return json.dumps({"pairs": [{"gene": "gene:a", "disease": "disease:b",
                                              "assertions": [assertion()]}]})
            return "bad"
        result = adjudicate.judge_paper(two_windows, mentions, "block", "joint",
                                        Scripted(first_only), NAMES, BUDGET, NONE)
        self.assertIs(result.records[0].disposition, Disposition.EDGE)
        self.assertEqual(result.records[0].reason, "some windows failed")

    def test_transport_failure_keeps_every_pair_on_record(self):
        def down(prompt):
            raise adjudicate.CallFailure("connection refused")
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                        Scripted(down), NAMES, BUDGET, NONE)
        self.assertEqual(len(result.records), 3)
        self.assertIs(by_pair(result)[("gene:a", "disease:b")], Disposition.FAILED)

    def test_block_is_retried_once_identically_then_recorded(self):
        def refuse(prompt):
            raise adjudicate.BlockedOutput("refused")
        backend = Scripted(refuse)
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                        backend, NAMES, BUDGET, NONE)
        self.assertEqual(len(backend.prompts), 2)
        self.assertEqual(backend.prompts[0], backend.prompts[1])
        self.assertEqual((result.blocked_calls, result.failed_calls), (1, 0))
        self.assertIs(by_pair(result)[("gene:a", "disease:b")], Disposition.BLOCKED)

    def test_final_block_is_cached_and_resume_sends_nothing(self):
        def refuse(prompt):
            raise adjudicate.BlockedOutput("refused")
        with tempfile.TemporaryDirectory() as folder:
            inner = Scripted(refuse)
            cached = adjudicate.CachedBackend(inner, folder)
            adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint", cached,
                                   NAMES, BUDGET, NONE)
            self.assertEqual(len(inner.prompts), 2)
            resumed = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                             adjudicate.CachedBackend(inner, folder),
                                             NAMES, BUDGET, NONE)
            self.assertEqual(len(inner.prompts), 2)        # no new call on resume
            self.assertIs(by_pair(resumed)[("gene:a", "disease:b")], Disposition.BLOCKED)

    def test_one_block_then_resume_spends_the_single_retry(self):
        def refuse(prompt):
            raise adjudicate.BlockedOutput("refused")
        with tempfile.TemporaryDirectory() as folder:
            cached = adjudicate.CachedBackend(Scripted(refuse), folder)
            with self.assertRaises(adjudicate.BlockedOutput):
                cached.complete("p", 10)                    # first block, then a crash
            inner = Scripted(reply)
            again = adjudicate.CachedBackend(inner, folder)
            self.assertEqual(again.complete("p", 10), reply("p"))   # the one retry
            self.assertEqual(len(inner.prompts), 1)

    def test_block_retry_success_counts_as_an_answer(self):
        attempts = []

        def once(prompt):
            attempts.append(prompt)
            if len(attempts) == 1:
                raise adjudicate.BlockedOutput("refused")
            return reply(prompt)
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                        Scripted(once), NAMES, BUDGET, NONE)
        self.assertEqual(result.blocked_calls, 0)
        self.assertIs(by_pair(result)[("gene:a", "disease:b")], Disposition.EDGE)

    def test_ineligible_entity_is_kept_and_not_sent(self):
        backend = Scripted(reply)
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint", backend, NAMES,
                                        BUDGET, frozenset({"disease:c"}))
        self.assertIs(by_pair(result)[("gene:a", "disease:c")], Disposition.INELIGIBLE_ENTITY)
        self.assertNotIn('"disease:c"', backend.prompts[0])


class RejectionAndLogTests(unittest.TestCase):
    def test_all_rejected_assertions_are_a_failure_not_no_assertion(self):
        def outside(prompt):
            return json.dumps({"pairs": [
                {"gene": "gene:a", "disease": "disease:b", "assertions": [assertion("s9")]},
                {"gene": "gene:a", "disease": "disease:c", "assertions": []}]})
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                        Scripted(outside), NAMES, BUDGET, NONE)
        records = {(r.gene, r.disease): r for r in result.records}
        self.assertIs(records[("gene:a", "disease:b")].disposition, Disposition.FAILED)
        self.assertEqual(records[("gene:a", "disease:b")].reason, "all assertions rejected")
        self.assertIs(records[("gene:a", "disease:c")].disposition, Disposition.NO_ASSERTION)

    def test_partly_rejected_pair_keeps_its_valid_assertion(self):
        def mixed(prompt):
            return json.dumps({"pairs": [{"gene": "gene:a", "disease": "disease:b",
                                          "assertions": [assertion("s9"), assertion()]}]})
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                        Scripted(mixed), NAMES, BUDGET, NONE)
        self.assertIs(by_pair(result)[("gene:a", "disease:b")], Disposition.EDGE)
        self.assertEqual(result.rejected_assertions, 1)

    def test_call_log_records_each_call_and_pair_outcome(self):
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "pairwise",
                                        Scripted(reply), NAMES, BUDGET, NONE)
        self.assertEqual(len(result.call_log), 2)
        entry = result.call_log[0]
        self.assertEqual((entry["window_id"], entry["pairs"], entry["window_pairs"],
                          entry["max_tokens"], entry["outcome"]), ("d.block0", 1, 2, 260, "ok"))
        outcomes = {key: value for item in result.call_log
                    for key, value in item["pair_outcomes"].items()}
        self.assertEqual(outcomes, {"gene:a|disease:b": "edge", "gene:a|disease:c": "no_assertion"})

    def test_call_log_names_the_failure_kind(self):
        def stop(prompt):
            raise adjudicate.TruncatedOutput("limit")
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                        Scripted(stop), NAMES, BUDGET, NONE)
        self.assertEqual(result.call_log[0]["outcome"], "truncated")
        self.assertEqual(set(result.call_log[0]["pair_outcomes"].values()), {"failed"})
        parse = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                       Scripted(lambda p: "sorry"), NAMES, BUDGET, NONE)
        self.assertEqual(parse.call_log[0]["outcome"], "parse_failure")


def line_reply(prompt):
    lines = []
    if '"disease:b"' in prompt:
        lines.append("gene:a|disease:b|pos|stated|causal|this|s0")
    if '"disease:c"' in prompt:
        lines.append("gene:a|disease:c|none")
    return "\n".join(lines)


class LineFormatTests(unittest.TestCase):
    EXPECTED = {("g", "d"), ("g", "e")}

    def parse(self, text):
        return adjudicate.parse_response(text, self.EXPECTED, {"s0", "s1"}, "lines")

    def test_lines_match_json_dispositions(self):
        lines = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint", Scripted(line_reply),
                                       NAMES, BUDGET, NONE, output_format="lines")
        as_json = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint", Scripted(reply),
                                         NAMES, BUDGET, NONE)
        self.assertEqual(by_pair(lines), by_pair(as_json))

    def test_formats_share_task_and_passage(self):
        first = adjudicate.build_prompt("[s0] x", {("g", "d")}, {}, "json")
        second = adjudicate.build_prompt("[s0] x", {("g", "d")}, {}, "lines")
        self.assertTrue(first.startswith(adjudicate.TASK) and second.startswith(adjudicate.TASK))
        self.assertEqual(first.split("Passage:")[1], second.split("Passage:")[1])

    def test_codes_and_several_sentences(self):
        parsed = self.parse("g|d|neg|hedged|assoc|cited|s0, s1\ng|e|none")
        self.assertEqual(parsed.returned, self.EXPECTED)
        item = parsed.assertions[0]
        self.assertEqual((item.polarity.value, item.certainty.value, item.relation_class.value,
                          item.provenance.value, item.sentence_ids),
                         ("negative", "hedged", "association", "cited_background", ("s0", "s1")))

    def test_several_assertions_for_one_pair(self):
        parsed = self.parse("g|d|pos|stated|assoc|this|s0\ng|d|neg|stated|assoc|this|s1")
        self.assertEqual(len(parsed.assertions), 2)

    def test_missing_pair_is_omitted(self):
        self.assertEqual(self.parse("g|d|none").returned, {("g", "d")})

    def test_none_with_assertion_fails_the_call(self):
        with self.assertRaises(adjudicate.ParseFailure):
            self.parse("g|d|none\ng|d|pos|stated|assoc|this|s0")
        with self.assertRaises(adjudicate.ParseFailure):
            self.parse("g|d|none\ng|d|none")

    def test_prose_fails_the_call_and_fences_are_skipped(self):
        with self.assertRaises(adjudicate.ParseFailure):
            self.parse("Here are the pairs:\ng|d|none")
        self.assertEqual(self.parse("```\ng|d|none\n```").returned, {("g", "d")})

    def test_bad_codes_and_evidence_are_rejected(self):
        parsed = self.parse("g|d|positive|stated|assoc|this|s0\ng|e|pos|stated|assoc|this|s9")
        self.assertEqual(parsed.rejected_assertions, 2)
        self.assertEqual(parsed.all_rejected, self.EXPECTED)
        self.assertEqual(parsed.returned, set())

    def test_canonical_reading_of_seen_variants(self):
        text = "g|d|pos|stated|assoc|this|[s0], [s1]\ng|e|none|none|none|none|none"
        parsed = self.parse(text)
        self.assertEqual(parsed.returned, self.EXPECTED)
        self.assertEqual(parsed.assertions[0].sentence_ids, ("s0", "s1"))
        raw = adjudicate.parse_response(text, self.EXPECTED, {"s0", "s1"}, "lines",
                                        canonical=False)
        self.assertEqual(raw.returned, set())
        self.assertEqual(raw.rejected_assertions, 2)

    def test_partly_none_line_is_still_an_assertion_line(self):
        parsed = self.parse("g|d|pos|none|none|none|s0")
        self.assertEqual((parsed.returned, parsed.all_rejected), (set(), {("g", "d")}))

    def test_entity_spellings_resolve_only_when_unique(self):
        pairs = {("gene:rac1", "disease:cancer"), ("gene:stat1", "disease:cancer"),
                 ("gene:stat1", "disease:lung cancer")}
        aliases = adjudicate.entity_aliases(pairs, {"gene:stat1": "STAT1",
                                                    "disease:lung cancer": "cancer"})
        self.assertEqual(aliases["rac1"], "gene:rac1")
        self.assertEqual(aliases["stat1"], "gene:stat1")
        self.assertNotIn("cancer", aliases)          # names two listed diseases
        text = json.dumps({"pairs": [
            {"gene": "STAT1", "disease": "lung cancer", "assertions": []},
            {"gene": "rac1", "disease": "cancer", "assertions": []}]})
        parsed = adjudicate.parse_response(text, pairs, set(), aliases=aliases)
        self.assertEqual(parsed.returned, {("gene:stat1", "disease:lung cancer")})
        self.assertEqual(parsed.rejected_assertions, 1)
        raw = adjudicate.parse_response(text, pairs, set(), canonical=False, aliases=aliases)
        self.assertEqual(raw.returned, set())

    def test_canonical_json_ids_and_raw_log(self):
        def bracketed(prompt):
            return json.dumps({"pairs": [{"gene": "gene:a", "disease": "disease:b",
                                          "assertions": [assertion("[s0]")]}]})
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "pairwise",
                                        Scripted(bracketed), NAMES, BUDGET, NONE)
        self.assertIs(by_pair(result)[("gene:a", "disease:b")], Disposition.EDGE)
        entry = result.call_log[0]
        self.assertEqual((entry["raw_outcome"], entry["raw_rejected_assertions"],
                          entry["rejected_assertions"]), ("ok", 1, 0))

    def test_unlisted_pair_is_rejected(self):
        parsed = self.parse("x|y|none")
        self.assertEqual((parsed.returned, parsed.rejected_assertions), (set(), 1))


class ParseTests(unittest.TestCase):
    EXPECTED = {("g", "d")}

    def parse(self, item):
        text = json.dumps({"pairs": [{"gene": "g", "disease": "d", "assertions": [item]}]})
        return adjudicate.parse_response(text, self.EXPECTED, {"s0"})

    def test_evidence_outside_window_is_rejected(self):
        parsed = self.parse(assertion("s9"))
        self.assertEqual((len(parsed.assertions), parsed.rejected_assertions), (0, 1))

    def test_empty_evidence_is_rejected(self):
        item = assertion()
        item["sentence_ids"] = []
        self.assertEqual(self.parse(item).rejected_assertions, 1)

    def test_unknown_label_is_rejected(self):
        item = assertion()
        item["polarity"] = "maybe"
        self.assertEqual(self.parse(item).rejected_assertions, 1)

    def test_unlisted_pair_is_rejected(self):
        text = json.dumps({"pairs": [{"gene": "x", "disease": "y", "assertions": [assertion()]}]})
        parsed = adjudicate.parse_response(text, self.EXPECTED, {"s0"})
        self.assertEqual(parsed.returned, set())
        self.assertEqual(parsed.rejected_assertions, 1)

    def test_missing_assertions_field_is_a_failed_call(self):
        text = json.dumps({"pairs": [{"gene": "g", "disease": "d"}]})
        with self.assertRaises(adjudicate.ParseFailure):
            adjudicate.parse_response(text, self.EXPECTED, {"s0"})

    def test_duplicate_pair_is_a_failed_call(self):
        entry = {"gene": "g", "disease": "d", "assertions": []}
        with self.assertRaises(adjudicate.ParseFailure):
            adjudicate.parse_response(json.dumps({"pairs": [entry, entry]}),
                                      self.EXPECTED, {"s0"})

    def test_explicit_empty_list_is_no_assertion(self):
        text = json.dumps({"pairs": [{"gene": "g", "disease": "d", "assertions": []}]})
        self.assertEqual(adjudicate.parse_response(text, self.EXPECTED, set()).returned,
                         {("g", "d")})

    def test_non_string_pair_id_is_a_failed_call(self):
        text = json.dumps({"pairs": [{"gene": ["g"], "disease": "d", "assertions": []}]})
        with self.assertRaises(adjudicate.ParseFailure):
            adjudicate.parse_response(text, self.EXPECTED, {"s0"})

    def test_fenced_json_is_accepted(self):
        text = '```json\n{"pairs": []}\n```'
        self.assertEqual(adjudicate.parse_response(text, self.EXPECTED, set()).returned, set())


class BackendTests(unittest.TestCase):
    def test_remote_backend_is_refused(self):
        with self.assertRaises(ValueError):
            adjudicate.LocalOpenAIBackend("https://api.example.com", "m", 1.0)

    def test_envelope_reasons(self):
        def payload(reason, content="x"):
            return {"choices": [{"finish_reason": reason, "message": {"content": content}}]}
        self.assertEqual(adjudicate.read_choice(payload("stop")), "x")
        with self.assertRaises(adjudicate.TruncatedOutput):
            adjudicate.read_choice(payload("length"))
        with self.assertRaises(adjudicate.BlockedOutput):
            adjudicate.read_choice(payload("content_filter", None))
        with self.assertRaises(adjudicate.CallFailure):
            adjudicate.read_choice(payload("stop", None))
        with self.assertRaises(adjudicate.CallFailure):
            adjudicate.read_choice({"error": "bad"})
        thinking = payload("stop")
        thinking["choices"][0]["message"]["reasoning_content"] = "let me think"
        with self.assertRaises(adjudicate.CallFailure):
            adjudicate.read_choice(thinking)
        empty = payload("stop")
        empty["choices"][0]["message"]["reasoning_content"] = ""
        self.assertEqual(adjudicate.read_choice(empty), "x")
        self.assertEqual(adjudicate.read_choice(thinking, allow_reasoning=True), "x")
        cut = payload("length")
        cut["choices"][0]["message"]["reasoning_content"] = "long thought"
        with self.assertRaises(adjudicate.TruncatedOutput):
            adjudicate.read_choice(cut)

    def test_request_carries_key_switches_and_reads_usage(self):
        seen = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *arguments):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 7},
                    "timings": {"cache_n": 40, "prompt_n": 10, "prompt_ms": 12.5,
                                "predicted_ms": 90.0}}).encode()

        def fake_urlopen(request, timeout):
            seen["body"] = json.loads(request.data)
            seen["auth"] = request.get_header("Authorization")
            return Response()
        backend = adjudicate.LocalOpenAIBackend("http://127.0.0.1:8081", "m", 5, api_key="k",
                                                thinking=False, cache_prompt=False)
        with mock.patch("gene2disease.adjudicate.urllib.request.urlopen", fake_urlopen):
            self.assertEqual(backend.complete("p", 10), "ok")
        self.assertEqual(seen["auth"], "Bearer k")
        self.assertEqual(seen["body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIs(seen["body"]["cache_prompt"], False)
        self.assertEqual((backend.last["prompt_tokens"], backend.last["completion_tokens"],
                          backend.last["cached_prompt_tokens"], backend.last["generation_ms"]),
                         (50, 7, 40, 90.0))

    def test_http_error_detail_is_kept(self):
        def refuse(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {},
                                         io.BytesIO(b'{"error": "context too long"}'))
        backend = adjudicate.LocalOpenAIBackend("http://127.0.0.1:8081", "m", 5)
        with mock.patch("gene2disease.adjudicate.urllib.request.urlopen", refuse):
            with self.assertRaisesRegex(adjudicate.CallFailure, "HTTP 400.*context too long"):
                backend.complete("p", 10)

    def test_cached_truncation_is_replayed(self):
        calls = []

        def stop(prompt):
            calls.append(prompt)
            raise adjudicate.TruncatedOutput("limit", 4)
        with tempfile.TemporaryDirectory() as folder:
            cached = adjudicate.CachedBackend(Scripted(stop), folder)
            for _ in range(2):
                with self.assertRaises(adjudicate.TruncatedOutput):
                    cached.complete("p", 10)
            self.assertEqual((len(calls), cached.hits), (1, 1))
            self.assertTrue(cached.last["cache_hit"])

    def test_cache_hit_returns_the_original_measurement(self):
        class Measured(Scripted):
            def complete(self, prompt, max_tokens):
                self.last = {"wall_seconds": 2.5, "completion_tokens": 9}
                return super().complete(prompt, max_tokens)
        with tempfile.TemporaryDirectory() as folder:
            cached = adjudicate.CachedBackend(Measured(lambda p: "answer"), folder)
            cached.complete("p", 10)
            self.assertFalse(cached.last["cache_hit"])
            cached.complete("p", 10)
            self.assertEqual(cached.last, {"wall_seconds": 2.5, "completion_tokens": 9,
                                           "cache_hit": True})

    def test_retry_logs_both_attempts_even_when_the_second_fails(self):
        class Flaky(Scripted):
            def complete(self, prompt, max_tokens):
                self.last = {"wall_seconds": float(len(self.prompts) + 1)}
                return super().complete(prompt, max_tokens)

        def refuse(prompt):
            raise adjudicate.BlockedOutput("refused")
        result = adjudicate.judge_paper(DOCUMENT, MENTIONS, "block", "joint",
                                        Flaky(refuse), NAMES, BUDGET, NONE)
        entry = result.call_log[0]
        self.assertEqual((entry["attempts"], entry["first_attempt"], entry["wall_seconds"]),
                         (2, {"wall_seconds": 1.0}, 2.0))

    def test_failures_other_than_truncation_are_not_cached(self):
        calls = []

        def down(prompt):
            calls.append(prompt)
            raise adjudicate.CallFailure("refused")
        with tempfile.TemporaryDirectory() as folder:
            cached = adjudicate.CachedBackend(Scripted(down), folder)
            for _ in range(2):
                with self.assertRaises(adjudicate.CallFailure):
                    cached.complete("p", 10)
            self.assertEqual(len(calls), 2)

    def test_meter_ignores_cache_hits(self):
        with tempfile.TemporaryDirectory() as folder:
            metered = adjudicate.MeteredBackend(Scripted(lambda p: "answer"))
            cached = adjudicate.CachedBackend(metered, folder)
            cached.complete("p", 10)
            cached.complete("p", 10)
            self.assertEqual((metered.attempts, cached.hits), (1, 1))
            self.assertEqual(metered.response_characters, 6)

    def test_meter_counts_truncated_output(self):
        def stop(prompt):
            return adjudicate.read_choice({"choices": [{"finish_reason": "length",
                                                        "message": {"content": "x" * 9}}]})
        metered = adjudicate.MeteredBackend(Scripted(stop))
        with self.assertRaises(adjudicate.TruncatedOutput):
            metered.complete("p", 10)
        self.assertEqual(metered.response_characters, 9)

    def test_cache_calls_model_once(self):
        with tempfile.TemporaryDirectory() as folder:
            inner = Scripted(lambda p: "answer")
            cached = adjudicate.CachedBackend(inner, folder)
            self.assertEqual(cached.complete("p", 10), "answer")
            self.assertEqual(cached.complete("p", 10), "answer")
            self.assertEqual(len(inner.prompts), 1)
            cached.complete("p", 20)
            self.assertEqual(len(inner.prompts), 2)


class PromptTests(unittest.TestCase):
    def test_table_header_is_shown(self):
        block = Block("t", "R", "table_cell", "EGFR | lung", [Sentence("t.s0", 0, 11)],
                      context="Gene | Disease")
        document = Document("d", [block])
        window = adjudicate.candidates.build_windows(document, "block")[0]
        passage, ids = adjudicate.render_window(document, window)
        self.assertIn("table header: Gene | Disease", passage)
        self.assertEqual(ids, {"t.s0"})


class GraphTests(unittest.TestCase):
    def records(self):
        hedged = adjudicate.parse_response(json.dumps({"pairs": [
            {"gene": "g", "disease": "d", "assertions": [
                {**assertion(), "certainty": "hedged", "relation_class": "association"},
                assertion(polarity="negative")]}]}), {("g", "d")}, {"s0"})
        return [graph.decide_pair("g", "d", hedged.assertions)]

    def test_conflict_stays_attached_and_views_differ(self):
        records = self.records()
        self.assertIs(records[0].disposition, Disposition.EDGE)
        self.assertEqual(len(records[0].assertions), 2)
        self.assertEqual(len(graph.edges(records, "main")), 1)
        self.assertEqual(graph.edges(records, "stated_only"), [])
        self.assertEqual(graph.edges(records, "causal_only"), [])

    def test_negative_only_is_not_an_edge(self):
        parsed = adjudicate.parse_response(json.dumps({"pairs": [
            {"gene": "g", "disease": "d", "assertions": [assertion(polarity="negative")]}]}),
            {("g", "d")}, {"s0"})
        record = graph.decide_pair("g", "d", parsed.assertions)
        self.assertIs(record.disposition, Disposition.NEGATIVE_ONLY)

    def test_graphml_escapes_ids(self):
        record = self.records()[0]
        record.gene = 'g<&">'
        root = ElementTree.fromstring(graph.to_graphml([record]))
        edge = next(item for item in root.iter() if item.tag.endswith("edge"))
        self.assertEqual(edge.get("source"), 'g<&">')

    def test_prompt_treats_inverse_association_as_positive(self):
        self.assertIn("including an inverse", adjudicate.INSTRUCTIONS)

    def test_guideline_rules_are_added_and_the_minimal_prompt_is_unchanged(self):
        pairs, names = [("gene:1", "disease:2")], {"gene:1": "A", "disease:2": "B"}
        minimal = adjudicate.build_prompt("[p.s1] A causes B.", pairs, names)
        # The minimal prompt must stay byte-identical to the one the cached runs used.
        self.assertTrue(minimal.startswith(f"{adjudicate.TASK}\n\n{adjudicate.JSON_FORMAT}"
                                           "\n\nPassage:\n[p.s1] A causes B.\n\nPairs:\n"))
        guided = adjudicate.build_prompt("[p.s1] A causes B.", pairs, names, "json", "biored_guideline")
        self.assertIn("BioRED annotation guideline", guided)
        self.assertNotIn("BioRED annotation guideline", minimal)
        self.assertEqual(guided.replace(f"\n\n{adjudicate.GUIDELINE_RULES}", ""), minimal)




class ContextRefusalTests(unittest.TestCase):
    def test_only_the_structured_error_type_is_a_context_refusal(self):
        body = json.dumps({"error": {"type": "exceed_context_size_error", "message": "x"}})
        self.assertIsInstance(adjudicate.http_failure(400, body), adjudicate.ContextRefused)
        other = json.dumps({"error": {"type": "server_error",
                                      "message": "exceeds the available context size"}})
        for text in (other, "exceed_context_size_error", "not json"):
            failure = adjudicate.http_failure(400, text)
            self.assertNotIsInstance(failure, adjudicate.ContextRefused)
            self.assertIsInstance(failure, adjudicate.CallFailure)


if __name__ == "__main__":
    unittest.main()
