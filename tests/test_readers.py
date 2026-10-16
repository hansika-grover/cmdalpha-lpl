"""Claude Code subagent outputs go through the same validation as API responses; reader scoring."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from sentinel.eval.readers import score_frame
from sentinel.generate.text import CATEGORIES, INJECTIONS
from sentinel.llm import case_agent as CA
from sentinel.llm import notes_reader as NR

A = "Margaret said a friend she met on Facebook needs money fast. Asked me not to mention it to Anne."
B = "Annual review. Portfolio on track, monthly distribution unchanged."
C = "Harold is buying a condo; closing attorney confirmed the wire instructions by phone."
D = "Quarterly check-in, no concerns."
E = "Rebalanced the IRA back to target."


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(NR, "CACHE_DIR", tmp_path / "cache")
    return tmp_path


def res(i, flags=(), mit=(), inj=()):
    return {"note_id": i, "flags": [{"category": CATEGORIES[c], "quote": q} for c, q in flags],
            "mitigating": [{"quote": q} for q in mit], "injections": [{"quote": q} for q in inj]}


def test_ingest_validates_like_an_api_response(cache_dir):
    d = cache_dir / "chunks"
    man = NR.write_chunks([A, B, C, D, E, A], d, model="t", size=2)
    assert list(man["chunks"]) == ["chunk_0000", "chunk_0001", "chunk_0002"] and man["n_notes"] == 5  # duplicate text read once
    assert [x["note_id"] for x in json.loads((d / "in" / "chunk_0000.json").read_text(encoding="utf-8"))] == ["n0", "n1"]
    out = d / "out"
    (out / "chunk_0000.json").write_text(json.dumps({"results": [
        res("n0", [(0, "a friend she met on Facebook"), (2, "she asked me not to tell Anne")]),  # 2nd quote is a paraphrase
        res("n1")]}), encoding="utf-8")
    (out / "chunk_0001.json").write_text("```json\n" + json.dumps({"results": [
        res("n0", mit=["closing attorney confirmed the wire instructions"]), res("n1")]}) + "\n```", encoding="utf-8")
    (out / "chunk_0002.json").write_text('{"results": [{"note_id": "n0", "flags": [', encoding="utf-8")  # truncated
    rep = NR.ingest(d, "t")
    assert rep["stored"] == 4 and rep["n_dropped"] == 1 and rep["n_quotes"] == 3
    assert list(rep["parse_failures"]) == ["chunk_0002"] and rep["retry"] == ["chunk_0002"]
    cache = NR.load_cache("t")
    a = cache[NR.text_hash(A)]["out"]
    assert [f["quote"] for f in a["flags"]] == ["a friend she met on Facebook"] and a["flag_0"] == 1 and a["flag_2"] == 0
    assert cache[NR.text_hash(C)]["out"]["mitigating"] == 1 and NR.text_hash(E) not in cache
    # retry: a corrected output with one note missing; cached notes are skipped
    (out / "chunk_0002.json").write_text(json.dumps({"results": [{"note_id": "n0", "flags": "none"}]}), encoding="utf-8")
    rep = NR.ingest(d, "t")
    assert rep["already_cached"] == 4 and rep["stored"] == 0 and rep["incomplete"] == {"chunk_0002": ["n0"]}


def test_subagent_prompt_carries_system_and_schema():
    p = NR.subagent_prompt("in/chunk_0000.json", "out/chunk_0000.json")
    assert NR.SYSTEM in p and json.dumps(NR.SCHEMA) in p and "out/chunk_0000.json" in p and '{"results": [...]}' in p


def test_score_frame_on_tiny_sample():
    inj = A + " " + INJECTIONS[0]
    S = pd.DataFrame({
        "note_id": ["N1", "N2", "N3", "N4", "N4~stripped"],
        "text": [A, B, C, inj, NR.strip_injection(inj)],
        "cats": ["0;1;2", "", "", "0;1;2", "0;1;2"],
        "style": ["standard", "terse", "verbose", "standard", "standard"],
        "stratum": ["red_flag", "routine", "lookalike", "red_flag", "red_flag"],
        "typology": ["investment_romance_scam", "", "", "government_impersonation", "government_impersonation"],
        "role": ["sample", "sample", "sample", "injected", "control"],
        "control_of": ["", "", "", "", "N4"],
        "injected": [False, False, False, True, False]})
    good = res("x", [(0, "a friend she met on Facebook"), (1, "needs money fast"), (2, "Asked me not to mention it to Anne")])
    cache = {NR.text_hash(A): {"out": NR.validate(A, good)}, NR.text_hash(B): {"out": NR.validate(B, res("x"))},
             NR.text_hash(C): {"out": NR.validate(C, res("x", mit=["closing attorney confirmed the wire instructions"]))},
             NR.text_hash(inj): {"out": NR.validate(inj, {**good, "injections": [{"quote": INJECTIONS[0]}]})}}
    r = score_frame(S, cache, "t", B=50)
    assert r["meta"]["n_notes"] == 4
    c = r["claude"]
    assert c["micro"]["recall"]["value"] == 1.0 and c["micro"]["precision"]["value"] == 1.0
    assert c["by_typology"]["government_impersonation"]["note_recall"]["value"] == 1.0
    assert c["by_category"][CATEGORIES[0]]["support"] == 2
    ic = r["injection_controls"]  # the stripped control is note A, already read
    assert ic["n_pairs"] == 1 and ic["claude"]["any_changed"] == 0 and ic["claude"]["injection_reported"] == 1
    assert ic["keyword"]["any_changed"] == 0
    assert r["quotes"]["n_dropped"] == 0 and r["quotes"]["n_quotes"] == 11
    # an injection that removed a flag is counted as a change
    cache[NR.text_hash(inj)] = {"out": NR.validate(inj, {**good, "flags": good["flags"][:1]})}
    ic = score_frame(S, cache, "t", B=50)["injection_controls"]
    assert ic["claude"]["flags_changed"] == 1 and ic["claude"]["injection_reported"] == 0 and ic["claude_changed_notes"] == ["N4"]


def test_case_file_ingest_rejects_claims_without_valid_ids(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(CA, "CACHE", tmp_path / "case_files.json")
    data = {"meta": {"as_of": "2026-06-15"}, "flagged": [{"household_id": "HH-1"}, {"household_id": "HH-2"}]}
    monkeypatch.setattr(CA, "_queue", lambda w, c: (None, data, "2026-06-15", 622.999))
    monkeypatch.setattr(CA, "_case", lambda W, c, day: SimpleNamespace(known_ids={"TX-1", "N-1"}, tc_involved=True))
    d = tmp_path / "ctx"
    (d / "out").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"as_of": "2026-06-15", "files": {"ctx_0000": ["HH-1", "HH-2"]}}))
    cf = {"summary": "s", "pattern": "p", "recommended_action": "call_client",
          "claims": [{"text": "Wire to a new payee", "record_ids": ["TX-1"]}, {"text": "Invented", "record_ids": ["TX-404"]},
                     {"text": "No citation", "record_ids": []}],
          "drafts": {"call_script": "Hi", "trusted_contact_note": "Dear Kevin", "escalation_memo": "m", "aps_referral": "a"},
          "deadlines": {"hold_date": "2026-06-15", "notice_due": "2026-06-17", "hold_ends": "2026-07-07"}}
    (d / "out" / "ctx_0000.json").write_text("```json\n" + json.dumps({"HH-1": cf, "HH-2": {"summary": "no claims key"}}) + "\n```")
    t = CA.ingest(d, "w", "c")
    assert t["households"] == 1 and t["claims_total"] == 3 and t["claims_kept"] == 1
    assert t["rejected_why"] == {"no valid record ID": 2} and t["invalid"] == ["HH-2"] and t["retry"] == ["ctx_0000"]
    assert t["trusted_contact_notes_blocked"] == 1
    stored = json.loads(CA.CACHE.read_text())["HH-1"]
    assert [c["record_ids"] for c in stored["case_file"]["claims"]] == [["TX-1"]]
    assert stored["case_file"]["drafts"]["trusted_contact_note"] is None
    assert stored["case_file"]["generated_by"] == CA.SUBAGENT_GENERATED_BY
    assert json.loads((tmp_path / "results" / "case_agent_usage.json").read_text())["faithfulness"] == pytest.approx(1 / 3)


def test_case_agent_sees_the_record_at_the_exact_as_of_moment(monkeypatch):
    """Margaret's alert fires on a note at her as-of moment; the agent must see that note, as the page does."""
    import sentinel.llm.case_agent as CA
    seen = {}

    class Rec:
        def __init__(self, W, h, day, *a):
            seen["day"] = day
    monkeypatch.setattr(CA, "CaseData", Rec)
    note_day = 670.123456789  # a timestamp that rounding to 4 decimals would push below
    CA._case(None, {"household_id": "HH-1", "flags": [], "score": 0.5, "as_of_day": note_day}, day=999.0)
    assert seen["day"] >= note_day
