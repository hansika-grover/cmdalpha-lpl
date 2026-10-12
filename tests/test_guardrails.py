"""Notes-reader and case-agent guardrails (playbook section 12, required tests)."""
import datetime as dt

import pandas as pd

from sentinel.generate.text import CATEGORIES, INJECTIONS
from sentinel.llm.case_agent import add_business_days, compute_deadlines, validate_case_file
from sentinel.llm.keyword_reader import keyword_read
from sentinel.llm.notes_reader import validate

NOTE = "Margaret called asking how fast she could get money out. She asked me not to mention it to Anne."


def test_quote_validator_rejects_non_substrings():
    out = validate(NOTE, {"flags": [
        {"category": CATEGORIES[1], "quote": "asking how fast she could get money out"},     # exact
        {"category": CATEGORIES[2], "quote": "she asked me not to tell Anne"},               # paraphrase
        {"category": CATEGORIES[0], "quote": "met a man on Facebook"},                       # invented
    ], "mitigating": [{"quote": "daughter is aware"}], "injections": []})
    assert [f["quote"] for f in out["flags"]] == ["asking how fast she could get money out"]
    assert out["flag_1"] == 1 and out["flag_2"] == 0 and out["flag_0"] == 0
    assert out["n_dropped"] == 3 and out["mitigating"] == 0


def test_validator_rejects_unknown_category():
    out = validate(NOTE, {"flags": [{"category": "Looks suspicious", "quote": "Margaret called"}], "mitigating": [], "injections": []})
    assert out["flags"] == [] and out["n_dropped"] == 1


def test_business_days_skip_weekends_and_holidays():
    # Fri 2026-10-09 + 2 business days: Mon 10-12 is Columbus Day -> Tue 13, Wed 14
    assert add_business_days(dt.date(2026, 10, 9), 2) == dt.date(2026, 10, 14)
    assert add_business_days(dt.date(2026, 10, 13), 1) == dt.date(2026, 10, 14)
    d = compute_deadlines("2026-10-13")
    assert d["rule_2165_notice_due"] == "2026-10-15"
    assert d["rule_2165_hold_ends"] == "2026-11-03"   # Oct 14-16, 19-23, 26-30, Nov 2-3
    assert d["rule_2166_proposed_delay_ends"] == "2026-10-27"
    assert "rule_2165_notice_due" not in compute_deadlines("2026-10-13", eligible_2165=False)


def test_injected_notes_never_change_keyword_reader_output():
    base = ["Annual review, all fine. Monthly distribution unchanged.",
            "Margaret said a friend she met on Facebook needs help; asked me not to mention it to Anne. Wants bitcoin."]
    rows = []
    for b in base:
        rows.append(b)
        rows += [b + " " + inj for inj in INJECTIONS]
    out = keyword_read(pd.DataFrame({"note_id": [f"N{i}" for i in range(len(rows))], "text": rows}))
    flags = [c for c in out.columns if c.startswith("flag_")] + ["mitigating"]
    k = 1 + len(INJECTIONS)
    for g in range(len(base)):
        block = out.iloc[g * k:(g + 1) * k]
        assert (block[flags].nunique() == 1).all(), "an injection changed the reader's output"
        assert block["injection"].tolist() == [0] + [1] * len(INJECTIONS)


def test_case_agent_never_drafts_to_involved_trusted_contact():
    cf = {"claims": [{"text": "Five transfers to the trusted contact", "record_ids": ["TX-1"]}],
          "drafts": {"call_script": "Hi", "trusted_contact_note": "Dear Kevin, please call me.", "escalation_memo": "m",
                     "aps_referral": "r"}}
    out, rep = validate_case_file(cf, {"TX-1"}, tc_involved=True)
    assert out["drafts"]["trusted_contact_note"] is None and rep["trusted_contact_note_blocked"]
    out, rep = validate_case_file(cf, {"TX-1"}, tc_involved=False)
    assert out["drafts"]["trusted_contact_note"] == "Dear Kevin, please call me."


def test_claims_without_records_or_using_age_are_rejected():
    cf = {"claims": [{"text": "Wire to a new payee", "record_ids": ["TX-9"]},
                     {"text": "Large withdrawal", "record_ids": ["TX-404"]},
                     {"text": "Client is 84 years old and elderly", "record_ids": ["TX-9"]},
                     {"text": "Signs of dementia", "record_ids": ["N-1"]}],
          "drafts": {}}
    out, rep = validate_case_file(cf, {"TX-9", "N-1"}, tc_involved=False)
    assert [c["text"] for c in out["claims"]] == ["Wire to a new payee"]
    assert rep["claims_total"] == 4 and rep["faithfulness"] == 0.25
