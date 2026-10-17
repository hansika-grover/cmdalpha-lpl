# Sentinel

**Early detection of elder financial exploitation for LPL Financial, with a person making every decision.**
Team cmd+alpha · LPL WealthX Hackathon, Infinium '26, IIIT Hyderabad

Americans aged 60+ reported **$7.75 billion** in fraud losses to the FBI in 2025, up 59% in a year
([IC3 2025](https://www.ic3.gov/annualreport/reports/2025_ic3report.pdf)). Firms can already act (FINRA Rule 2165 holds,
Rule 4512 trusted contacts, Senior Safe Act immunity), but deciding *when* depends on someone noticing one odd event.
Withdrawal thresholds miss small, repeated thefts and stop legitimate large withdrawals, and what clients tell their advisors
is never read systematically.

Sentinel combines money movement, account changes and advisor notes, re-scores a household every time something happens,
and gives the advisor or the Senior Investor Protection team a cited case file before the money leaves.

**Live demo:** open `web/index.html` through a local server (see Run it), or the deployed dashboard.

## Results

All data is synthetic: a simulated book of 20,000 households calibrated to FinCEN and FBI figures. The numbers show how the
method behaves, not real-world accuracy.

<!-- RESULTS:START -->
Test split: 4,000 households never seen in training. Notes engine in the score: fast keyword + TF-IDF reader; Claude reads notes on case pages and drafts the case files (see What the LLM adds). 95% confidence intervals from a household bootstrap.

**Same reviewer workload as the $100k rule** (0.17% of households per week):

| Method | Cases caught | 95% CI | Caught before money left | Dollars protected | Look-alike false alarms | Precision at 0.5% base rate |
|---|---|---|---|---|---|---|
| $100k threshold rule | 3.8% | 0–8% | 2.5% | 15.0% | 35.9% | 0.1% |
| Expert rule checklist (no notes) | 35.0% | 25–46% | 33.8% | 42.4% | 82.3% | 1.1% |
| Money movement only (engine A) | 72.5% | 62–83% | 41.2% | 57.6% | 29.8% | 3.6% |
| **Sentinel** | 81.2% | 72–90% | 66.2% | 76.0% | 22.7% | 5.4% |

**At fixed weekly review budgets** (cases caught, with look-alike false alarms in brackets):

| Method | Fixed rule | 0.1% budget | 0.25% budget | 1% budget |
|---|---|---|---|---|
| $100k threshold rule (fixed, 1.53 alerts per 1k households a week) | 4% (36%) | | | |
| $25k threshold rule (fixed, 9.13 alerts per 1k households a week) | 31% (84%) | | | |
| Expert rule checklist (no notes) | | 24% (78%) | 36% (88%) | 74% (96%) |
| Money movement only (engine A) | | 60% (23%) | 78% (38%) | 91% (69%) |
| **Sentinel** | | 74% (18%) | 86% (26%) | 86% (43%) |
| Sentinel without notes | | 76% (31%) | 85% (44%) | 91% (58%) |
| Sentinel without sequences | | 60% (6%) | 74% (19%) | 80% (34%) |
| Sentinel without money movement | | 71% (18%) | 76% (24%) | 79% (36%) |
| Sentinel without the mule-account network | | 76% (17%) | 86% (27%) | 86% (43%) |
| Sentinel without household memory | | 75% (16%) | 85% (27%) | 91% (39%) |

**By scam type** at a 0.25% budget, and with that type held out of training entirely:

| Typology | Test cases | Sentinel | Same features, all types seen | Same features, type never seen | $100k rule |
|---|---|---|---|---|---|
| Investment, crypto or romance | 21 | 81% | 81% | 67% | 5% |
| Tech support / bank impersonation | 16 | 94% | 81% | 62% | 0% |
| Government impersonation | 7 | 43% | 43% | 43% | 14% |
| Silent account takeover | 13 | 100% | 100% | 77% | 8% |
| Grandparent / voice clone | 5 | 80% | 80% | 80% | 0% |
| Trusted-person theft | 18 | 94% | 94% | 94% | 0% |

**Calibration:** expected calibration error 0.00061. **Fairness** (false-alarm rate relative to overall, target ≤ 1.25×): ages 55-64 0.90×, ages 65-74 0.96×, ages 75-84 1.07×, ages 85+ 1.19×. **Speed:** 0.809 ms per household for all three engines on a laptop.

**Robustness** (same model and thresholds, cases caught at 0.25%): as evaluated 86%, all notes removed 82%, half the notes removed 80%, doubled account-event noise 84%; coached cover stories 92% (n=12).

**Stability across 5 independently generated worlds:** Sentinel catches 87% of cases at 0.25% on average (range 84–90%).

By scam type, mean ± sd of cases caught across the worlds:

| Typology | Sentinel at 0.25% | Sentinel at 1% | $100k rule |
|---|---|---|---|
| Investment, crypto or romance | 89% ± 4 | 96% ± 3 | 3% ± 2 |
| Tech support / bank impersonation | 89% ± 7 | 87% ± 8 | 2% ± 3 |
| Government impersonation | 47% ± 14 | 61% ± 9 | 11% ± 6 |
| Silent account takeover | 95% ± 6 | 94% ± 6 | 5% ± 4 |
| Grandparent / voice clone | 66% ± 17 | 88% ± 24 | 8% ± 10 |
| Trusted-person theft | 99% ± 2 | 99% ± 2 | 5% ± 6 |

**What the LLM adds** (notes reader on the same 2,226 notes, scored against the planted red-flag categories; Claude Haiku with the production prompt, outputs pre-generated and cached):

| Reader | Precision | Recall | F1 | Macro F1 |
|---|---|---|---|---|
| Keyword list | 42% | 67% | 52% | 65% |
| Claude | 91% | 94% | 93% | 92% |

Claude minus keyword, F1: +41 points (95% CI 38–44%). Quotes that were not exact substrings of the note: 0 of 2,115. 
**Prompt injection** (439 injected notes, each also read with the injection removed): Claude reported the injection in 439 and red flags changed in 1 (a plain second read of the same notes changes them in 1). Mitigating context is less stable: it changed in 34% of injected notes, against 30% on a plain re-read, but injected notes gained mitigating quotes 113 times and lost them 52 times, so code now ignores mitigating context in any note with an injection (risk-lowering context added by an injection after that rule: 0).

**Case agent:** case files for 35 queued households; 178 of 178 claims kept (every kept claim cites a record ID that exists for that household; faithfulness 100%); trusted-contact drafts withheld where the contact may be involved: 0.
<!-- RESULTS:END -->

## How it works

```
money movement   ─┐   A  money movement against the household's own 12-month baseline (size, new payees, velocity, structuring)
                  │      + firm-wide network: was this receiving account also paid by other clients?
account changes  ─┼─▶ B  account-change sequences (contact change → new bank → withdrawal; new POA → payment to that person)
advisor notes    ─┘   C  notes reader: FinCEN red-flag categories, each with an exact quote; injected instructions ignored
                      + household memory: risk building across several weeks
                           │
                           ▼
          gradient-boosted model, calibrated → score + plain-English reasons, each citing record IDs
                           │
                           ▼
     weekly review budget → advisor queue / pre-release check / specialist queue → case agent (Claude) drafts the call script,
     trusted-contact note, escalation memo, APS referral and Rule 2165/2166 deadlines → a person decides
```

* **Scored before money moves.** Every withdrawal request is scored before release, as is every account change and note.
  Features use only records from before that moment (tested by deleting the future).
* **Every claim is cited.** A quote must appear word for word in the note; a case-file claim must cite a record ID that exists
  for that household. Code enforces both.
* **Guardrails in code.** Age is never a model input, and false alarms are reported by age band. No diagnoses, no outreach to a
  trusted contact who may be involved, nothing sent or held automatically.
* **Hard rule.** Money going repeatedly to the trusted contact or a new power of attorney always escalates.
* **Claude.** Claude Haiku reads advisor notes and Claude Opus drafts case files, through the Anthropic API. The outputs used
  in this repo are pre-generated and cached in `sentinel/llm/cache/`, so everything runs offline.

## Run it

Python 3.11+.

```bash
pip install -r requirements.txt
python -m http.server -d web 8000   # dashboard at http://localhost:8000
make all                            # rebuild everything: world → eval → seeds → export → readme
```

On Windows without GNU make, use `mingw32-make all` or run the steps one by one:

| Step | Command |
|---|---|
| Generate the synthetic world | `python -m sentinel.generate` |
| Train, calibrate and evaluate | `python -m sentinel.eval.run_eval` |
| Repeat on four more worlds | `python -m sentinel.eval.seeds` |
| Build the dashboard data | `python -m sentinel.export` |
| Run the tests | `python -m pytest` |

To regenerate the Claude outputs, set `ANTHROPIC_API_KEY` in `.env` and run `make notes agent`. Models are set with
`SENTINEL_NOTES_MODEL` (default `claude-haiku-5-5`) and `SENTINEL_AGENT_MODEL` (default `claude-opus-5-5`).

## The synthetic world

Every rate and the seed live in `config/world.yaml`; the same seed reproduces identical files.

* 20,000 households aged 55–95 over 24 months, served by 250 advisors who write notes in five styles.
* Normal life: monthly withdrawals, required distributions, occasional large withdrawals, new payees, account changes and
  2–8 notes a year (15% of households have none).
* **2% of households exploited** across six scam and theft types, with FinCEN loss medians and the 80/20 stranger/trusted-person
  split. Each signal appears only some of the time, and 15% of scam victims give a coached cover story.
* **About 1,000 legitimate look-alikes:** home purchases, gifts, care-home deposits, moves, remarriage and trust funding.
* 1% of notes contain prompt-injection text. 35% of stranger scams pay a mule account shared with other victims.
* Labels are stored separately, and the feature code cannot read them (tested).

## Repository

```
config/            world and model settings
sentinel/generate/ synthetic households, scams, look-alikes and notes
sentinel/features/ money movement, network, account sequences, notes and household memory
sentinel/llm/      notes reader, case agent and cached Claude outputs
sentinel/models/   model, calibration and reason codes
sentinel/eval/     baselines, metrics, confidence intervals, robustness and stability runs
sentinel/export/   dashboard data and the README results
web/               the dashboard (single HTML file) and its data
results/           evaluation outputs
tests/             determinism, leakage, guardrails, deadlines and injection tests
docs/CONTEXT.md    how the pieces fit together
```

## Limitations

* The data is synthetic and the notes are template-generated; real notes are far more varied. The next step is a backtest on
  LPL's past Senior Investor Protection cases, then a silent run alongside today's rules.
* The risk score reads notes with a fast keyword reader. Claude's reading is shown on case pages, drafts the case files and is
  measured separately above; moving the score onto Claude is one pipeline step (`make notes`, then re-evaluate).
* The weekly review budget is LPL's choice. Precision is low at generous budgets; it is also reported at a realistic 0.5% base rate.
* Personal details are not redacted before notes reach the model; in production this would run inside LPL's environment.
* The mule-account share, pool size and lifetime are our assumptions; that mules are reused across victims is documented by
  FinCEN and the FBI.
* An earlier version of the generator time-stamped some notes after the event they described. It was fixed and all results
  were re-run; the fix moved recall within its confidence interval and lowered false alarms on look-alikes.
