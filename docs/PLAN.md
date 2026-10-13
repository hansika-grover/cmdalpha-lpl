# Plan: who does what

**Dates:** submission **16 Oct** (moved from 13 Oct), final presentation **17 Oct**.
**Team:** Parth (models, data, Claude), Shaurya (UI and demo), Navya (pitch, evidence, submission).

**Change log:** 13 Oct: generator note-timing bug fixed (found by Parth); Shaurya's UI merged; `events_after` added;
deadline moved to 16 Oct; plan rewritten to beat a team with a very similar problem statement.

`docs/CONTEXT.md` explains the code. This file says what each person does and in what order.

## Ground rules (these are how we win, not just how we stay honest)

1. **No number is typed by hand.** It changes only because a script re-ran.
2. **Never tune on the test split.** Choose models and settings on validation; report test once.
3. **Freeze v1 before changing the world.** Before any generator change, copy `results/metrics.json` to
   `results/v1_metrics.json`. Any new signal we add to the world needs a source (FinCEN, FBI, FINRA) and a line in the
   README limitations, and the baselines must be re-run on the same world. A judge who sees us make the data easier to
   suit our model will stop trusting every number.
4. **Every claim cites a record**, slides included.
5. **Numbers freeze on 15 Oct at 20:00.** After that: bug fixes, rehearsals and the video only.

## What judges will compare

Does it **work live**? Is the evidence **rigorous**? Does it **fit LPL** (pre-release check, Rule 2165/2166, trusted
contacts)? Is the AI **doing something a rule can't**? Every task below serves one of those four.

---

## Parth: models, data and Claude

| Pri | Task | Why it wins | Where | By |
|---|---|---|---|---|
| 1 | **Run Claude.** `ANTHROPIC_API_KEY` in `.env`; trial `python -m sentinel.llm.notes_reader --sync --limit 200`; then `make notes eval export agent readme`. Commit `sentinel/llm/cache/*`, `results/*`, `web/data/*`. | Without it, the "Sentinel" row in every table is the keyword reader. | `llm/notes_reader.py`, `llm/case_agent.py` | 14 Oct |
| 2 | **"What the LLM adds" metrics**: precision and recall of keyword vs Claude reader against `labels/note_truth.csv` (overall and by advisor style); injection controls (target: zero changed outputs); quotes dropped. | Proves the LLM is doing something a keyword list can't, with numbers. | new `eval/readers.py`; Evidence page | 14 Oct |
| 3 | **Mule-account network.** Scam rings reuse receiving accounts and wallets across victims. Payee features: how many *other* households paid this payee in the last 30/90 days, how many were first-time payers, how young the payee is. In the generator, a sourced share of scam cases pays accounts from a small pool of mule accounts (rate in `config/world.yaml`). Follow ground rule 3. | Only a firm-wide view like LPL's can see this; no single-client app or per-household rule can. Demo line: "this wallet received money from 3 other LPL clients this month". | `features/money.py` (same `Stream` trick, keyed by payee); `generate/world.py` | 15 Oct |
| 4 | **Train for the task we're scored on.** Try LightGBM's ranking objective (`lambdarank`, grouped by week) and a small search (depth 3–4, trees, learning rate, class weight), chosen on **validation** recall at the 0.25% budget. | We alert the top of a weekly ranking, so a ranking objective fits better than classifying each moment. | `models/fusion.py`; new `eval/tune.py` | 15 Oct |
| 5 | **Household memory features**: engines fired in the last 30 days, peak score in the last 14 days. Re-run the future-deletion test. | Catches slow-burn theft where each step looks small. | new `features/memory.py` | 15 Oct |
| 6 | **Export the model for the browser**: dump the LightGBM trees, the isotonic calibration table and the feature list to `web/data/model.json` for Shaurya's scenario builder. | Lets judges drive the real model live. | `export/build.py` (`booster_.dump_model()`) | 14 Oct |
| 7 | Per-typology averages across the 5 worlds; re-check government and grandparent scams after Claude runs. | Shows we know our weak spots. | `eval/seeds.py` | 15 Oct |

On "can we train a deep model": we already train one (gradient-boosted trees, calibrated). With ~400 planted cases a
neural sequence model is unlikely to beat it and is harder to explain. Items 4 and 5 get the same benefit (order and
memory) and keep cited reasons. Try a small GRU only as an ablation if there's spare time, and report it honestly if it
loses.

## Shaurya: UI and live demo

| Pri | Task | Why it wins | By |
|---|---|---|---|
| 1 | Fix the "Review line" label overlapping the score line at 375px; re-check every page against new data after each Parth re-run (desktop, 375px, both themes). | Polish. | each re-run |
| 2 | **"Try it yourself" scenario builder.** Pick a household, add events (change email, link a bank, type a note, request $40k to a new payee) and watch the score and reasons update live, using Parth's `model.json` with a ~50-line JS tree evaluator. Notes use the keyword reader in JS, or Claude via item 3. Show the review line and the $100k rule's verdict side by side. | Live interaction beats any slide, and proves the model is real. | 15 Oct |
| 3 | **Live Claude reading through a Netlify Function** (key in Netlify env vars, never in the page); "Read live with Claude" on the notes tab, with the exact-quote check in JS. | Shows the LLM and its guardrails working in front of judges. | 15 Oct |
| 4 | **Mule network view** once Parth's item 3 lands: households → shared payee. | Visual proof of the firm-wide advantage. | 15 Oct |
| 5 | **Evidence packet print view**: a print stylesheet so a case prints as a clean, cited PDF. | Speaks to compliance (FINRA recordkeeping). | 16 Oct |
| 6 | **Reviewer feedback**: "Confirmed" / "False alarm" on Decide, written to the activity log ("used to retrain weekly"). | Shows it learns from LPL's own decisions. | 16 Oct |

## Navya: pitch, evidence and submission

| Pri | Task | By |
|---|---|---|
| 1 | **Deploy**: connect Netlify to the repo (`netlify.toml` publishes `web/`); confirm the public URL on desktop and phone. | 14 Oct |
| 2 | **Competitor scan**: find what the other team shows (demo, deck, repo if public). Build one comparison slide against the *problem*, not against them by name: firm-wide view, notes read with cited quotes, pre-release timing, Rule 2165/2166 workflow, trusted-contact safety, evidence with confidence intervals and an unseen scam type. | 14 Oct |
| 3 | **Submission package**: confirm portal, cut-off time and format; repo link; solution description and AI approach (from `README.md` and `docs/CONTEXT.md`). | draft 15 Oct, submit 16 Oct |
| 4 | **Deck with real numbers** from `results/metrics.json` after the 15 Oct freeze: same-workload chart, unseen scam type, ablations, LLM vs keyword, fairness, calibration, and the timing-bug fix as a credibility point. Sizing slide uses Δ from the Evidence page. | 16 Oct |
| 5 | **Q&A sheet** (playbook §14) mapped to Evidence-page sections and code, one owner per question. Include "isn't the synthetic data rigged?" (answer: frozen v1, sources for every rate, baselines on the same world, held-out typologies, 5 worlds). | 16 Oct |
| 6 | **Human check**: each teammate rates 5 Claude case files (20 total) 1–5 for usefulness and accuracy; quote the average. | 15 Oct |
| 7 | **2-minute video** after the freeze (queue → Margaret with "What happened next" → Harold → Ruth → scenario builder → Evidence), plus three timed rehearsals with the live demo. | 16 Oct |

## Timeline

| Day | Parth | Shaurya | Navya |
|---|---|---|---|
| **14 Oct** | Claude run, LLM metrics, `model.json` | label fix, start scenario builder | deploy, competitor scan |
| **15 Oct** | mule network, ranking objective, memory features; full re-run; **freeze 20:00** | scenario builder, live Claude, network view | submission draft, human check |
| **16 Oct** | bug fixes only | print view, feedback buttons, final pass | deck, Q&A sheet, video, **submit** |
| **17 Oct** | present | drive the demo | narrate |
