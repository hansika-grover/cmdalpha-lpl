# Plan: what's left before submission (13 Oct) and finals (17 Oct)

**P0** = must be in the 13 Oct submission. **P1** = before the 17 Oct presentation. Each item says where in the code to
start; `docs/CONTEXT.md` explains the pieces. Rule for everyone: no number is typed by hand. If a number changes, it
changes because a script re-ran.

Current state: the full pipeline runs, the evaluation is in `results/metrics.json`, and the dashboard reads live data.
The Claude notes reader and case agent are built but **have not been run** (no API key yet), so "Sentinel" currently uses
the keyword notes reader and template drafts. Getting Claude running is the single most important item.

---

## Shaurya: UI design (`web/index.html`)

| Pri | Task | Notes |
|---|---|---|
| P0 | **Margaret's replay is too short.** At her alert date only a baseline and one note exist, so the chart is one jump. Add a "What happened next" toggle that reveals the later events (the wires Sentinel would have held) greyed out, with "Sentinel alerted here" marked on the chart. | Needs the later events in the JSON: ask B to add `events_after` in `sentinel/export/build.py:client()` (the next 45 days of notable events). |
| P0 | **Mobile pass at 375px**: Evidence charts (labels clip), Specialists table (scrolls but is cramped), case header score size. | The site uses a 16px gutter rule; charts are SVG `viewBox`, so pass a narrower width on small screens. |
| P0 | **Evidence page hierarchy.** Lead with the one chart judges should remember (same workload: Sentinel vs $100k rule vs checklist), make the 4 stat tiles match it, and collapse the long budget table by default. | Data is in `metrics.json → matched_workload_100k`. |
| P0 | **Show the reader honestly but nicely.** The "keyword baseline (Claude not run yet)" label appears in several places; after B runs Claude it flips automatically. Design both states. | `META.reader`, `c.reader`, `c.case_file.generated_by`. |
| P1 | **Case file on the case page.** Show the agent's claims, each record ID clickable, scrolling to and highlighting the timeline row or note. | `c.case_file.claims[].record_ids`; timeline rows carry `ref`. |
| P1 | **Queue filters**: chips for pattern (scam / takeover / someone close) and tier; a hard-rule badge; alert date on each row. | `c.pattern`, `c.tier`, `c.hard_rule`, `c.alert_date`. |
| P1 | **Specialists page**: deadline countdown chips, trusted-contact warning, and a link from each case to its evidence packet. | `specialist.json`. |
| P1 | Light/dark check of every new page; loading and "no data" states; favicon/title. | |

## Teammate B: data, models and Claude (`sentinel/`)

| Pri | Task | Notes |
|---|---|---|
| P0 | **Run Claude.** Put `ANTHROPIC_API_KEY` in `.env`, then `make notes eval export agent readme`. Commit `sentinel/llm/cache/*`, `results/*`, `web/data/*`. | Reading ≈95k notes with Haiku through the Batches API should cost under $1 at list prices; the batch can take up to an hour. Try `python -m sentinel.llm.notes_reader --sync --limit 200` first to check the output. |
| P0 | **Reader quality metric: "what the LLM adds".** Precision and recall of the keyword vs Claude reader against `data/world/labels/note_truth.csv` (planted categories), overall and by advisor style. Add to `metrics.json` and the Evidence page. | New function in `sentinel/eval/`; the truth file is only for evaluation. |
| P0 | **Injection resistance and quote validity for Claude.** The reader already reads each injected note twice (with and without the injection); compare the flags and count changes (target 0). Count `n_dropped` quotes from the cache. | `notes_reader.py:main` creates the controls; add the comparison to `run_eval.py`. |
| P0 | Add `events_after` to the export for Shaurya's replay toggle. | `export/build.py:client()`. |
| P1 | **Weak spot: government impersonation** is caught only ≈29% at a 0.25% budget (n=7). Check whether the single fast withdrawal looks too normal; fixes go in features, never by tuning on test. | `features/money.py`; compare with LOTO results. |
| P1 | **More varied notes.** Rewrite a sample of notes through Claude in each advisor's style (playbook §9), regenerate, and re-run eval to show the result holds on messier text. | `generate/text.py`, a new rewrite step with its own cache. |
| P1 | Case-agent faithfulness across all queued households (`results/case_agent_usage.json`), and the cost and seconds per household with Claude, on the Evidence page. | |

## Teammate C: pitch, demo and submission

| Pri | Task | Notes |
|---|---|---|
| P0 | **Submission package**: confirm the portal, deadline time and format; repo link; README check; short solution description and "AI approach" text (take both from `README.md` and `docs/CONTEXT.md`). | Open questions in playbook §1. |
| P0 | **2-minute demo video** following playbook §13: queue → Margaret (replay, verified quotes) → Harold (trusted contact withheld) → Ruth (threshold false alarm) → Decide (deadlines) → Evidence (one chart). Record against the deployed site. | Use the numbers on the Evidence page, never older slides. |
| P0 | **Deploy**: connect the Netlify site to this repo (publish dir `web/`, already set in `netlify.toml`). Check that all pages load on the public URL. | |
| P1 | **Deck update** with the real numbers: headline (same workload), by scam type, unseen scam type, ablations, fairness, calibration. Sizing slide: Δ = Sentinel's dollars-protected share minus the $100k rule's at the same workload (Evidence page computes it). | Every number cites `results/metrics.json`. |
| P1 | **Q&A sheet** (playbook §14): map each likely question to an Evidence-page section or code file; assign an owner per question. | |
| P1 | **Human check**: each teammate rates 5 case files (20 total) for usefulness and accuracy, 1–5. Quote the average honestly. | After B runs the case agent. |
| P1 | Three timed rehearsals; one person drives, one narrates. Keep the recorded video as a backup. | |

---

**Order for tonight:** B starts the Claude batch first (it runs in the background), C deploys and drafts the submission
text, Shaurya does the replay toggle and the mobile pass. Once the batch lands: B re-runs `make eval export agent readme`,
everyone re-checks their pages against the new numbers, then C records the video.
