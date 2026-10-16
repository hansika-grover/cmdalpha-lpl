# Sentinel: presentation brief

Give this whole file to Claude (or any teammate) to build the final deck. It is self-contained: the problem, what we built,
the AI, every result with its source, the demo story, and the lines we must not cross. All numbers come from
`results/metrics.json`, `results/seeds.json` and `results/readers.json` in the repo
(github.com/hansika-grover/cmdalpha-lpl); if a number changes there, it changes here.

---

## 0. What to make

* **Event:** LPL WealthX Hackathon, Infinium '26, IIIT Hyderabad. Team **cmd+alpha**: Parth, Shaurya, Navya and [add the fourth name].
* **Format:** final presentation, about 6 minutes plus Q&A, with a live demo of the dashboard (2 minutes inside the talk).
  Judges score problem, solution, the AI behind it, the live demo, impact and future potential.
* **Deck:** 10–12 slides, one idea per slide, a headline sentence on every slide that states the takeaway, numbers large.
* **Look (match the dashboard):** Public Sans for text, IBM Plex Mono for record IDs; ink `#1C2128`, petrol `#0F5866`
  for Sentinel's own elements (its chart line, logo ring, "read by Claude" badges); red `#E5484D`, amber `#F5A524`,
  green `#30A46C` mean risk state only. White panels with a 14px radius on a faint grey-green page. Today's methods in grey,
  Sentinel in petrol.
* **Logo:** an outline shield with a petrol ring at its centre, "SENTINEL" in spaced capitals.

## 1. The one-line claim

**On a realistic simulated book of business, Sentinel catches far more elder financial exploitation, earlier, with fewer
false alarms than today's threshold rules, at the same reviewer workload, inside LPL's workflow, with a person making every
decision.**

## 2. The problem (sources in brackets)

* Older Americans reported **$7.75 billion** in fraud losses to the FBI in 2025, up **59%** in a year and more than 4× 2021.
  People 60+ filed 20% of complaints but suffered 37% of losses. (FBI IC3 2025; ELACA analysis)
* **One complaint every 2.6 minutes** from someone 60+ (201,266 complaints in 2025). About **34 people a day lose more
  than $100,000**; average loss **$38,500**. (IC3 2025; our arithmetic)
* What is reported is the tip: the FTC estimates true 2024 losses of older adults at **$10.1B–$81.5B**. Older adults are
  13% less likely to report fraud. (FTC 2024-25; Senate letter citing FTC)
* Scams are getting smarter: 3,100+ AI-related complaints from seniors in 2025 (>$352M); $4.35B lost to crypto-linked fraud.
* About **20%** of cases are theft by someone the victim knows, most often an adult child. (FinCEN)
* Boomers hold **51.6%** of US household net worth (~$89.8T). 7.4M Americans 65+ have Alzheimer's. (Fed DFA; Alzheimer's Assn)
* The human line: *a scam is a crime against a person's future, and for an older client there is no time left to rebuild it.*

**Where today's process breaks:** firms have the legal tools (FINRA Rule 2165 holds, Rule 4512 trusted contacts, Senior Safe
Act immunity, proposed Rule 2166 "speed bump" filed Aug 2026), but deciding *when* depends on one person noticing one odd
event. Each signal sits in a different system; fixed thresholds miss small repeated withdrawals and flag legitimate large
ones; the richest evidence (what the client told the advisor) is never read systematically; deadlines are tracked by hand.
In March 2026 LPL itself asked FINRA for clearer "reasonable belief" factors and good-faith protection.

## 3. What Sentinel is

Sentinel joins three streams LPL already holds, scores every household **each time something happens**, and hands the
advisor or the Senior Investor Protection (SIP) team a cited case file **before the money leaves**.

| Engine | Reads | Example signal |
|---|---|---|
| A. Money movement | withdrawals, wires, payees, against the household's own 12-month baseline | "a withdrawal many times her largest in a year, to a payee first used this month, funded by selling a 15-year holding" |
| Firm-wide mule network | which *other* LPL households paid the same receiving account or wallet | "this receiving account was also paid by other LPL households in the last 90 days" (in the demo data, TD Escrow Services was paid by two households, one of them in the queue) |
| B. Account-change sequences | email/phone changes, new devices, bank links, power of attorney, trusted contacts | "email changed, then a new bank linked, then a withdrawal, all in 4 days" |
| C. Notes reader (Claude) | advisor notes and call summaries | quotes like "She told me she was instructed not to tell anyone at the bank" |
| Household memory | risk building across several weeks | "3 warning signs across 2 engines in 30 days" |

* **Fusion:** a shallow, monotone gradient-boosted model (LightGBM, depth 3), **calibrated** so a score of 0.7 means about
  70% of similar moments in the simulation were real cases. Every score comes with plain-English reasons, each citing the
  record IDs behind it.
* **Weekly review budget:** LPL sets how many households reviewers can look at (e.g. 0.25% a week); Sentinel fills it with
  the highest-risk households. **Hard rule:** money moving repeatedly to the trusted contact or a new power of attorney
  always escalates.
* **Case agent (Claude):** for each flagged household it uses read-only tools (household profile, timeline, notes, reasons,
  deadline calculator) and drafts the evidence packet: summary, cited claims, recommended action, a call script, a
  trusted-contact note, an escalation memo and an APS referral, plus **Rule 2165 / 2166 deadlines** in business days.
* **A person decides:** check in (lightest), speed bump (proposed Rule 2166, up to 10 business days), or escalate for a
  Rule 2165 hold (notice within 2 business days, hold up to 15 + 10). Nothing is sent or held automatically. Every signal,
  draft and decision lands in an audit log.

**How it fits LPL** (self-clearing, ClientWorks, Cyan, SIP team, Anthropic relationship expanded Feb 2026):
1. A **pre-release check** on money movement: score the request before funds are released.
2. A **Cyan skill** in ClientWorks: "who needs a call today", with reasons and drafts.
3. A **case queue for SIP**: evidence packets, deadlines, every decision archived (books and records).

## 4. The AI, and how Claude is used

* **Claude via the Anthropic API** does two jobs: **Claude Haiku** reads advisor notes (Batches API, cached by note hash,
  cost-controlled: only notes near a risky event or flagged by a cheap first pass); **Claude Opus** is the case agent
  (tool-use loop with read-only tools, structured JSON output).
* **For the live demo, Claude's outputs are pre-generated and cached**, so the dashboard runs offline at the venue. On a
  case's notes tab, "Read with Claude" plays back Claude's reading through the same exact-quote check.
* **Guardrails are enforced in code, not trusted to the model:**
  * A quote counts only if it is an **exact substring** of the note: **0 of 2,115** Claude quotes failed this check.
  * Notes are data, never instructions: of **439** notes with planted prompt-injection text, Claude reported the injection
    in **all 439** and its red flags changed in **1** (the same as a plain re-read). Mitigating context from an injected note
    never counts.
  * Every case-file claim must cite a record ID that exists for that household: **178 of 178** claims across 35 case files.
  * No age as a reason, no diagnoses, no draft to a trusted contact who may be involved, nothing sent automatically.
  * Age is never a model input; false alarms are reported by age band.
* **What the LLM adds** (2,226 notes scored against the planted red-flag categories):

| Reader | Precision | Recall | F1 |
|---|---|---|---|
| Keyword list (what a team builds without an LLM) | 42% | 67% | 52% |
| **Claude** | **91%** | **94%** | **93%** |

  Claude is **+41 F1 points** better (95% CI 38–44).
* **Honest detail for Q&A:** the headline risk-score numbers below were produced with the fast keyword + TF-IDF reader
  inside the score; Claude's reading is shown on case pages, drafts the case files and is measured separately (table
  above). Switching the score's notes engine to Claude is one pipeline step. Say "the score's notes engine is the fast
  reader today; Claude reads 41 F1 points better, so the numbers are a floor".

## 5. How we tested it (why the numbers deserve trust)

* **Synthetic world, calibrated to real figures:** 20,000 households aged 55–95 over 24 months, 250 advisors writing in five
  styles, ~526,000 money movements, ~209,000 notes. **2% planted exploitation** (≈400 cases) in six typologies with FinCEN
  loss medians ($33.5k scams, $23.8k theft) and the FinCEN 80/20 stranger/trusted split; **~1,000 legitimate look-alikes**
  (home purchases, gifts, care-home deposits, moving states, remarriage, trust funding); 15% of contact scams come with a
  coached cover story; 1% of notes carry prompt-injection text. Every rate lives in one config file; same seed, identical
  files.
* **Fair test:** split by household 60/20/20; thresholds and model choice set on validation only; test reported once.
  Scored forward in time (a test deletes the future and checks no score changes). 95% CIs from 1,000 bootstrap resamples.
* **Hard checks:** a scam type held out of training entirely; ablations; robustness (notes removed, noise added); fairness by
  age band; calibration; 5 independently generated worlds.
* **Integrity moments worth telling:** Parth found a timing bug in our own generator (notes about a request were sometimes
  time-stamped after it); we fixed it and published the before/after. Before adding the mule network we froze v1 results,
  re-ran every baseline on the same world, and chose the final model on validation only.

## 6. Results (test households the model never saw)

### Headline: same reviewer workload as the $100k rule (0.17% of households a week)

| Method | Cases caught | 95% CI | Caught before money left | Stolen dollars protected | Legit look-alikes flagged |
|---|---|---|---|---|---|
| $100k threshold rule | **3.7%** | 0–8% | 2.5% | 15.0% | 35.9% |
| Expert rule checklist (no notes) | 35.0% | 25–46% | 33.8% | 42.4% | 82.3% |
| Money movement only | 72.5% | 62–83% | 41.2% | 57.6% | 29.8% |
| **Sentinel** | **81.2%** | **72–90%** | **66.2%** | **76.0%** | **22.7%** |

Precision at a realistic 0.5% base rate: Sentinel 5.4% vs the $100k rule 0.1% (about **50× more true alerts per alert**).

### At a 0.25% weekly review budget
Sentinel catches **86.3%** of cases, **75.0%** before any stolen money leaves, protects **87.7%** of stolen dollars, flags
26.3% of legitimate look-alikes, and alerts about 2.4 households per 1,000 a week. Median warning: 1.5 days before the first
withdrawal.

### Stable across 5 independently generated worlds (0.25% budget)
Sentinel **87%** of cases (range 84–90%), 74% before money left, 82% of stolen dollars protected. The $100k rule: **4.5%**
(range 1–6%). Expert checklist: 45% while flagging 85% of look-alikes. Money movement only: 74%.

### By scam type (mean across 5 worlds, 0.25% budget)

| Scam type | Sentinel | $100k rule |
|---|---|---|
| Investment, crypto or romance | 89% | 3% |
| Tech support / bank impersonation | 89% | 2% |
| Government impersonation | 47% | 11% |
| Silent account takeover | 95% | 5% |
| Grandparent / AI voice clone | 66% | 8% |
| Trusted-person theft (family, caregiver) | 99% | 5% |

### A scam type it never saw (that type removed from training)
Investment/romance 67%, tech support 62%, government 43%, account takeover 77%, grandparent 80%, trusted-person theft 94%.
The $100k rule: 0–14%.

### What each engine adds (0.25% budget; cases caught / legit look-alikes flagged)
Sentinel 86.3% / 26.3% · without notes 85.0% / **44.4%** · without account sequences 73.8% / 18.7% · without money
movement 76.2% / 24.2% · money movement only 77.5% / 38.4%. **Notes are what clears innocent clients:** without them false
alarms on look-alikes rise from 26% to 44%. Sequences add the account-takeover and family-theft catches.

### Trust checks
* **Calibrated:** expected calibration error 0.0006.
* **Fair:** false-alarm rate by age band vs overall: 55–64 0.90×, 65–74 0.96×, 75–84 1.07×, 85+ 1.19× (target ≤ 1.25×).
* **Robust:** all notes removed 82.5%, half removed 80.0%, doubled account-event noise 83.8%; coached cover stories 92%.
* **Fast:** 0.8 ms per household for all engines on a laptop.
* **v1 → final:** same-workload catch rate 76.2% → 81.2%; caught before money left at 0.25% 67.5% → 75.0%.

## 7. Business case (assumptions labelled)

Value protected per year = LPL clients 60+ × annual exploitation rate × average loss × Δ, where Δ is the extra share of
stolen dollars Sentinel stops at the same workload. From our evaluation, **Δ = 76.0% − 15.0% = 61 points vs the $100k rule**
(34 points vs the expert checklist).

* Assumptions: 3.2M clients 60+ (40% of the ~8M Americans LPL's advisors serve; to confirm with LPL); $38,500 average loss
  (IC3 2025); exploitation rate 0.25% / 0.5% / 1% of households a year (well below survey estimates of 41 per 1,000).
* vs the $100k rule: **$188M / $376M / $751M** a year. vs a strong checklist: $103M / $207M / $414M.
* Plus: investigator hours saved (cited evidence packets), readiness for Rule 2166, a defensible record for every hold,
  retained assets and trust. UK context: where firms carry reimbursement liability, scam losses fell ~21% (PSR, Bratby).

## 8. The demo (2 minutes, in this order)

The dashboard opens on **Saturday, August 2, 2025**: 35 households in the review queue out of 4,000 test households.

1. **Queue:** "Out of 4,000 households, these need review this week; every score explains itself." Point at the reader and
   agent badges and the filter chips.
2. **Margaret Ellis, 80, $3.78M in assets, score 0.71.** Flagged *from the advisor's notes alone, before any unusual
   withdrawal*: "The money hasn't left yet." Notes tab → **Read with Claude**: three red flags, each a verified quote:
   "Margaret wanted to know the quickest way to move the money", "Margaret has a new 'boyfriend', a gentleman named Andrew,
   who she has never met in person", "She told me she was instructed not to tell anyone at the bank or here." Line:
   *"If it can't quote it, it can't claim it."* Toggle **What happened next**: three wires to Apex Trade Ltd follow
   ($2,390, $6,070, $10,920; the last funded by selling a 7-year holding). The $100k rule never fires.
3. **Harold Brennan, 77, escalated by the hard rule.** His daughter Kelly was added as joint owner, the trusted contact was
   removed, and repeated transfers go to her ($11,550 in 90 days, one funded by selling a 15-year holding). The hard rule
   sends it straight to specialists whatever the score; with no trusted contact left on file, the draft suggests asking him
   to name one (FINRA Rule 4512).
4. **Ruth Castellano, 79, $700,000 wire for a condo** to First American Title. The $100k rule would interrupt her;
   Sentinel's score stays at 0.001 because the context checks out. "No call needed."
5. **Decide** on Margaret: pick "Escalate for a hold" and show the computed Rule 2165 deadlines (business days, holidays
   skipped), then the **Activity log**.
6. **Evidence page:** one chart, one sentence: "Same workload: 81% vs 4%."

**Safety:** the dashboard works offline (Claude outputs cached); keep a recorded backup video; pre-open the tabs.

## 9. Why us (vs a team with a similar idea)

1. **Firm-wide view no single-client app has:** the mule network ("this account was also paid by other LPL households") and every
   client covered by default, no enrolment.
2. **Reads the advisor's notes**, with verified quotes, and measured: +41 F1 over keywords, 0 invented quotes, injections
   reported and ignored.
3. **Scores before money moves** (pre-release check), and **caught 75% of cases before any stolen money left**.
4. **Built around FINRA's workflow:** Rule 2165/2166 deadlines, Rule 4512 trusted contacts, SIP queue, audit trail.
5. **Evidence a judge can check:** same-workload comparison, CIs, a scam type it never saw, ablations, fairness, 5 worlds,
   v1 frozen, a bug we found and fixed in public. Everything rerun by `make all`.
6. **A person decides.** Nothing is blocked or sent automatically.

## 10. Q&A (short answers)

| Question | Answer |
|---|---|
| It's synthetic. Why trust it? | Calibrated to FinCEN/FBI figures, hard look-alikes, a scam type never seen in training, 5 worlds, results also at a realistic 0.5% base rate. Next step: backtest on LPL's past SIP cases, then run silently beside today's rules. |
| Did you rig the data to suit the model? | v1 frozen before any change; every rate sourced and in one file; baselines re-run on the same world; model chosen on validation; we published a bug that had made our own results look worse. |
| Why not better rules? | Our expert checklist is the better-rules answer: 35% at the same workload while flagging 82% of legitimate look-alikes. Rules see one event at a time and can't read notes. |
| Won't you annoy innocent clients? | Look-alike false alarms are a core metric: 23% vs 36% for the $100k rule at the same workload. The lightest action is a call; nothing is blocked automatically. |
| What if the LLM makes things up? | Quotes must be exact substrings (0 of 2,115 failed); claims without a record ID are rejected by code (178 of 178 kept). |
| Prompt injection? | 439 injected notes: all reported, red flags changed in 1, mitigating context from injected notes never counts. |
| Client notes into an LLM? | Runs inside LPL's environment through Anthropic's API (LPL already works with Anthropic); personal details redacted before the model in production. |
| Isn't this ageist? | Age is never an input; false alarms by age band all within 1.25× of the average. |
| What does it cost? | 0.8 ms per household for the engines; Claude Haiku reads only notes near a risky event, cached by hash; Opus only for flagged households. |
| Is it real-time? | Every event re-scores; the pre-release check acts before money moves. |
| What if a family member is the thief? | POA/joint-owner/trusted-contact payee signals and a hard rule; never contacts an involved trusted contact. 99% of trusted-person theft caught across 5 worlds. |
| What if Rule 2166 isn't approved? | Rule 2165 works today for clients 65+; 2166 is upside. |
| Weak spots? | Government impersonation (47%) and grandparent scams (66%): single, fast withdrawals; small samples. Next: richer notes reading and the network. |

## 11. Do not claim

* Real-world accuracy. Every number is on synthetic data; say "on our simulated book of business".
* That the headline score numbers come from the Claude reader (they come from the fast reader; Claude's +41 F1 is measured
  separately).
* That the live demo is calling Claude in real time (its Claude outputs are pre-generated for offline reliability).
* That anything is integrated with LPL systems today.

## 12. Suggested slides

1. **Title:** Sentinel. "Protect the client, the advisor and the firm." Team cmd+alpha.
2. **Margaret's week** (hook): four events that each look normal; "The money hasn't left yet." One complaint every 2.6 minutes.
3. **The problem:** $7.75B (+59%), 37% of losses, 34 people a day lose >$100k, 20% by someone they know.
4. **Why today's process breaks:** signals in different systems, thresholds miss small repeated theft, notes unread; LPL asked FINRA for clarity.
5. **How Sentinel works:** three streams → engines + mule network + memory → calibrated score with cited reasons → case agent → a person decides.
6. **The AI and its guardrails:** Claude reads notes (+41 F1, 0 invented quotes, injections ignored) and drafts case files (178/178 cited).
7. **Live demo** (switch to dashboard).
8. **Proof:** same workload, 81% vs 4% (with CI); caught before money left 66% vs 2.5%; false alarms 23% vs 36%.
9. **Proof it generalises:** a scam type it never saw; by scam type; 5 worlds; fairness by age.
10. **Fit for LPL:** pre-release check, Cyan skill in ClientWorks, SIP case queue, FINRA 2165/2166/4512.
11. **Impact:** sizing range with labelled assumptions; plus investigator time and regulatory readiness.
12. **Close:** what's next (backtest on LPL's SIP cases, silent run, Claude in the score), and "Protect the client, the advisor and the firm."
