# Sentinel dashboard: UI

Owner: Shaurya (UI design, `web/index.html`). Written 12 Oct 2026 for the 13 Oct submission and the 17 Oct finals.

`web/index.html` is the final dashboard. Part 1 is what the research says good tools of this kind do. Part 2 is the design:
look, layout, charts, components and mobile rules. Part 3 is the implementation, with what Teammate B still needs to add.

Rule for everything below: **no number is typed into the page.** Every number comes from `web/data/*.json`. The only
constants in the code are UI logic: for example, a deadline chip turns red at 1 business day or less.

---

## Part 1. What the research says

The sources were vendor docs (Unit21, Sardine, Hummingbird, Hawk AI, Stripe Radar, Feedzai), design systems (IBM Carbon,
GitHub Primer), Google's People + AI Guidebook, Microsoft's HAX guidelines, NN/g, WCAG 2.2, and FINRA's rule texts. Links
are at the end.

**Alert queues.** Fraud and AML tools rank by risk first and urgency second. Each alert shows its status, its pattern
(typology) and its deadline, and filter chips narrow the list. Stripe Radar always shows a band word next to the score
("elevated", "high"), never the number alone. Sardine groups alerts by customer, which is the same as Sentinel's
household rows.

**Case screens.** Unit21 says outright that "a probability score with no supporting rationale" won't hold up in an exam.
Every claim has to trace back to the data behind it. Hawk AI cites a source for each conclusion so an auditor can follow
it. The common layout is a header (entity, score, status, deadline), then the evidence: why, timeline, notes, drafts and a
decision.

**Explaining a score.** Google PAIR says numeric confidence "assumes users understand probability", so give it context and
say what action to take. Microsoft HAX says to "match the level of precision in the UI with the system performance" and to
"make clear why the system did what it did". People reason better with natural frequencies ("about 75 in 100") than with
decimals. Explanations alone don't stop people over-trusting AI (Buçinca et al. 2021), so the decision has to stay with a
person, and the screen should say so.

**"What happened next" charts.** News graphics reveal the future part of a line in grey after a clear marker (the NYT "You
Draw It" pattern). The key annotation, here the alert moment, must be visible without hovering: "if you make a tooltip,
assume no one will ever see it" (Archie Tse, NYT). Context goes grey, and only the story is in colour.

**Colour and accessibility.** WCAG needs 4.5:1 for text and 3:1 for chart marks and icons. IBM Carbon's status pattern
uses at least two of colour, shape and symbol, plus a word. About 1 in 12 men can't tell red from green, so tiers can't
rely on colour.

**Mobile.** NN/g says to narrow the data with filters first, then turn table rows into cards or lock the first column.
WCAG 2.2 sets touch targets at 24px minimum (44px is better), and gutters should be 16px.

**Evidence pages.** Lead with one chart, then tiles that restate its numbers with a comparison ("$100k rule: 4%"). Twenty
equally weighted tiles prioritise nothing. Use direct labels instead of legends, keep the baselines grey, and collapse long
tables.

**Being honest about the AI.** Stripe keeps "not evaluated" separate from "unknown". The provenance-marking pattern labels
every machine output. A "not run yet" state has to look different from a real result.

**FINRA specifics.** Rule 2165 requires notice within 2 business days and allows a hold of 15 business days plus 10. Rule
2166 is *proposed* and pending SEC approval; it would allow a delay of up to 10 business days for any adult. Firms must
not contact a trusted contact who may be involved. FINRA's exam findings list missing hold-review records as a common gap,
so the audit trail matters.

## Part 2. Design

### Look

- **Accents.** Ink (`#1C2128`; `#E8EBEF` in dark mode) marks everything you can press or have selected: primary buttons,
  the chosen filter, toggles, the active tab and the selected evidence tile. Petrol (`#0F5866`; `#6CC0CC` in dark mode)
  marks only Sentinel's own things: its chart line, the wave pattern in page headers, the ring in the logo and the "read by
  Claude" badges. Red, amber and green mean risk state only.
- **Nav.** A light, see-through bar with a thin bottom line, like Stripe and Linear. Links are 14px at regular weight, and
  the current page gets a soft grey pill.
- **Logo.** A clean outline shield with a petrol ring at its centre and "SENTINEL" in spaced capitals. The browser-tab icon
  is a solid petrol shield with a white ring.
- **Type.** Public Sans for text and IBM Plex Mono for record IDs, both embedded. Every score, date, amount and ID uses
  tabular numerals.
- **Surfaces.** White panels with a 14px radius on a faint grey-green page. Dark mode is neutral graphite.
- **Contrast.** Every text and background pair is at least 5.2:1 (WCAG AA). Amber is used for fills only, never text.
- **Motion.** Only when you arrive on a page: headers and cards rise in, the case score counts up, the risk chart draws
  left to right with a pulse at the alert point, and the reason bars grow. The header waves drift slowly. Switching themes
  fades. None of it runs if the computer is set to reduce motion.

### Layout

| Page | Order on screen |
|---|---|
| Queue | Headline · provenance badges · 4 stats · **At a glance** (alerts by pattern, risk-score strip, which engine raised them) · filter chips · ranked rows with a small score-history line |
| Case | Pattern · name · meta · badges · score with tier and a frequency line · callout · tabs · replay chart with "What happened next" · why this score (each reason opens to its evidence) + timeline · case file. The Decide tab adds a deadline timeline. |
| Specialists | **At a glance** (cases by score, coloured by pattern; deadlines if a hold is placed today) · case table on desktop, labelled cards under 1100px |
| Evidence | Headline + lede + badge · hero card (4 tiles that switch one chart) · detail sections · what the model leans on · legitimate life events that look like exploitation |

### Charts: click for the words

Every chart has a hover tooltip and keyboard focus (Enter or Space picks), and clicking opens a plain-English panel.

| Chart | Click shows |
|---|---|
| Queue: donut of open alerts by pattern | What the pattern means, alerts, assets, money held, top case. It also filters the list below. |
| Queue: risk-score strip (▲ higher risk, ◆ lower risk, dashed review line) | That household's summary and a link to the case |
| Queue: engine bars | What the engine looks at, and which households it flagged. The "Claude not run yet" label stays visible. |
| Case: "What this means" on each reason | The engine behind it and the word-for-word quotes from the advisor note |
| Case and Specialists: deadline timeline | The rule behind each date. Rule 2166 is always labelled "proposed". |
| Specialists: score bars | The case agent's claims with record IDs, and the trusted-contact warning |
| Evidence: donut of learned weight by engine | The top signals in that engine, in plain names |
| Evidence: look-alike false alarms vs the $100k rule | What each life event looks like. A sentence computed from the data says where Sentinel does better and where it does worse. |

Donuts are used only for parts of a whole, with at most four slices. Reason contributions can be negative, so they are bars,
not pies. Pattern colours are teal, violet, pink and grey, checked for colour-blind safety, and every slice also has a
direct label.

### Components

| Component | What it does |
|---|---|
| Tier pill | Colour **and** shape **and** word: ▲ red (crit), ◆ amber (warn), ● green (safe). |
| Provenance badge | Dashed outline + ⓘ for "Keyword notes reader · Claude not run yet" and "Template drafts · Claude not run yet". Solid filled + ✦ for "Notes read by Claude" and "Drafted by the Claude case agent". The tooltip includes the raw value from the JSON. |
| Hard-rule badge | Solid red flag pill, shown on the queue row and the case header. |
| Record chip | A record ID like `N-0087769`. If the record is in the timeline, clicking scrolls to it and flashes it yellow. If it's a note, clicking opens the notes tab at that note. If it's outside the window shown on the page, it's dashed and inactive. |
| Switch | "What happened next" toggle with a 44px tap target and `aria-pressed`. |
| Filter chip | Pill with a count; pressed state is ink-filled. On phones the chips scroll sideways in one row. |
| Countdown chip | "Notice in 2 business days", counted from `meta.as_of` and skipping weekends and holidays. Red at ≤1 day, amber at ≤3, outlined otherwise. |
| Evidence tile | Sentinel's value, the metric name and "$100k rule: X". Clicking a tile switches the hero chart to that measure. |
| Theme toggle | Moon/sun in the nav (in the footer on phones). It follows the system setting until someone picks; the choice is saved per browser. |

### Mobile at 375px

* 16px gutters (`--gut: clamp(16px,5vw,56px)`).
* The nav fits with no sideways scroll: logo only, short labels (SIP, Log, About), and links that scroll if a phone is
  even narrower. It was 457px wide before.
* Evidence bar and dot charts put labels above the bars instead of beside them. Before, labels clipped and the bars were
  about 100px wide.
* The Specialists table becomes labelled cards, and the status select is full width and 44px tall.
* Chart points have 32px invisible tap targets.
* Section headers stack instead of squeezing the title into four lines.
* Charts stack in one column, the score-history lines are hidden, and bar labels sit above the bars.

## Part 3. Implementation (all done in `web/index.html`)

### P0 for 13 Oct

1. **"What happened next" replay toggle.** Done; it needs B's data.
   * The toggle appears only when `c.events_after` is a non-empty array, so it hides cleanly today.
   * When on, the chart extends right. A grey band marks the period after the alert, a red vertical line with the static
     label "Sentinel alerted here" marks the alert, the later scores are a dotted grey line, and money requests are red
     rings.
   * Under the chart a summary reads, for example, "3 requests for $205,000 came after the alert", summing
     `events_after[].amount`.
   * The timeline adds the same events in grey under "After Sentinel alerted · not visible to it at the time".
   * The alert marker shows on every case, with or without the toggle.
2. **Mobile pass at 375px.** Done (see Part 2). There is no horizontal scroll on any page, which was checked
   automatically at 375px.
3. **Evidence hierarchy.**
   * Done: the page now leads with the hero card, "Same workload, very different results".
   * The four tiles restate the chart (cases caught, caught before money left, stolen dollars protected, look-alikes
     flagged), each against the $100k rule, and each switches the chart to that measure.
   * The method table under the chart and the "Every reviewer budget" table are collapsed by default.
   * Chart width is now measured inside the page gutters, so desktop text no longer shrinks.
4. **Both reader states.** Done.
   * Badges appear on the queue (reader and agent), the case header, the notes tab, the drafts tab, the case file and the
     Evidence page.
   * They read `META.reader`, `META.agent`, `c.reader`, `c.case_file.generated_by` and `metrics.meta.notes_reader`.
   * They flip on their own when the text contains "Claude" and not "not run". Both states were checked with a mocked
     data file.

### P1 for 17 Oct (also done)

5. **Case file on the case page.** It shows the suggested action, the claims, and record chips on every claim and every
   "Why this score" reason (see the record chip above).
6. **Queue filters.**
   * Chips for pattern (scams, account takeover, someone close, unusual money movement), tier ("Higher risk" / "Lower risk")
     and hard rule, each with a count.
   * Every row shows its alert date, hard-rule badge and pattern, and there is a "No households match" empty state.
7. **Specialists.** Deadline countdown chip, readable dates ("Tue, Aug 5"), a warning that names the trusted contact
   ("Do not notify Lisa Brennan. Sentinel withheld the draft."), and an "Open case file" button.
8. **Polish.**
   * Light and dark checked on every page; a theme toggle.
   * A loading state, and "no data" states for Evidence, Specialists and Activity.
   * A per-page `<title>`; the favicon already existed.
   * The calibrated frequency line under the case score: "About 75 in 100 moments scored like this were real cases, in
     the simulation". It's computed from `c.score`, and hard-rule cases say so instead.

### What Teammate B needs to add (`sentinel/export/build.py`, inside `client()`)

**1. `events_after` (needed for P0 task 1).** Add this before `c = {...}` is built, then add `"events_after": events_after`
to the dict. It must use the unfiltered rows: `rows` is cut at the as-of date, and Margaret's as-of date *is* her alert
date, so the later wires never reach it.

```python
        # what happened next: display only, never a model input
        allrows = self.Fi.loc[idx]
        last = max(self.Fi.loc[j, "day"] for j in keep)
        nxt = allrows[(allrows["day"] > last + 1e-6) & (allrows["day"] <= ar["day"] + 45)]
        nxt = nxt[(nxt["kind"] != "mm") | (nxt["recurring"] == 0)]
        events_after = []
        for j in nxt.index:
            d, wd, _ = date_of(self.Fi.loc[j, "day"])
            rid = self.Fi.loc[j, "ref_id"]
            e = {"date": d, "day": wd, "text": self.describe(j), "ref": rid, "hot": False,
                 "score": round(float(self.cal[j]), 3)}
            if self.Fi.loc[j, "kind"] == "mm":
                e["amount"] = float(self.mm.loc[rid, "amount"])
            events_after.append(e)
```

Shape: the same keys as `events`, plus `amount` on money movements. `score` is optional; if it's missing, the chart
carries the previous score forward. An empty list hides the toggle. The `rename()` step should rename names inside
`text` too; check that "Margaret" appears and not the original name. Then run `python -m sentinel.export`.

**2. Cited records missing from the timeline (nice to have).** In 14 of 37 queued households, a record cited in the case
file is not in `events`, because recurring money movements are filtered out and only the last 11 rows are kept. Margaret's
`TX-0218855` is one example. Those chips show as dashed and inactive. To fix this, always keep `hot` records:
`notable = notable[...] | notable["ref_id"].isin(hot)`.

### How to check it

```
python -m http.server -d web 8000     # open http://localhost:8000
```

1. Open Queue. Try the chips, then clear them. At 375px, use the browser's device toolbar.
2. Open Harold. Click a record chip in the case file; the timeline row should flash. Press Replay.
3. Open Margaret. Once B ships `events_after`, turn on "What happened next".
4. On the queue, click a donut slice, a mark on the score strip and an engine bar; a panel should open each time.
5. Open Evidence. Click each tile; the chart should switch. Click a slice of "What the model leans on".
6. Use the moon/sun button to check dark mode on every page.

### Not done (ideas for after the finals)

* A required "reason" field on the Decide panel (Unit21; FINRA exam findings on documenting hold reviews).
* A sticky compact case header with a deadline countdown.
* An "audit log export" button.
* A colour-blind palette toggle. Shapes and labels already carry tier and pattern, so this is lower priority.

### Sources

Unit21 alerts overview (support.unit21.ai/hc/en-us/articles/6712819596564) · Unit21 agentic AML guide (unit21.ai/blog) ·
Sardine alert queues (docs.sardine.ai) · Hummingbird audit features (help.hummingbird.co) · Hawk AI investigative process
(hawk.ai) · Stripe Radar risk evaluation and risk insights (docs.stripe.com/radar) · Google PAIR Explainability + Trust
(pair.withgoogle.com/chapter/explainability-trust) · Microsoft HAX toolkit (microsoft.com/en-us/haxtoolkit/library) ·
Buçinca et al. 2021, cognitive forcing (eecs.harvard.edu/~kgajos) · Chromik et al. 2021, SHAP explanations for lay users
(LMU) · data.europa.eu narrative story patterns · FlowingData on "You Draw It" · Observable, chart annotations ·
Datawrapper, fewer colours · WCAG 2.2 non-text contrast (w3.org/WAI/WCAG22/Understanding/non-text-contrast) · IBM Carbon
status indicators and type sets (carbondesignsystem.com) · GitHub Primer primitives · NN/g mobile tables
(nngroup.com/articles/mobile-tables) · Smashing Magazine on WCAG target size · FINRA Rule 2165 (finra.org/rules-guidance)
· SR-FINRA-2026-018 (proposed Rule 2166) · FINRA senior investor effective practices and 2023 exam findings on trusted
contacts.
