"""Export pipeline output for the dashboard (web/data/*.json).

Every household shown is a real output of the pipeline on a test-split household (never seen in
training). Labels are used only to pick which caught cases to feature in the demo and to show
the ground truth on the Evidence page; the queue itself is the model's alerts.
"""
from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import START, load_labels, load_world
from ..eval import alerts as AL
from ..eval.baselines import hard_rule
from ..eval.run_eval import kw_reader
from ..features import memory
from ..features.build import build_features
from ..generate.text import CATEGORIES
from ..llm import case_agent as CA
from ..llm import notes_reader as NR
from ..models.reasons import Explainer

WEB = Path("web/data")
SHOWCASE_NAMES = {"margaret": ("Margaret", "Ellis"), "harold": ("Harold", "Brennan"), "ruth": ("Ruth", "Castellano")}
KIND_LABEL = {"crypto_exchange": "crypto exchange", "individual": "individual", "external_bank": "newly linked bank",
              "business": "business", "own_bank": "own bank account"}
EVENT_LABEL = {"password_reset": "Password reset", "new_device": "Login from a new device", "email_change": "Email address changed",
               "phone_change": "Phone number changed", "address_change": "Mailing address changed",
               "bank_link_added": "New external bank account linked", "poa_added": "Power of attorney added",
               "joint_owner_added": "Joint owner added", "trusted_contact_changed": "Trusted contact changed",
               "trusted_contact_removed": "Trusted contact removed"}
NOTE_KIND = {"phone_call": "Phone call", "review_meeting": "Review meeting", "email": "Email", "service_request": "Service request"}


def fmt_date(day):
    return (START + pd.Timedelta(days=float(day))).strftime("%b %-d") if hasattr(pd.Timestamp, "strftime") else ""


def date_of(day):
    t = START + pd.Timedelta(days=float(day))
    return f"{t:%b} {t.day}", f"{t:%a}", t


def money(x):
    return f"${x:,.0f}"


class Exporter:
    def __init__(self, world="data/world"):
        self.world = world
        self.W, self.L = load_world(world), load_labels(world)
        with open(Path(world) / "_run" / "sentinel.pkl", "rb") as f:
            run = pickle.load(f)
        self.model, self.split, self.taus, self.reader = run["model"], run["split"], run["taus"], run["reader"]
        self.kw = kw_reader(self.W, self.L, self.split, world)
        readers = {"kw": self.kw}
        llm = NR.cached_outputs(self.W) if self.reader == "llm" else None
        if llm is not None:
            readers["llm"] = llm
        self.reader_out = llm if llm is not None else self.kw
        self.F = build_features(self.W, readers)
        # ---- household memory: full-train stage-1 model scores every row (only test rows are shown)
        self.stage1 = run.get("stage1")
        fired = None
        if self.stage1 is not None:
            self.F = memory.attach(self.F, self.stage1, run.get("memory_prefix", self.reader))
            fired = memory.fired_points(self.F, self.reader_out, self.W["notes"])
        # ---- end household memory
        self.raw = self.model.raw(self.F)
        self.cal = self.model.calibrated(self.raw)
        self.force = hard_rule(self.F)
        self.tau = self.taus["0.25%"]
        self.review_line = float(self.model.calibrated(np.array([self.tau]))[0])
        self.quotes = self._quotes(llm is not None)
        self.explainer = Explainer(self.W, self.reader_out, {k: [q["quote"] for q in v["flags"]] for k, v in self.quotes.items()},
                                   fired=fired)
        t = (self.F["household_id"].map(self.split) == "test").to_numpy()
        self.test = t
        hh = self.W["hh_index"].loc[self.F["household_id"]].to_numpy()
        idx = AL.simulate(hh[t], self.F["day"].to_numpy()[t], self.raw[t], self.tau, self.force[t])
        self.alerts = AL.alerts_frame(self.F[t], idx, self.raw[t])
        self.alerts["cal"] = self.model.calibrated(self.alerts["score"].to_numpy())
        self.alerts["hard"] = self.force[t][idx] if len(idx) else []
        self.Fi = self.F.reset_index(drop=True)
        self.by_h = {h: g.index.to_numpy() for h, g in self.Fi.groupby("household_id")}
        self.mm = self.W["money_movements"].merge(self.W["payees"][["payee_id", "kind", "display_name", "person_id"]], on="payee_id", how="left").set_index("txn_id")
        self.ae = self.W["account_events"].set_index("event_id")
        self.notes = self.W["notes"].set_index("note_id")
        self.hh = self.W["households"].set_index("household_id")
        self.persons = self.W["persons"].set_index("person_id")
        self.accounts = self.W["accounts"]

    def _quotes(self, use_llm):
        if use_llm:
            return NR.cached_quotes(self.W)
        out = {}
        for r in self.kw.itertuples():
            if r.quotes or r.mit_quotes or r.inj_quotes:
                out[r.note_id] = {"flags": [{"cat": CATEGORIES[i], "quote": q} for i, q in r.quotes],
                                  "mitigating": list(r.mit_quotes), "injections": list(r.inj_quotes)}
        return out

    # ---------------------------------------------------------- showcase selection
    def pick_showcase(self):
        A, C = self.alerts, self.L["cases"]
        C = C[C["household_id"].map(self.split) == "test"]
        cand = {"margaret": [], "harold": []}
        for c in C.itertuples():
            a = A[(A["household_id"] == c.household_id) & (A["day"] >= c.start_day) & (A["day"] <= c.last_outflow_day + 1e-6)]
            if not len(a):
                continue
            first = a.iloc[0]
            pre = first["day"] <= c.first_outflow_day + 1e-6
            notes = self.W["notes"]
            hn = notes[(notes["household_id"] == c.household_id) & (notes["day"] <= first["day"]) & (notes["day"] > first["day"] - 90)]
            nflags = sum(len(self.quotes.get(n, {}).get("flags", [])) for n in hn["note_id"])
            if c.typology in ("investment_romance_scam", "tech_support_impersonation", "government_impersonation", "grandparent_voice_clone"):
                # prefer: caught on a pending request, before money left, with flagged notes
                amt = self.mm.loc[first["ref_id"], "amount"] if first["kind"] == "mm" else 1.0
                peak = float(self.model.calibrated(np.array([a["score"].max()]))[0])
                quality = 4 * pre + 3 * (first["kind"] in ("mm", "note")) + 3 * min(nflags, 3) + 6 * peak
                cand["margaret"].append((quality + np.log10(amt), c.household_id, first["day"], c.case_id))
            if c.typology == "trusted_person_theft":
                hard = a[a["hard"].astype(bool)]
                peak = float(self.model.calibrated(np.array([a["score"].max()]))[0])
                day = hard.iloc[0]["day"] if len(hard) else first["day"]  # show the moment the hard rule fired
                cand["harold"].append((5 * bool(len(hard)) + 3 * peak + nflags + c.total_loss / 1e4, c.household_id, day, c.case_id))
        neg = self.L["negatives"]
        neg = neg[(neg["household_id"].map(self.split) == "test") & neg["kind"].isin(["home_purchase", "moving_states", "long_term_care_deposit"])]
        ruth = []
        for e in neg.itertuples():
            ids = [r for r in str(e.related_record_ids).split(";") if r.startswith("TX-")]
            if not ids:
                continue
            big = self.mm.loc[ids, "amount"].max()
            d = self.mm.loc[ids, "day"].min() if "day" in self.mm else None
            a = A[(A["household_id"] == e.household_id) & (A["day"] >= d - 30) & (A["day"] <= d + 30)]
            hn = self.W["notes"][(self.W["notes"]["household_id"] == e.household_id) & (self.W["notes"]["day"].between(d - 30, d + 1))]
            explained = any(r.startswith("N-") for r in str(e.related_record_ids).split(";"))
            if big >= 100000 and not len(a):
                ruth.append((big / 1e6 + 10 * explained + (e.kind == "home_purchase"), e.household_id, d))
        self._cand = {**cand, "ruth": ruth}
        g = self.hh["gender"]
        pick = lambda lst, sex: max([x for x in lst if g[x[1]] == sex] or lst) if lst else None
        m, h, r = pick(cand["margaret"], "F"), pick(cand["harold"], "M"), pick(ruth, "F")
        self.as_of = m[2]
        return {"margaret": (m[1], m[2]), "harold": (h[1], h[2]) if h else None, "ruth": (r[1], r[2]) if r else None}

    # ---------------------------------------------------------- one client
    def describe(self, i):
        r = self.Fi.loc[i]
        if r["kind"] == "mm":
            t = self.mm.loc[r["ref_id"]]
            new = ", new payee" if t["payee_is_new"] else ""
            held = f", sold a position held {t['source_holding_days'] / 365:.0f} years" if pd.notna(t["source_holding_days"]) and t["source_holding_days"] >= 1825 else ""
            meth = {"ach": "Transfer", "wire": "Wire", "check": "Check", "acat": "Account transfer", "journal": "Journal"}.get(t["method"], "Withdrawal")
            return f"{meth} of {money(t['amount'])} requested to {t['display_name']} ({KIND_LABEL.get(t['kind'], t['kind'])}{new}){held}."
        if r["kind"] == "ae":
            e = self.ae.loc[r["ref_id"]]
            who = ""
            if pd.notna(e["related_person_id"]) and e["related_person_id"] in self.persons.index:
                p = self.persons.loc[e["related_person_id"]]
                who = f": {p['name']} ({p['relationship']})"
            return f"{EVENT_LABEL.get(e['type'], e['type'])}{who} via {e['channel']}."
        n = self.notes.loc[r["ref_id"]]
        txt = n["text"] if len(n["text"]) < 110 else n["text"][:107].rsplit(" ", 1)[0] + "..."
        return f"{NOTE_KIND.get(n['kind'], 'Note')}: {txt}"

    def client(self, h, as_of, key=None):
        idx = self.by_h[h]
        rows = self.Fi.loc[idx]
        rows = rows[rows["day"] <= as_of + 1e-6]
        hh = self.hh.loc[h]
        A = self.alerts[(self.alerts["household_id"] == h) & (self.alerts["day"] <= as_of + 1e-6) & (self.alerts["day"] > as_of - 30)]
        if len(A):
            # peak risk since the household was alerted (later points are not re-alerted while under review)
            since = rows[rows["day"] >= A.iloc[0]["day"] - 1e-6].index
            alert_i = since[np.argmax(self.raw[since])]
        else:
            # cleared client: explain the largest withdrawal in the last 30 days
            recent = rows[(rows["day"] > as_of - 30) & (rows["kind"] == "mm")]
            recent = recent if len(recent) else rows
            alert_i = recent.index[np.argmax(np.expm1(recent["amt_log"].to_numpy()))]
        ar = self.Fi.loc[alert_i]
        score = float(self.cal[alert_i])
        hard = bool(self.force[alert_i])
        contrib = self.model.contributions(self.Fi.loc[[alert_i]]).iloc[0]
        reasons = self.explainer.explain(h, ar["day"], ar["ref_id"], contrib)
        pos = sum(max(x["points"], 0) for x in reasons) or 1
        base = float(self.cal[rows.index[0]]) if len(rows) else 0
        for x in reasons:
            x["display"] = round((score - min(base, score)) * x["points"] / pos, 2) if x["points"] > 0 else round(max(-0.2, x["points"] / 20), 2)
        hot = {rid for x in reasons for rid in x["records"]}
        # timeline: notable events in the 90 days up to the alert (and anything since)
        win = rows[(rows["day"] > ar["day"] - 90)]
        notable = win[(win["kind"] != "mm") | (win["recurring"] == 0)]
        notable = notable[(notable["kind"] != "note") | notable["ref_id"].isin(hot) | (notable.index >= alert_i - 0)]
        keep = list(notable.index[-11:])
        if alert_i not in keep:
            keep = keep[-10:] + [alert_i]
        # every record a reason or claim cites stays on the timeline, even if older or recurring
        keep = sorted(set(keep) | set(rows.index[rows["ref_id"].isin(hot)]))
        events = []
        for j in sorted(set(keep)):
            d, wd, _ = date_of(self.Fi.loc[j, "day"])
            rid = self.Fi.loc[j, "ref_id"]
            events.append({"date": d, "day": wd, "text": self.describe(j), "ref": rid, "hot": rid in hot, "score": round(float(self.cal[j]), 3)})
        first_day = min(self.Fi.loc[keep, "day"].min(), win["day"].min() if len(win) else ar["day"])
        # what happened next: shown in grey on the dashboard, never a model input (records after the as-of date)
        allrows = self.Fi.loc[idx]
        last = self.Fi.loc[keep, "day"].max()
        nxt = allrows[(allrows["day"] > last + 1e-6) & (allrows["day"] <= ar["day"] + 45)]
        nxt = nxt[(nxt["kind"] != "mm") | (nxt["recurring"] == 0)]
        events_after = []
        for j in nxt.index:
            d, wd, _ = date_of(self.Fi.loc[j, "day"])
            rid = self.Fi.loc[j, "ref_id"]
            e = {"date": d, "day": wd, "text": self.describe(j), "ref": rid, "hot": False, "score": round(float(self.cal[j]), 3)}
            if self.Fi.loc[j, "kind"] == "mm":
                e["amount"] = float(self.mm.loc[rid, "amount"])
            events_after.append(e)
        events.insert(0, {"date": date_of(first_day - 1)[0], "day": "Baseline", "text": "Routine activity before this period.", "ref": "Baseline",
                          "score": round(float(self.cal[rows.index[rows["day"] < first_day][-1]]) if (rows["day"] < first_day).any() else 0.0, 3)})
        # notes with verified quotes
        hn = self.W["notes"][(self.W["notes"]["household_id"] == h) & (self.W["notes"]["day"] <= as_of + 1e-6) & (self.W["notes"]["day"] > as_of - 120)].tail(8)
        notes, flags, mit, inj = [], [], [], []
        for n in hn.itertuples():
            d, _, _ = date_of(n.day)
            notes.append({"id": n.note_id, "date": d, "kind": NOTE_KIND.get(n.kind, "Note"), "text": n.text})
            q = self.quotes.get(n.note_id, {})
            flags += [{"cat": f["cat"], "quote": f["quote"], "note": n.note_id} for f in q.get("flags", []) if f["quote"] in n.text]
            mit += [{"quote": x, "note": n.note_id} for x in q.get("mitigating", []) if x in n.text]
            inj += [{"quote": x, "note": n.note_id} for x in q.get("injections", []) if x in n.text]
        # case file (Claude agent cache if present, else template)
        cd = CA.CaseData(self.W, h, as_of, reasons, {k: [q["quote"] for q in v["flags"]] for k, v in self.quotes.items()}, score)
        pattern = self.pattern(reasons, flags)
        as_of_date = (START + pd.Timedelta(days=float(as_of))).date().isoformat()
        cache = json.loads(CA.CACHE.read_text()) if CA.CACHE.exists() else {}
        if h in cache and cache[h].get("as_of") == as_of_date:
            cf, rep = cache[h]["case_file"], cache[h]["report"]
        else:
            cf, rep = CA.template_case_file(cd, as_of_date, pattern)
        tc = cd.tc
        tcname = None
        if tc and tc in self.persons.index:
            p = self.persons.loc[tc]
            tcname = f"{p['name']}, {p['relationship']}"
        dr = cf["drafts"]
        drafts = {"Call script": dr["call_script"],
                  "Trusted-contact note": dr["trusted_contact_note"] or ("Withheld.\n\nThe trusted contact on file may be involved, so Sentinel does not draft outreach to them."
                                                                         if cd.tc_involved else "No trusted contact on file. Consider asking the client to name one (FINRA Rule 4512)."),
                  "Escalation memo": dr["escalation_memo"], "APS referral": dr["aps_referral"]}
        # pending money: the alerting request itself, if it is an outflow at the as-of date
        pending = float(self.mm.loc[ar["ref_id"], "amount"]) if ar["kind"] == "mm" else 0.0
        mm30 = self.mm[(self.mm["household_id"] == h) & (self.mm["day"].between(as_of - 30, as_of + 1e-6))]
        thr_flag = bool((mm30["amount"] >= 100000).any())
        is_alert = len(A) > 0
        tier = "crit" if (hard or score >= 0.5) and is_alert else "warn" if is_alert else "safe"
        tier_text = ("Escalate to specialists" if hard else "Hold candidate" if pending and tier == "crit" else "Call today") if is_alert else "No action"
        if is_alert and not thr_flag:
            thr = f"Misses it. Largest withdrawal in 30 days was {money(mm30['amount'].max())}" if len(mm30) else "Misses it. No withdrawal over $100k"
        elif is_alert:
            thr = "Flags it, with no context or case file"
        elif thr_flag:
            thr = "False alarm. Interrupts a legitimate withdrawal"
        else:
            thr = "Not flagged"
        if cd.tc_involved:
            callout = "The trusted contact may be involved: they are receiving money or recently gained a role on the account. Sentinel will not draft a message to them."
        elif inj:
            callout = "A note contained an instruction aimed at an AI. Sentinel treated it as data and reported it."
        elif is_alert and ar["kind"] == "note" and not pending:
            callout = "Flagged from the advisor's notes before any unusual withdrawal. The money hasn't left yet."
        elif pending and is_alert:
            callout = f"The {money(pending)} request has not been released. It is waiting for review."
        elif not is_alert and thr_flag:
            callout = "A threshold rule would interrupt this client. Sentinel's score stays low because the context checks out."
        else:
            callout = "Sentinel re-scores this household the moment anything new happens on the account."
        opened = pd.to_datetime(self.accounts.loc[self.accounts["household_id"] == h, "opened_date"]).min()
        c = {"id": key or h.lower(), "household_id": h, "name": hh["client_name"], "age": int(hh["age"]), "since": int(opened.year),
             "assets": float(hh["assets"]), "pending": pending, "score": round(score, 3), "tier": tier, "tierText": tier_text,
             "pattern": pattern if is_alert else "Large withdrawal, context checks out" if thr_flag else "No unusual activity",
             "short": "; ".join(x["text"] for x in reasons[:2] if x["family"] != "mitigating") + "." if reasons else "No unusual activity.",
             "threshold": {"flag": thr_flag, "text": thr}, "trusted": {"name": tcname or "None on file", "safe": not cd.tc_involved},
             "callout": callout, "events": events, "events_after": events_after,
             "reasons": [{"t": x["text"], "pts": x["display"], "src": f"{x['engine']} · {', '.join(x['records'][:4])}", "quotes": x.get("quotes", [])} for x in reasons],
             "notes": notes, "flags": flags, "mitigating": mit, "injections": inj, "drafts": drafts,
             "case_file": {"claims": cf["claims"], "deadlines": cf["deadlines"], "recommended_action": cf["recommended_action"],
                           "generated_by": cf.get("generated_by", "Claude case agent"), "faithfulness": rep.get("faithfulness"),
                           "claims_rejected": len(rep.get("claims_rejected", []))},
             "reason_codes": reasons, "alert_date": date_of(ar["day"])[0], "hard_rule": hard, "reader": "Claude" if self.reader == "llm" else "keyword baseline"}
        return c

    def pattern(self, reasons, flags):
        fams = {x["family"] for x in reasons}
        cats = {f["cat"] for f in flags}
        if "role" in fams or "structuring" in fams:
            return "Possible exploitation by someone close to the client"
        if CATEGORIES[6] in cats:
            return "Possible impersonation or emergency scam"
        if CATEGORIES[0] in cats or "crypto" in fams or CATEGORIES[7] in cats:
            return "Possible investment, crypto or romance scam"
        if "takeover" in fams:
            return "Possible account takeover"
        return "Unusual money movement"

    # ---------------------------------------------------------- everything
    def run(self):
        show = self.pick_showcase()
        as_of = self.as_of
        A = self.alerts
        open_ = A[(A["day"] > as_of - 30) & (A["day"] <= as_of + 1e-6)].groupby("household_id")["day"].max()
        clients = {}
        for key, v in show.items():
            if v and key != "ruth":
                inside = as_of - 30 < v[1] <= as_of
                # outside today's window: show the household at the moment its alert fired
                clients[v[0]] = self.client(v[0], as_of if inside else v[1] + 1e-3, key)
        for h in open_.index:
            if h not in clients:
                clients[h] = self.client(h, as_of)
        flagged = sorted(clients.values(), key=lambda c: -c["score"])
        cleared = []
        if show.get("ruth"):
            h, d = show["ruth"]
            cleared.append(self.client(h, as_of if as_of - 30 < d <= as_of else min(d + 1, 729.9), "ruth"))
        for c in flagged + cleared:
            k = c["id"]
            if k in SHOWCASE_NAMES:
                rename(c, self.hh.loc[c["household_id"], "client_name"], *SHOWCASE_NAMES[k])
        meta = {"as_of": (START + pd.Timedelta(days=float(as_of))).date().isoformat(), "review_line": round(self.review_line, 3),
                "budget": "0.25% of households per week", "households_scored": int(self.test.sum() and (self.split == "test").sum()),
                "reader": "Claude" if self.reader == "llm" else "keyword baseline (Claude not run yet)",
                "agent": "Claude case agent" if CA.CACHE.exists() else "template drafts (Claude case agent not run yet)",
                "showcase_note": "Margaret, Harold and Ruth are real pipeline outputs on synthetic test households, renamed to match the pitch."}
        specialist = []
        for c in flagged:
            if c["tier"] == "crit" or c["hard_rule"]:
                specialist.append({"id": c["id"], "name": c["name"], "pattern": c["pattern"], "score": c["score"], "alert_date": c["alert_date"],
                                   "pending": c["pending"], "trusted_contact_involved": not c["trusted"]["safe"],
                                   "deadlines": CA.compute_deadlines(meta["as_of"], eligible_2165=c["age"] >= 65),
                                   "claims": c["case_file"]["claims"], "status": "Awaiting specialist review"})
        WEB.mkdir(parents=True, exist_ok=True)
        write(WEB / "clients.json", {"meta": meta, "flagged": flagged, "cleared": cleared})
        write(WEB / "specialist.json", {"meta": meta, "cases": specialist})
        print("showcase candidates:", {k: len(v) for k, v in self._cand.items()})
        print(f"as of {meta['as_of']}: {len(flagged)} in queue, {len(specialist)} for specialists, showcase={ {k: (v[0] if v else None) for k, v in show.items()} }")
        return meta


def rename(obj, old_full, new_first, new_last):
    of, ol = old_full.split(" ", 1)

    def fix(s):
        s = s.replace(old_full, f"{new_first} {new_last}").replace(of, new_first).replace(ol, new_last)
        return s.replace(of.lower(), new_first.lower()).replace(ol.lower(), new_last.lower())  # typo-prone notes are lowercase
    for k, v in list(obj.items()):
        if isinstance(v, str):
            obj[k] = fix(v)
        elif isinstance(v, dict):
            rename(v, old_full, new_first, new_last)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                if isinstance(x, dict):
                    rename(x, old_full, new_first, new_last)
                elif isinstance(x, str):
                    v[i] = fix(x)


def write(p: Path, obj):
    p.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def evidence():
    """Copy the evaluation outputs the Evidence page reads."""
    WEB.mkdir(parents=True, exist_ok=True)
    for f in ("metrics.json", "seeds.json"):
        if Path("results", f).exists():
            shutil.copy(Path("results", f), WEB / f)
