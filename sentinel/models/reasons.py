"""Reason codes: TreeSHAP contributions grouped into plain-English sentences, each with the
record IDs behind it. Age is never a feature, so it can never be a reason."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..generate.text import CATEGORIES

FAMILIES = {
    "size": ["robust_z", "ratio_max", "amt_log", "share_assets", "out_sum_7d", "out_sum_30d", "out_30d_vs_typical",
             "nonrecurring_sum_30d", "share_out_30d", "out_cnt_7d", "out_cnt_30d", "recurring", "thin_history", "is_outflow"],
    "payee": ["payee_new", "young_payee", "young_payee_sum_30d", "payee_individual", "individual_sum_30d",
              "payee_ext_bank", "method_wire", "method_check", "method_acat", "method_journal"],
    "crypto": ["payee_crypto", "crypto_sum_30d"],
    "long_held": ["sold_long_held"],
    "structuring": ["struct_cnt_60d"],
    "takeover": ["days_since_contact_change", "ato_chain", "ato_chain_young_payee", "access_then_new_payee"]
                + [f"{p}_{t}" for p in ("cnt30", "since") for t in ("password_reset", "new_device", "email_change",
                                                                       "phone_change", "address_change", "bank_link_added")],
    "role": ["tc_change_then_out", "poa_then_out", "payee_is_trusted_contact", "payee_is_poa_or_joint", "payee_is_recent_poa",
             "role_payee_sum_90d", "role_payee_cnt_90d"]
            + [f"{p}_{t}" for p in ("cnt30", "since") for t in ("poa_added", "joint_owner_added",
                                                               "trusted_contact_changed", "trusted_contact_removed")],
}
ENGINE = {"size": "Money movement", "payee": "Money movement", "crypto": "Money movement", "long_held": "Money movement",
          "structuring": "Money movement", "takeover": "Account changes", "role": "Account changes", "notes": "Notes reader"}


def family(feature: str) -> str:
    for fam, feats in FAMILIES.items():
        if feature in feats:
            return fam
    if "mitigating" in feature:
        return "mitigating"
    return "notes"


def money(x):
    return f"${x:,.0f}"


class Explainer:
    def __init__(self, W, reader_out: pd.DataFrame | None, quotes: dict | None = None):
        self.W = W
        mm = W["money_movements"].merge(W["payees"][["payee_id", "kind", "display_name", "person_id", "created_ts"]], on="payee_id", how="left")
        self.mm = {h: g.sort_values("day") for h, g in mm.groupby("household_id")}
        self.ae = {h: g.sort_values("day") for h, g in W["account_events"].groupby("household_id")}
        self.notes = {h: g.sort_values("day") for h, g in W["notes"].groupby("household_id")}
        self.reader = reader_out.set_index("note_id") if reader_out is not None else None
        self.quotes = quotes or {}
        self.persons = W["persons"].set_index("person_id")
        self.hh = W["households"].set_index("household_id")

    def explain(self, household: str, day: float, ref_id: str, contrib: pd.Series, top=3) -> list[dict]:
        fam = contrib.groupby(contrib.index.map(family)).sum()
        out = []
        for f, v in fam.sort_values(ascending=False).items():
            if v <= 0.05 or f == "mitigating" or len(out) >= top:
                continue
            r = self.sentence(f, household, day, ref_id)
            if r:
                out.append({**r, "points": round(float(v), 3), "engine": ENGINE.get(f, "Notes reader"), "family": f})
        if fam.get("mitigating", 0) < -0.05:
            r = self.sentence("mitigating", household, day, ref_id)
            if r:
                out.append({**r, "points": round(float(fam["mitigating"]), 3), "engine": "Notes reader", "family": "mitigating"})
        return out

    def _window(self, d, table, day, w):
        g = d.get(table)
        if g is None:
            return g
        return g[(g["day"] <= day + 1e-9) & (g["day"] > day - w)]

    def sentence(self, f, h, day, ref):
        mm = self._window(self.mm, h, day, 30)
        prior = self.mm.get(h)
        cur = mm[mm["txn_id"] == ref] if mm is not None else None
        cur = cur.iloc[0] if cur is not None and len(cur) else None
        if f == "size":
            if cur is not None:
                p = prior[(prior["day"] < cur["day"]) & (prior["day"] > cur["day"] - 365)]
                mx = p["amount"].max() if len(p) else np.nan
                times = f", {cur['amount'] / mx:.0f}x the largest in the past year" if mx and mx > 0 and cur["amount"] / mx >= 2 else ""
                return {"text": f"Withdrawal request of {money(cur['amount'])}{times}", "records": [cur["txn_id"]]}
            if mm is not None and len(mm):
                return {"text": f"{money(mm['amount'].sum())} requested out in the last 30 days", "records": list(mm["txn_id"].tail(5))}
        if f in ("payee", "crypto"):
            src = mm[(mm["day"] - (mm["created_ts"] - pd.Timestamp('2024-10-01')).dt.total_seconds() / 86400 < 60) & (mm["kind"] != "own_bank")] if mm is not None else None
            if f == "crypto" and mm is not None:
                src = mm[mm["kind"] == "crypto_exchange"]
            if src is not None and len(src):
                r = src.iloc[-1]
                kind = {"crypto_exchange": "a crypto exchange", "individual": "an individual", "external_bank": "a newly linked bank",
                        "business": "a new business payee"}.get(r["kind"], "a new payee")
                return {"text": f"{money(src['amount'].sum())} to {kind} ({r['display_name']}) first used in the last 60 days",
                        "records": list(src["txn_id"].tail(5))}
        if f == "long_held" and cur is not None and pd.notna(cur["source_holding_days"]):
            return {"text": f"Funded by selling a position held {cur['source_holding_days'] / 365:.0f} years", "records": [cur["txn_id"]]}
        if f == "structuring" and prior is not None:
            p = prior[(prior["day"] <= day) & (prior["day"] > day - 60) & (prior["amount"] >= 8000) & (prior["amount"] < 10000)]
            if len(p):
                return {"text": f"{len(p)} transfers just under $10,000 in 60 days", "records": list(p["txn_id"])}
        if f == "takeover":
            ev = self._window(self.ae, h, day, 30)
            if ev is not None:
                ev = ev[ev["type"].isin(["email_change", "phone_change", "bank_link_added", "new_device", "password_reset", "address_change"])]
                if len(ev):
                    names = {"email_change": "email changed", "phone_change": "phone changed", "bank_link_added": "new bank linked",
                             "new_device": "login from a new device", "password_reset": "password reset", "address_change": "address changed"}
                    seq = ", then ".join(dict.fromkeys(names[t] for t in ev["type"]))
                    span = day - ev["day"].min()
                    return {"text": f"Account changes in the {max(1, round(span))} days before: {seq}", "records": list(ev["event_id"])}
        if f == "role":
            ev = self._window(self.ae, h, day, 365)
            recs, bits = [], []
            if ev is not None:
                ev = ev[ev["type"].isin(["poa_added", "joint_owner_added", "trusted_contact_changed", "trusted_contact_removed"])]
                for _, e in ev.iterrows():
                    who = self.persons.loc[e["related_person_id"], "name"] if pd.notna(e["related_person_id"]) and e["related_person_id"] in self.persons.index else None
                    bits.append({"poa_added": f"power of attorney granted to {who}", "joint_owner_added": f"{who} added as joint owner",
                                 "trusted_contact_changed": f"trusted contact changed to {who}", "trusted_contact_removed": "trusted contact removed"}[e["type"]])
                    recs.append(e["event_id"])
            tc = self.hh.loc[h, "trusted_contact_id"]
            people = set(ev["related_person_id"].dropna()) if ev is not None else set()
            if pd.notna(tc):
                people.add(tc)
            p90 = self._window(self.mm, h, day, 90)
            if p90 is not None:
                to = p90[p90["person_id"].isin(people)]
                if len(to):
                    name = to.iloc[-1]["display_name"]
                    bits.append(f"{money(to['amount'].sum())} sent to {name} in 90 days, who holds a role on the account")
                    recs += list(to["txn_id"].tail(6))
            if bits:
                t = "; ".join(bits[-2:])
                return {"text": t[0].upper() + t[1:], "records": recs[-8:]}
        if f in ("notes", "mitigating") and self.reader is not None:
            nt = self._window(self.notes, h, day, 90)
            if nt is None or not len(nt):
                return None
            rows = self.reader.reindex(nt["note_id"])
            if f == "notes":
                hit = rows[[c for c in rows.columns if c.startswith("flag_")]].fillna(0).sum(axis=1) > 0
                ids = list(rows.index[hit.to_numpy()])
                if not ids:
                    return None
                cats = []
                for c in [c for c in rows.columns if c.startswith("flag_")]:
                    if rows.loc[ids, c].fillna(0).sum() > 0:
                        cats.append(CATEGORIES[int(c.split("_")[1])].split(" (")[0].split(",")[0].lower())
                q = [qq for i in ids for qq in self.quotes.get(i, [])][:3]
                return {"text": "Advisor notes show " + ", ".join(cats[:3]), "records": ids[-4:], "quotes": q}
            ids = list(rows.index[rows["mitigating"].fillna(0).to_numpy() > 0])
            if ids:
                return {"text": "Notes contain context that lowers concern", "records": ids[-3:]}
        return None
