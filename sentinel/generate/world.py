"""Seeded synthetic LPL book of business (playbook section 9).

Writes data/world/*.csv (what the models may read) and data/world/labels/*.csv
(ground truth, read only by the evaluation harness). Record IDs are assigned after
sorting by time, so no ID ordering can leak a label.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import text as T

DATA_FILES = ["advisors", "households", "persons", "accounts", "positions", "payees",
              "money_movements", "account_events", "notes"]
LABEL_FILES = ["cases", "negatives"]
VERB_START = {"asked", "said", "wanted", "kept", "called", "pressed", "requested", "told", "did", "wants", "plans",
              "was", "could", "gave", "had", "seemed", "thought", "repeated", "forgot", "got", "keeps", "has", "is",
              "only", "gets", "described", "brought", "and"}


def load_config(path="config/world.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class World:
    def __init__(self, cfg: dict, seed: int | None = None, households: int | None = None):
        self.cfg = cfg
        self.seed = cfg["seed"] if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.N = households or cfg["population"]["households"]
        self.start = pd.Timestamp(cfg["start_date"])
        self.end_day = (self.start + pd.DateOffset(months=cfg["months"]) - self.start).days
        self.month_starts = [(self.start + pd.DateOffset(months=m) - self.start).days for m in range(cfg["months"])]
        self.rows = {k: [] for k in DATA_FILES + LABEL_FILES}
        self._k = 0  # tmp key counter
        self.person_names = {}
        self._buf, self._i = [], 0
        self.tag_times = {}  # tag -> timestamps of its money/account records (anchors for notes)

    # ------------------------------------------------------------ helpers
    def key(self):
        self._k += 1
        return self._k

    def r(self):
        # buffered uniform stream: same seed -> same sequence, without per-call numpy overhead
        if self._i >= len(self._buf):
            self._buf, self._i = self.rng.random(1 << 20).tolist(), 0
        self._i += 1
        return self._buf[self._i - 1]

    def ri(self, lo, hi):  # inclusive
        return lo + int(self.r() * (hi - lo + 1))

    def pick(self, seq):
        return seq[int(self.r() * len(seq))]

    def hour(self, day, lo=8, hi=18):
        return day + (lo + self.r() * (hi - lo)) / 24.0

    def slots(self, h, **extra):
        f = h["gender"] == "F"
        s = {"S": "she" if f else "he", "O": "her" if f else "him", "P": "her" if f else "his",
             "Name": h["first"], "fam": h["fam_label"], "statename": T.STATE_NAMES[h["state"]],
             "friend": self.pick(T.SCAMMER), "platform": self.pick(T.PLATFORMS), "agency": self.pick(T.AGENCIES),
             "techco": self.pick(T.TECH_COS), "coin": self.pick(T.COINS),
             "relative": self.pick(["grandson", "granddaughter", "nephew", "grandson Tyler", "niece"]),
             "perp": h["fam_label"], "hobby": "", "extra": ""}
        s.update(extra)
        return s

    def sentence(self, phrase, sl):
        first = phrase.split(" ", 1)[0]
        if first in VERB_START:
            phrase = ("{Name} " if self.r() < 0.5 else "{S} ") + phrase
        out = T.fill(phrase, sl)
        out = out[0].upper() + out[1:]
        return out if out.endswith((".", "!", "?")) else out + "."

    # ------------------------------------------------------------ add rows
    def add_payee(self, h, day, kind, name, person=None, tag=None):
        k = self.key()
        self.rows["payees"].append(dict(_k=k, household_id=h["id"], kind=kind, display_name=name,
                                        person_id=person, _day=day, _tag=tag))
        return k

    def add_mm(self, h, day, amount, method, payee, holding_days=None, tag=None, account=None):
        k = self.key()
        status = "cancelled" if self.r() < self.cfg["normal_money"]["cancelled_rate"] else "completed"
        if tag is not None:
            self.tag_times.setdefault(tag, []).append(day)
        self.rows["money_movements"].append(dict(
            _k=k, household_id=h["id"], account_id=account or h["accounts"][0], _day=day,
            amount=round(float(amount), 2), method=method, _payee=payee,
            source_holding_days=holding_days, status=status, _tag=tag))
        return k

    def add_ae(self, h, day, typ, channel="web", device=None, person=None, tag=None):
        k = self.key()
        if tag is not None:
            self.tag_times.setdefault(tag, []).append(day)
        self.rows["account_events"].append(dict(_k=k, household_id=h["id"], _day=day, type=typ, channel=channel,
                                                device_id=device or h["device"], related_person_id=person, _tag=tag))
        return k

    def add_note(self, h, day, kind, text, tag=None, cats=None, inject=None):
        k = self.key()
        adv = self.advisors[h["advisor"]]
        text = T.apply_style(text, adv["style"], self.rng, self.slots(h))
        if inject is None:
            inject = self.r() < self.cfg["notes"]["injection_rate"]
        if inject:
            text = text + " " + self.pick(T.INJECTIONS)
        self.rows["notes"].append(dict(_k=k, household_id=h["id"], advisor_id=h["advisor"], _day=day, kind=kind,
                                       text=text, _tag=tag, _cats=";".join(map(str, cats or [])), _inj=int(inject)))
        return k

    def add_person(self, h, name, rel):
        pid = f"PR-{h['idx']:05d}-{len(h['persons'])}"
        h["persons"].append(pid)
        self.person_names[pid] = name
        self.rows["persons"].append(dict(person_id=pid, household_id=h["id"], name=name, relationship=rel))
        return pid

    def first_for(self, rel):
        if rel in ("daughter", "niece", "sister", "granddaughter", "wife"):
            return self.pick(T.YOUNG_F)
        if rel in ("son", "nephew", "brother", "grandson", "husband"):
            return self.pick(T.YOUNG_M)
        return self.pick(T.YOUNG_F + T.YOUNG_M)

    def holding_days(self, h, long=True):
        if long:
            return int(self.rng.integers(5 * 365 + 30, 25 * 365))
        return int(self.rng.integers(60, 5 * 365))

    # ------------------------------------------------------------ build
    def build(self):
        cfg, pop = self.cfg, self.cfg["population"]
        n_adv = pop["advisors"]
        styles = cfg["notes"]["styles"]
        self.advisors = []
        for a in range(n_adv):
            self.advisors.append(dict(advisor_id=f"ADV-{a:03d}", name=f"{self.pick(T.YOUNG_F + T.YOUNG_M)} {self.pick(T.LAST)}",
                                      style=styles[a % len(styles)], region=self.pick(["Northeast", "Southeast", "Midwest", "Southwest", "West"])))
        self.rows["advisors"] = [dict(a) for a in self.advisors]
        sizes = self.rng.uniform(pop["advisor_book_min"], pop["advisor_book_max"], n_adv)
        sizes = np.floor(sizes / sizes.sum() * self.N).astype(int)
        sizes[: self.N - sizes.sum()] += 1
        adv_of = self.rng.permutation(np.repeat(np.arange(n_adv), sizes))

        ages = np.clip(np.round(self.rng.normal(pop["age_mean"], pop["age_sd"], self.N)), pop["age_min"], pop["age_max"]).astype(int)
        assets = np.clip(np.exp(self.rng.normal(math.log(pop["assets_median"]), pop["assets_sigma"], self.N)),
                         pop["assets_min"], pop["assets_max"])
        self.H = []
        for i in range(self.N):
            self.H.append(self.make_household(i, int(ages[i]), float(assets[i]), int(adv_of[i])))
        for h in self.H:
            self.normal_activity(h)
        self.plant_cases()
        self.plant_lookalikes()
        return self.finalize()

    def make_household(self, i, age, assets, adv):
        pop = self.cfg["population"]
        g = "F" if self.r() < 0.56 else "M"
        first = self.pick(T.FIRST_F if g == "F" else T.FIRST_M)
        last = self.pick(T.LAST)
        h = dict(idx=i, id=f"HH-{i:05d}", age=age, assets=round(assets, -2), advisor=adv, gender=g, first=first,
                 last=last, state=self.pick(T.STATES), persons=[], device=f"D{int(self.rng.integers(16**8)):08x}",
                 notes_on=self.r() >= self.cfg["notes"]["no_notes_rate"])
        # family
        kids = []
        for _ in range(self.ri(0, 3)):
            kg = self.r() < 0.5
            name = f"{self.pick(T.YOUNG_F if kg else T.YOUNG_M)} {last if self.r() < 0.7 else self.pick(T.LAST)}"
            rel = "daughter" if kg else "son"
            kids.append((self.add_person(h, name, rel), name, rel))
        h["kids"] = kids
        tc = None
        if self.r() < pop["trusted_contact_rate"]:
            if kids:
                tc = kids[0]
            else:
                rel = self.pick(["niece", "nephew", "friend", "sister", "brother"])
                name = f"{self.first_for(rel)} {self.pick(T.LAST)}"
                tc = (self.add_person(h, name, rel), name, rel)
        h["tc"] = tc
        fam = kids[0] if kids else tc
        h["fam_label"] = f"{fam[2]} {fam[1].split()[0]}" if fam else "family"
        h["fam_person"] = fam[0] if fam else None
        poa_rate = pop["poa_at_start_rate_over_80"] if age > 80 else pop["poa_at_start_rate"]
        h["has_poa"] = bool(kids) and self.r() < poa_rate
        # accounts and positions
        types = ["ira"] + [t for t, p in [("brokerage", .7), ("advisory", .4), ("trust", .1), ("joint", .25)] if self.r() < p]
        w = self.rng.dirichlet(np.ones(len(types)) * 2)
        h["accounts"] = []
        for t, share in zip(types, w):
            aid = f"AC{i:05d}{len(h['accounts'])}"
            h["accounts"].append(aid)
            opened = -self.ri(365, 25 * 365)
            self.rows["accounts"].append(dict(account_id=aid, household_id=h["id"], type=t,
                                              opened_date=(self.start + pd.Timedelta(days=opened)).date().isoformat()))
            vals = self.rng.dirichlet(np.ones(self.ri(2, 6))) * share * h["assets"]
            for v in vals:
                acq = self.ri(opened, -30)
                self.rows["positions"].append(dict(account_id=aid, security_type=self.pick(
                    ["equity_fund", "bond_fund", "stock", "etf", "annuity", "money_market", "muni_bond"]),
                    acquired_date=(self.start + pd.Timedelta(days=acq)).date().isoformat(), value=round(float(v), 2)))
        h["own_bank"] = self.add_payee(h, -self.ri(400, 4000), "own_bank", f"{first} {last} checking")
        self.rows["households"].append(dict(household_id=h["id"], advisor_id=self.advisors[adv]["advisor_id"],
                                            client_name=f"{first} {last}", gender=g, age=age, state=h["state"],
                                            assets=h["assets"], trusted_contact_id=tc[0] if tc else None,
                                            has_poa=h["has_poa"]))
        return h

    # ------------------------------------------------------------ normal life
    def normal_activity(self, h):
        nm, ac = self.cfg["normal_money"], self.cfg["account_change_rates_per_year"]
        years = self.cfg["months"] / 12
        if self.r() < nm["systematic_withdrawal_rate"]:
            amt = round(h["assets"] * self.rng.uniform(*nm["systematic_annual_pct"]) / 12, -1)
            dom = self.ri(1, 28)
            h["swp"] = amt
            for ms in self.month_starts:
                self.add_mm(h, ms + dom - 1 + 6 / 24, amt, "ach", h["own_bank"])
        if h["age"] >= nm["rmd_age"]:
            for y in range(int(years)):
                d = self.month_starts[min(2 + 12 * y, len(self.month_starts) - 1)] + self.ri(0, 27)
                self.add_mm(h, self.hour(d), round(h["assets"] * 0.5 * nm["rmd_pct"], -1), "ach", h["own_bank"])
        # lumpy withdrawals
        n_new_payees = 0
        for y in range(int(years)):
            for _ in range(self.ri(*nm["lumpy_per_year"])):
                kind, purpose, month = self.pick(T.LUMPY_PURPOSES)
                if month is not None:
                    mi = (month - 10) % 12 + 12 * y
                    day = self.month_starts[min(mi, len(self.month_starts) - 1)] + self.ri(0, 27)
                else:
                    day = 365 * y + self.ri(0, 364)
                day = min(day, self.end_day - 1)
                amt = min(h["assets"] * 0.4, max(500.0, h["assets"] * math.exp(self.rng.normal(math.log(nm["lumpy_median_pct_assets"]), nm["lumpy_sigma"]))))
                amt = round(amt, -1)
                u = self.r()
                if u < nm["lumpy_to_new_business"]:
                    payee = self.add_payee(h, day - self.ri(0, 10), "business", self.pick(["Contractor", "Auto dealer", "Medical center", "Travel agency", "County tax office", "Jeweler"]))
                    method = self.pick(["check", "check", "wire", "ach"])
                    n_new_payees += 1
                elif u < nm["lumpy_to_new_business"] + nm["lumpy_to_new_individual"]:
                    if kind == "gift" and h["fam_person"]:
                        payee = self.add_payee(h, day - self.ri(0, 10), "individual", self.rows_name(h["fam_person"]), person=h["fam_person"])
                    else:
                        payee = self.add_payee(h, day - self.ri(0, 10), "individual", f"{self.pick(T.YOUNG_F + T.YOUNG_M + T.FIRST_M)} {self.pick(T.LAST)}")
                    method = self.pick(["check", "wire", "ach", "ach"])
                    n_new_payees += 1
                else:
                    payee, method = h["own_bank"], self.pick(["ach", "ach", "ach", "wire", "check"])
                hd = self.holding_days(h, self.r() < 0.6) if amt > 10000 and self.r() < nm["sale_of_long_held_rate"] * 2 else None
                t = self.hour(day)
                if kind == "home_repair" and payee != h["own_bank"] and self.r() < nm["renovation_burst_rate"]:
                    parts = self.ri(2, 3)
                    for j in range(parts):
                        dj = min(self.end_day - 1, day + j * self.ri(7, 21))
                        self.add_mm(h, self.hour(dj), round(amt / parts, -1), method, payee, hd if j == 0 else None)
                else:
                    self.add_mm(h, t, amt, method, payee, hd)
                if h["notes_on"] and self.r() < self.cfg["notes"]["withdrawal_note_rate"]:
                    sl = self.slots(h, amt=f"{amt:,.0f}")
                    sl["purpose"] = T.fill(purpose, sl, cap=False)
                    self.add_note(h, t - 0.05, "phone_call", T.fill(self.pick(T.REQUEST_TEMPLATES), sl))
        # other new payees (small)
        for y in range(int(years)):
            for _ in range(max(0, self.ri(*nm["new_payees_per_year"]) - n_new_payees // 2)):
                day = 365 * y + self.ri(0, 364)
                kind = self.pick(["business", "business", "individual"])
                name = self.pick(["Charity", "Plumber", "Landscaper", "Club dues", "Insurance co"]) if kind == "business" else f"{self.pick(T.YOUNG_F + T.YOUNG_M)} {self.pick(T.LAST)}"
                p = self.add_payee(h, day - self.ri(0, 5), kind, name)
                self.add_mm(h, self.hour(day), round(self.rng.uniform(100, 3000), -1), self.pick(["check", "ach"]), p)
        if self.r() < 0.015 * years:  # benign crypto purchase
            day = self.ri(0, self.end_day - 1)
            p = self.add_payee(h, day - self.ri(1, 10), "crypto_exchange", self.pick(["Coinbase", "Kraken", "Gemini"]))
            self.add_mm(h, self.hour(day), round(self.rng.uniform(1000, 15000), -1), "ach", p, None)
        # account changes
        for typ, rate in ac.items():
            if typ == "poa_added_over_80":
                continue
            if typ == "poa_added" and h["age"] > 80:
                rate = ac["poa_added_over_80"]
            for _ in range(int(self.rng.poisson(rate * years))):
                self.normal_event(h, typ, self.ri(0, self.end_day - 1))
        # routine notes
        if h["notes_on"]:
            for y in range(int(years)):
                for _ in range(self.ri(*self.cfg["notes"]["per_year"])):
                    self.routine_note(h, self.hour(365 * y + self.ri(0, 364)))

    def normal_event(self, h, typ, day, tag=None):
        t = self.hour(day, 7, 22)
        if typ == "new_device":
            self.add_ae(h, t, "new_device", self.pick(["web", "mobile"]), device=f"D{int(self.rng.integers(16**8)):08x}", tag=tag)
        elif typ == "bank_link_added":
            self.add_ae(h, t, "bank_link_added", self.pick(["web", "phone", "branch"]), tag=tag)
            p = self.add_payee(h, day, "external_bank", f"{h['first']} {h['last']} ext. bank", tag=tag)
            if self.r() < 0.5:  # legitimately used soon after
                d2 = min(self.end_day - 1, day + self.ri(2, 40))
                self.add_mm(h, self.hour(d2), round(h["assets"] * self.rng.uniform(0.01, 0.08), -1), "ach", p, None, tag=tag)
        elif typ == "poa_added":
            pid = h["fam_person"] or self.add_person(h, f"{self.pick(T.YOUNG_M)} {h['last']}", "son")
            self.add_ae(h, t, "poa_added", "branch", person=pid, tag=tag)
        elif typ == "trusted_contact_changed":
            rel = self.pick(["niece", "nephew", "friend", "daughter", "son"])
            pid = self.add_person(h, f"{self.first_for(rel)} {self.pick(T.LAST)}", rel)
            self.add_ae(h, t, "trusted_contact_changed", self.pick(["phone", "branch"]), person=pid, tag=tag)
        elif typ == "trusted_contact_removed":
            self.add_ae(h, t, "trusted_contact_removed", "phone", tag=tag)
        else:
            self.add_ae(h, t, typ, self.pick(["web", "web", "phone", "mobile"]), tag=tag)

    def routine_note(self, h, t):
        sl = self.slots(h)
        sl["hobby"] = T.fill(self.pick(T.HOBBIES), sl, cap=False)
        sl["extra"] = T.fill(self.pick(T.EXTRAS), sl)
        text = T.fill(self.pick(T.ROUTINE), sl).strip()
        if self.r() < self.cfg["notes"]["benign_risky_mention_rate"]:
            text += " " + T.fill(self.pick(T.BENIGN_RISKY), sl)
        self.add_note(h, t, self.pick(["review_meeting", "phone_call", "phone_call", "email"]), text)

    # ------------------------------------------------------------ exploitation
    def plant_cases(self):
        ex = self.cfg["exploitation"]
        n = int(round(self.N * ex["household_rate"]))
        ages = np.array([h["age"] for h in self.H], dtype=float)
        w = 1 + (ages - 55) / 25
        idx = self.rng.choice(self.N, size=n, replace=False, p=w / w.sum())
        names = list(ex["typologies"])
        shares = np.array([ex["typologies"][t]["share"] for t in names])
        typ = self.rng.choice(len(names), size=n, p=shares / shares.sum())
        self.case_households = set(int(i) for i in idx)
        for c, (i, ti) in enumerate(zip(idx, typ)):
            self.plant_case(c, self.H[int(i)], names[int(ti)])

    def loss(self, h, theft):
        ex = self.cfg["exploitation"]
        med, sig = (ex["loss_theft_median"], ex["loss_theft_sigma"]) if theft else (ex["loss_scam_median"], ex["loss_scam_sigma"])
        return round(min(h["assets"] * ex["max_loss_share_of_assets"], math.exp(self.rng.normal(math.log(med), sig))), -1)

    def flag_note(self, h, t, typology, tag, kind="phone_call", request=None, k=None):
        cats = T.TYPOLOGY_CATEGORIES[typology]
        k = k or self.ri(1, min(3, len(cats)))
        weights = np.linspace(1.6, 0.6, len(cats))
        chosen = list(self.rng.choice(len(cats), size=k, replace=False, p=weights / weights.sum()))
        sl = self.slots(h, **self.case_slots)
        parts = [self.sentence(self.pick(T.RED_FLAGS[cats[j]]), sl) for j in chosen]
        lead = T.fill(self.pick(["{Name} called about the account.", "Call with {Name}.", "{Name} came into the office.",
                                 "Spoke with {Name} by phone.", ""]), sl)
        body = " ".join(([lead] if lead else []) + ([request] if request else []) + parts)
        return self.add_note(h, t, kind, body, tag=tag, cats=[cats[j] for j in chosen])

    def cover_note(self, h, t, tag, request=None):
        sl = self.slots(h, **self.case_slots)
        body = " ".join(([request] if request else []) + [self.sentence(self.pick(T.COVER_STORIES), sl)])
        return self.add_note(h, t, "phone_call", body, tag=tag)

    def request_text(self, h, amt):
        sl = self.slots(h, amt=f"{amt:,.0f}")
        return T.fill(self.pick(["{Name} requested ${amt}.", "Request to send ${amt}.", "{Name} wants to withdraw ${amt}."]), sl)

    def plant_case(self, c, h, typology):
        ex = self.cfg["exploitation"]
        p = ex["typologies"][typology]
        tag = ("case", c)
        theft = typology == "trusted_person_theft"
        L = self.loss(h, theft)
        start = self.ri(ex["first_case_day"], ex["last_case_day"])
        self.case_slots = {"friend": self.pick(T.SCAMMER), "platform": self.pick(T.PLATFORMS)}
        red = p["p_red_flag_notes"] * (1.0 if h["notes_on"] else 0.6)
        cover = self.r() < ex["coached_cover_story_rate"]
        outflows = []  # (day, amount)

        def new_payee(day, crypto_p):
            if self.r() < crypto_p:
                return self.add_payee(h, day, "crypto_exchange", self.pick(["Coinbase", "Kraken", "Crypto.com", "Bitstamp", "Gemini"]), tag=tag), "ach"
            if self.r() < 0.6:
                return self.add_payee(h, day, "individual", f"{self.pick(T.YOUNG_F + T.YOUNG_M + T.FIRST_M)} {self.pick(T.LAST)}", tag=tag), "wire"
            return self.add_payee(h, day, "business", self.pick(["Global Asset Partners LLC", "Secure Holdings Inc", "TD Escrow Services", "Federal Clearing Acct", "Apex Trade Ltd"]), tag=tag), "wire"

        perp_person = None
        if typology == "investment_romance_scam":
            t0 = start + self.ri(7, 45)
            if self.r() < p["p_bank_link"]:
                self.add_ae(h, self.hour(t0 - self.ri(1, 6)), "bank_link_added", "web", tag=tag)
            payee, method = new_payee(t0 - self.ri(0, 4), p["p_crypto"]) if self.r() < p["p_new_payee"] else (h["own_bank"], "ach")
            k = self.ri(2, 5)
            wts = np.cumprod(np.full(k, 1.8)); wts /= wts.sum()
            days = [t0]
            for _ in range(k - 1):
                days.append(days[-1] + self.ri(4, 21))
            amts = list(L * wts)
            if self.r() < p["p_test_transfer"]:
                test = round(self.rng.uniform(300, 2500), -1)
                outflows.append((t0, test, payee, method))
                days = [d + self.ri(3, 10) for d in days]
                amts = [a * (L - test) / L for a in amts]
            outflows += [(d, a, payee, method) for d, a in zip(days, amts)]
            if self.r() < red:
                self.flag_note(h, self.hour(t0 - self.ri(2, 30)), typology, tag)
            sell = p["p_sell_long_held"]
        elif typology == "tech_support_impersonation":
            t0 = start + self.ri(0, 3)
            if self.r() < p["p_password_reset"]:
                self.add_ae(h, self.hour(start, 9, 20), "password_reset", "phone", tag=tag)
            if self.r() < p["p_new_device"]:
                self.add_ae(h, self.hour(start, 9, 20), "new_device", "web", device=f"D{int(self.rng.integers(16**8)):08x}", tag=tag)
            payee, method = new_payee(t0, p["p_crypto"]) if self.r() < p["p_new_payee"] else (h["own_bank"], "wire")
            k = self.ri(1, 2)
            outflows += [(t0 + j * self.ri(1, 4), L / k, payee, method) for j in range(k)]
            sell = p["p_sell_long_held"]
        elif typology == "government_impersonation":
            t0 = start + self.ri(0, 3)
            payee, method = new_payee(t0, p["p_crypto"]) if self.r() < p["p_new_payee"] else (h["own_bank"], self.pick(["wire", "check"]))
            outflows.append((t0, L, payee, method))
            sell = p["p_sell_long_held"]
        elif typology == "silent_account_takeover":
            d = start
            dev = f"D{int(self.rng.integers(16**8)):08x}"
            if self.r() < p["p_new_device"]:
                self.add_ae(h, self.hour(d, 0, 24), "new_device", "web", device=dev, tag=tag)
            if self.r() < p["p_email_change"]:
                self.add_ae(h, self.hour(d, 0, 24), "email_change", "web", device=dev, tag=tag)
            d += self.ri(0, 3)
            if self.r() < p["p_phone_change"]:
                self.add_ae(h, self.hour(d, 0, 24), "phone_change", "web", device=dev, tag=tag)
            d += self.ri(0, 4)
            if self.r() < p["p_bank_link"]:
                self.add_ae(h, self.hour(d, 0, 24), "bank_link_added", "web", device=dev, tag=tag)
                payee = self.add_payee(h, d, "external_bank", f"{h['first']} {h['last']} ext. bank", tag=tag)
            else:
                payee = new_payee(d, 0.2)[0]
            t0 = d + self.ri(1, 4)
            k = self.ri(1, 3)
            outflows += [(t0 + j * self.ri(0, 2), L / k, payee, self.pick(["ach", "wire"])) for j in range(k)]
            if self.r() < red:
                sl = self.slots(h)
                self.add_note(h, self.hour(t0 + self.ri(-1, 2)), "service_request", T.fill(self.pick(T.ATO_NOTES), sl),
                              tag=tag, cats=[T.PURPOSE])
            red = 0
            sell = p["p_sell_long_held"]
        elif typology == "grandparent_voice_clone":
            t0 = start + self.ri(0, 2)
            payee, method = new_payee(t0, p["p_crypto"]) if self.r() < p["p_new_payee"] else (h["own_bank"], "wire")
            k = 1 if self.r() < 0.75 else 2
            outflows += [(t0 + j * self.ri(1, 3), L / k, payee, method) for j in range(k)]
            sell = p["p_sell_long_held"]
        else:  # trusted_person_theft
            rel = self.pick(["son", "daughter", "nephew", "caregiver", "grandson", "neighbor"])
            existing = [kd for kd in h["kids"]] if rel in ("son", "daughter") else []
            if existing and self.r() < 0.6:
                perp_person, pname, rel = existing[0]
            else:
                pname = f"{self.first_for(rel)} {h['last'] if rel in ('son', 'daughter', 'nephew', 'grandson') else self.pick(T.LAST)}"
                perp_person = self.add_person(h, pname, rel)
            self.case_slots["perp"] = f"{P_rel(h, rel)} {pname.split()[0]}"
            if self.r() < p["p_poa_added"]:
                self.add_ae(h, self.hour(start), "poa_added", "branch", person=perp_person, tag=tag)
            elif self.r() < p["p_joint_owner_added"] / (1 - p["p_poa_added"]):
                self.add_ae(h, self.hour(start), "joint_owner_added", "branch", person=perp_person, tag=tag)
            if self.r() < p["p_tc_changed"]:
                if h["tc"] and self.r() < 0.5:
                    self.add_ae(h, self.hour(start + self.ri(1, 20)), "trusted_contact_removed", "phone", tag=tag)
                else:
                    self.add_ae(h, self.hour(start + self.ri(1, 20)), "trusted_contact_changed", "phone", person=perp_person, tag=tag)
            payee = self.add_payee(h, start + self.ri(2, 20), "individual", pname, person=perp_person, tag=tag)
            struct = self.r() < p["p_structuring"]
            per = self.rng.uniform(8000, 9900) if struct else self.rng.uniform(1500, 7000)
            k = max(2, int(math.ceil(L / per)))
            t0 = start + self.ri(5, 30)
            gap = max(3, min(30, (self.end_day - 5 - t0) // max(1, k)))
            gaps = self.rng.integers(max(2, gap // 2), gap + 1, size=k)
            days = t0 + np.concatenate([[0], np.cumsum(gaps[:-1])])
            amts = self.rng.dirichlet(np.ones(k) * 20) * L
            if struct:
                amts = np.minimum(amts, 9900)
            outflows += [(int(d), float(a), payee, self.pick(["ach", "ach", "check", "wire"])) for d, a in zip(days, amts) if d < self.end_day - 1]
            sell = 0.15
            if self.r() < red:
                for _ in range(self.ri(1, 2)):
                    self.flag_note(h, self.hour(t0 + self.ri(-10, 60)), typology, tag)
            red = 0
            t0 = outflows[0][0] if outflows else t0

        outflows = sorted([(d, a, pay, m) for d, a, pay, m in outflows if d < self.end_day - 1], key=lambda o: o[0])
        if not outflows:
            return
        t0 = outflows[0][0]
        txn_keys, txn_times = [], []
        for j, (d, a, pay, m) in enumerate(outflows):
            hd = self.holding_days(h, True) if (a > 5000 and self.r() < sell) else None
            tt = self.hour(d)
            txn_times.append(tt)
            txn_keys.append(self.add_mm(h, tt, round(a, -1) if a > 100 else round(a, 2), m, pay, hd, tag=tag))
        # the advisor's note about the call that prompted the largest request, ~30 minutes before it
        if typology in ("tech_support_impersonation", "government_impersonation", "grandparent_voice_clone", "investment_romance_scam"):
            ib = max(range(len(outflows)), key=lambda j: outflows[j][1])
            big = outflows[ib]
            req = self.request_text(h, round(big[1], -1))
            if cover:
                self.hour(big[0])  # unused draw: keeps the random stream (households, cases, amounts) unchanged
                self.cover_note(h, txn_times[ib] - 0.02, tag, request=req)
            elif self.r() < red:
                self.hour(big[0])  # unused draw, as above
                self.flag_note(h, txn_times[ib] - 0.02, typology, tag, request=req)
        self.rows["cases"].append(dict(_case=c, household_id=h["id"], typology=typology,
                                       perpetrator_type=p["perpetrator"], _start=start, _first_out=t0,
                                       total_loss=round(sum(o[1] for o in outflows), 2), coached_cover_story=cover))

    # ------------------------------------------------------------ look-alikes
    def plant_lookalikes(self):
        lk = self.cfg["lookalikes"]
        n = int(round(self.N * lk["household_rate_per_year"] * self.cfg["months"] / 12))
        pool = [i for i in range(self.N) if i not in self.case_households]
        idx = self.rng.choice(pool, size=n, replace=False)
        kinds = list(lk["kinds"])
        w = np.array([lk["kinds"][k] for k in kinds])
        ks = self.rng.choice(len(kinds), size=n, p=w / w.sum())
        for e, (i, ki) in enumerate(zip(idx, ks)):
            self.plant_lookalike(e, self.H[int(i)], kinds[int(ki)])

    def plant_lookalike(self, e, h, kind):
        tag = ("neg", e)
        A = h["assets"]
        d = self.ri(30, self.end_day - 20)
        start = d
        p_note = self.cfg["lookalikes"]["p_explaining_note"]
        if kind == "home_purchase":
            pay = self.add_payee(h, d - self.ri(1, 7), "business", self.pick(["First American Title", "Stewart Title Escrow", "Fidelity National Title"]), tag=tag)
            self.add_mm(h, self.hour(d), round(min(0.7 * A, max(60000, A * self.rng.uniform(0.2, 0.6))), -2), "wire", pay, self.holding_days(h, self.r() < 0.6), tag=tag)
            if self.r() < 0.6:
                self.normal_event(h, "address_change", min(self.end_day - 1, d + self.ri(10, 60)), tag=tag)
        elif kind == "gift_to_family":
            pid = h["fam_person"] or self.add_person(h, f"{self.pick(T.YOUNG_M)} {h['last']}", "son")
            pay = self.add_payee(h, d - self.ri(0, 5), "individual", self.rows_name(pid), person=pid, tag=tag)
            for j in range(self.ri(1, 2)):
                self.add_mm(h, self.hour(d + 30 * j), round(min(0.3 * A, self.rng.uniform(10000, 60000)), -2), self.pick(["wire", "check", "ach"]), pay, None, tag=tag)
        elif kind == "long_term_care_deposit":
            if self.r() < 0.4 and h["fam_person"]:
                self.add_ae(h, self.hour(d - self.ri(10, 40)), "poa_added", "branch", person=h["fam_person"], tag=tag)
            pay = self.add_payee(h, d - self.ri(1, 7), "business", self.pick(["Sunrise Senior Living", "Brookdale", "Oak Grove Memory Care"]), tag=tag)
            self.add_mm(h, self.hour(d), round(min(0.5 * A, self.rng.uniform(20000, 120000)), -2), self.pick(["wire", "check"]), pay, self.holding_days(h, self.r() < 0.5), tag=tag)
        elif kind == "car_purchase":
            pay = self.add_payee(h, d - self.ri(1, 7), "business", self.pick(["Toyota of Springfield", "Honda Motors", "Lexus dealer"]), tag=tag)
            self.add_mm(h, self.hour(d), round(min(0.4 * A, self.rng.uniform(25000, 75000)), -2), self.pick(["wire", "check"]), pay, None, tag=tag)
        elif kind == "large_tax_payment":
            d = self.month_starts[6 + (12 if self.r() < 0.5 else 0)] + self.ri(0, 14)
            start = d
            pay = self.add_payee(h, d - 2, "business", "US Treasury", tag=tag)
            self.add_mm(h, self.hour(d), round(min(0.4 * A, self.rng.uniform(20000, 150000)), -2), "ach", pay, self.holding_days(h, self.r() < 0.5), tag=tag)
        elif kind == "remarriage":
            sp = self.add_person(h, f"{self.pick(T.FIRST_M if h['gender'] == 'F' else T.FIRST_F)} {self.pick(T.LAST)}", "spouse")
            self.add_ae(h, self.hour(d), "joint_owner_added", "branch", person=sp, tag=tag)
            if self.r() < 0.5:
                self.add_ae(h, self.hour(d + self.ri(1, 10)), "trusted_contact_changed", "branch", person=sp, tag=tag)
            pay = self.add_payee(h, d + 3, "individual", self.rows_name(sp), person=sp, tag=tag)
            for j in range(self.ri(2, 4)):
                self.add_mm(h, self.hour(min(self.end_day - 1, d + 7 + 20 * j)), round(min(0.1 * A, self.rng.uniform(5000, 30000)), -2), "ach", pay, None, tag=tag)
        elif kind == "moving_states":
            self.normal_event(h, "address_change", d, tag=tag)
            if self.r() < 0.7:
                self.normal_event(h, "phone_change", d + self.ri(0, 3), tag=tag)
            if self.r() < 0.3:
                self.add_ae(h, self.hour(d + self.ri(0, 3)), "email_change", "web", tag=tag)
            if self.r() < 0.5:
                self.add_ae(h, self.hour(d + self.ri(0, 3)), "new_device", "web", device=f"D{int(self.rng.integers(16**8)):08x}", tag=tag)
            d2 = d + self.ri(1, 6)
            self.add_ae(h, self.hour(d2), "bank_link_added", "web", tag=tag)
            pay = self.add_payee(h, d2, "external_bank", f"{h['first']} {h['last']} new bank", tag=tag)
            self.add_mm(h, self.hour(min(self.end_day - 1, d2 + self.ri(2, 20))), round(A * self.rng.uniform(0.05, 0.35), -2), self.pick(["ach", "wire"]), pay, self.holding_days(h, self.r() < 0.4), tag=tag)
        elif kind == "transfer_to_other_firm":
            pay = self.add_payee(h, d - 1, "business", self.pick(["Schwab", "Fidelity", "Vanguard", "Edward Jones"]), tag=tag)
            self.add_mm(h, self.hour(d), round(A * self.rng.uniform(0.4, 1.0), -2), "acat", pay, self.holding_days(h, True), tag=tag)
        elif kind == "trust_funding":
            if h["fam_person"] and self.r() < 0.6:
                self.add_ae(h, self.hour(d - self.ri(5, 30)), "poa_added", "branch", person=h["fam_person"], tag=tag)
            pay = self.add_payee(h, d - 2, "business", f"{h['last']} Family Revocable Trust", tag=tag)
            self.add_mm(h, self.hour(d), round(A * self.rng.uniform(0.2, 0.6), -2), "journal", pay, self.holding_days(h, True), tag=tag)
        if h["notes_on"] and self.r() < p_note:
            sl = self.slots(h)
            # explaining notes come just before the episode's first record on or after its start day
            later = [t for t in self.tag_times.get(tag, []) if t >= start]
            anchor = min(later) if later else None
            t1 = self.hour(start)  # drawn either way so the random stream is unchanged
            self.add_note(h, (anchor if anchor is not None else t1) - 0.03, "phone_call", T.fill(self.pick(T.LOOKALIKE_NOTES[kind]), sl), tag=tag)
            if self.r() < 0.3:
                t2 = self.hour(start)
                self.add_note(h, (anchor if anchor is not None else t2) - 0.02, "phone_call", T.fill(self.pick(T.MITIGATING), sl), tag=tag)
        self.rows["negatives"].append(dict(_ep=e, household_id=h["id"], kind=kind, _start=start))

    def rows_name(self, pid):
        return self.person_names.get(pid, "Family member")

    # ------------------------------------------------------------ destination accounts (mule network)
    def dest_refs(self, pay: pd.DataFrame) -> list[str]:
        """Receiving account / wallet behind each payee. Uses its own random stream, so every other
        record matches the world without this column. Mule refs are not marked; only labels reveal them."""
        rng = np.random.default_rng(self.seed + 9001)
        mu = self.cfg["exploitation"]["mule"]
        shared_n = self.cfg["normal_money"].get("shared_destinations", {})
        scale = self.N / 20000
        slots = {t: [None] * max(1, round(n * scale)) for t, n in mu["pool_size"].items()}
        stranger = {c["_case"] for c in self.rows["cases"] if c["perpetrator_type"] == "stranger"}
        used, shared, person = set(), {}, {}

        def new(prefix):
            while True:
                r = f"{prefix}-{int(rng.integers(16 ** 8)):08x}"
                if r not in used:
                    used.add(r)
                    return r

        out = []
        for hid, kind, name, pid, day, tag in zip(pay["household_id"], pay["kind"], pay["display_name"], pay["person_id"],
                                                  pay["_day"], pay["_tag"]):
            prefix = "WAL" if kind == "crypto_exchange" else "ACCT"
            if isinstance(tag, tuple) and tag[0] == "case" and tag[1] in stranger and kind != "own_bank"                     and rng.random() < mu["share_of_stranger_scams"]:
                pool = slots["wallet" if prefix == "WAL" else "account"]
                j = int(rng.integers(len(pool)))
                if pool[j] is None or day >= pool[j][1]:
                    pool[j] = (new(prefix), day + mu["pool_lifetime_days"])
                out.append(pool[j][0])
            elif name in shared_n and kind == "business":
                refs = shared.setdefault(name, [])
                j = int(rng.integers(shared_n[name]))
                while len(refs) <= j:
                    refs.append(new(prefix))
                out.append(refs[j])
            elif pid is not None and not (isinstance(pid, float) and math.isnan(pid)):
                out.append(person.setdefault((hid, pid), new(prefix)))
            else:
                out.append(new(prefix))
        return out

    # ------------------------------------------------------------ finalize
    def ts(self, days):
        return (self.start + pd.to_timedelta(np.asarray(days, dtype=float) * 86400, unit="s")).floor("min")

    def finalize(self):
        R = self.rows
        pay = pd.DataFrame(R["payees"]).sort_values(["_day", "household_id", "_k"], kind="stable").reset_index(drop=True)
        pay["payee_id"] = [f"PY-{i:06d}" for i in range(len(pay))]
        pay["created_ts"] = self.ts(pay["_day"]).strftime("%Y-%m-%d %H:%M")
        pay["dest_ref"] = self.dest_refs(pay)
        pmap = dict(zip(pay["_k"], pay["payee_id"]))
        new_payee = dict(zip(pay["_k"], pay["_day"] >= 0))

        out = {}
        for name, prefix in [("money_movements", "TX"), ("account_events", "EV"), ("notes", "N")]:
            df = pd.DataFrame(R[name])
            df = df[df["_day"] < self.end_day].copy()
            df["ts_dt"] = self.ts(df["_day"])
            df = df.sort_values(["ts_dt", "household_id", "_k"], kind="stable").reset_index(drop=True)
            idcol = {"money_movements": "txn_id", "account_events": "event_id", "notes": "note_id"}[name]
            df[idcol] = [f"{prefix}-{i:07d}" for i in range(len(df))]
            df["ts"] = df["ts_dt"].dt.strftime("%Y-%m-%d %H:%M")
            out[name] = df
        mm = out["money_movements"]
        mm["payee_id"] = mm["_payee"].map(pmap)
        first = ~mm.duplicated("_payee")
        mm["payee_is_new"] = first & mm["_payee"].map(new_payee).astype(bool)
        mm["source_holding_days"] = mm["source_holding_days"].astype("Int64")

        # labels
        key_to = {}
        for name, idcol in [("money_movements", "txn_id"), ("account_events", "event_id"), ("notes", "note_id")]:
            df = out[name]
            for tag, rid in zip(df["_tag"], df[idcol]):
                if tag is not None:
                    key_to.setdefault(tag, []).append(rid)
        cases = pd.DataFrame(R["cases"])
        cases["case_id"] = [f"CASE-{i:04d}" for i in range(len(cases))]
        # a case starts at its planned start or its earliest record (a red-flag note can come first)
        first_rec = {}
        for name in ("money_movements", "account_events", "notes"):
            df = out[name]
            for tag, d in zip(df["_tag"], df["_day"]):
                if tag is not None and tag[0] == "case":
                    first_rec[tag[1]] = min(first_rec.get(tag[1], np.inf), d)
        cases["_start"] = [min(s, first_rec.get(c, np.inf)) for c, s in zip(cases["_case"], cases["_start"])]
        cases["start_ts"] = self.ts(cases["_start"]).strftime("%Y-%m-%d %H:%M")
        case_tx = mm[mm["_tag"].notna()].copy()
        case_tx["_tagt"] = case_tx["_tag"]
        first_out = {}
        last_out = {}
        for tag, g in case_tx.groupby(case_tx["_tagt"].astype(str)):
            first_out[tag] = g["ts"].min()
            last_out[tag] = g["ts"].max()
        cases["first_outflow_ts"] = [first_out.get(str(("case", c))) for c in cases["_case"]]
        cases["last_outflow_ts"] = [last_out.get(str(("case", c))) for c in cases["_case"]]
        cases["related_record_ids"] = [";".join(key_to.get(("case", c), [])) for c in cases["_case"]]
        loss = case_tx[case_tx["_tag"].astype(str).str.startswith("('case'")].groupby(case_tx["_tag"].astype(str))["amount"].sum()
        cases["total_loss"] = [round(float(loss.get(str(("case", c)), 0.0)), 2) for c in cases["_case"]]
        cases = cases[cases["first_outflow_ts"].notna()]
        neg = pd.DataFrame(R["negatives"])
        neg["episode_id"] = [f"NEG-{i:04d}" for i in range(len(neg))]
        neg["start_ts"] = self.ts(neg["_start"]).strftime("%Y-%m-%d %H:%M")
        neg["related_record_ids"] = [";".join(key_to.get(("neg", e), [])) for e in neg["_ep"]]
        # note-level truth for reader evaluation (labels dir only)
        nt = out["notes"][["note_id", "household_id", "_cats", "_inj"]].rename(columns={"_cats": "planted_categories", "_inj": "has_injection"})

        files = {
            "advisors": pd.DataFrame(R["advisors"])[["advisor_id", "name", "style", "region"]],
            "households": pd.DataFrame(R["households"]),
            "persons": pd.DataFrame(R["persons"]),
            "accounts": pd.DataFrame(R["accounts"]),
            "positions": pd.DataFrame(R["positions"]),
            "payees": pay[["payee_id", "household_id", "kind", "display_name", "person_id", "created_ts", "dest_ref"]],
            "money_movements": mm[["txn_id", "account_id", "household_id", "ts", "amount", "method", "payee_id",
                                   "payee_is_new", "source_holding_days", "status"]],
            "account_events": out["account_events"][["event_id", "household_id", "ts", "type", "channel", "device_id", "related_person_id"]],
            "notes": out["notes"][["note_id", "household_id", "advisor_id", "ts", "kind", "text"]],
        }
        labels = {
            "cases": cases[["case_id", "household_id", "typology", "perpetrator_type", "start_ts", "first_outflow_ts",
                            "last_outflow_ts", "total_loss", "coached_cover_story", "related_record_ids"]],
            "negatives": neg[["episode_id", "household_id", "kind", "start_ts", "related_record_ids"]],
            "note_truth": nt,
        }
        # positions: add id
        files["positions"].insert(0, "position_id", [f"POS-{i:06d}" for i in range(len(files["positions"]))])
        return files, labels


def P_rel(h, rel):
    return {"caregiver": "caregiver", "neighbor": "neighbor"}.get(rel, ("her " if h["gender"] == "F" else "his ") + rel)


def write_world(out_dir="data/world", config="config/world.yaml", seed=None, households=None) -> dict:
    cfg = load_config(config)
    files, labels = World(cfg, seed=seed, households=households).build()
    out = Path(out_dir)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for k, df in files.items():
        df.to_csv(out / f"{k}.csv", index=False, lineterminator="\n")
    for k, df in labels.items():
        df.to_csv(out / "labels" / f"{k}.csv", index=False, lineterminator="\n")
    summary = summarize(files, labels)
    (out / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize(files, labels) -> dict:
    hh = files["households"]
    cases, neg = labels["cases"], labels["negatives"]
    scam = cases[cases.perpetrator_type == "stranger"].total_loss
    theft = cases[cases.perpetrator_type == "trusted"].total_loss
    return {
        "rows": {k: int(len(v)) for k, v in files.items()},
        "households": int(len(hh)),
        "case_rate": round(len(cases) / len(hh), 4),
        "cases": int(len(cases)),
        "typology_counts": cases.typology.value_counts().to_dict(),
        "stranger_share": round(float((cases.perpetrator_type == "stranger").mean()), 3),
        "loss_median_scam": float(scam.median()),
        "loss_median_theft": float(theft.median()),
        "share_cases_over_100k": round(float((cases.total_loss > 100000).mean()), 3),
        "lookalike_episodes": int(len(neg)),
        "households_without_notes": round(1 - files["notes"].household_id.nunique() / len(hh), 3),
        "notes_with_injection": int(labels["note_truth"].has_injection.sum()),
        "assets_median": float(hh.assets.median()),
    }
