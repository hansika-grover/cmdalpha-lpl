"""Ablation baseline notes reader: a compliance-style keyword lexicon plus a TF-IDF risk model.

This is what a team could build without an LLM. It shows what the Claude reader adds.
Quotes are the sentence containing the match, so they are always exact substrings.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..features.notes import FLAG_COLS

LEXICON = [
    r"met (?:him|her|them)?\s?(?:on|through|online)|online friend|boyfriend|girlfriend|dating|facebook|whatsapp|instagram|never met",
    r"urgent|asap|right away|immediately|deadline|as soon as possible|same[- ]day|expires|quickest|before friday|go out today|happen today",
    r"don'?t tell|not to tell|not to mention|between us|secret|private|not want .{0,30} to know|not to discuss",
    r"on the (?:other )?line|in the background|on (?:his|her) behalf|answered for|most of the talking|on speaker|script",
    r"could not explain|couldn'?t explain|vague|not sure who|explain later|no details|did not recognize|different reasons|could not tell me",
    r"confused|did not remember|forgot|repeated|unsure|could not recall|trouble recalling|lost track",
    r"arrest|warrant|jail|bail|hacked|compromised|virus|threat|accident|safe account|investigated",
    r"crypto|bitcoin|gift card|courier|overseas|usdt|ethereum|kraken|coinbase|wallet",
]
MITIGATING = r"fully aware|confirmed|verified|attorney|invoice|purchase agreement|planning this|cpa|trust documents|annual review"
INJECTION = r"ignore (?:all )?(?:prior|previous) instructions|automated reviewer|ai (?:assistant|system)|admin override|system note|assistant:|do not flag"
_SENT = re.compile(r"[^.;!?]+[.;!?]?")


def _quote(text: str, m: re.Match) -> str:
    for s in _SENT.finditer(text):
        if s.start() <= m.start() < s.end():
            return s.group(0).strip()
    return m.group(0)


def keyword_read(notes: pd.DataFrame) -> pd.DataFrame:
    pats = [re.compile(p, re.I) for p in LEXICON]
    mit, inj = re.compile(MITIGATING, re.I), re.compile(INJECTION, re.I)
    rows = []
    for nid, text in zip(notes["note_id"], notes["text"].astype(str)):
        r = {"note_id": nid}
        # injection sentences are stripped before matching, so they can never add a flag
        clean = inj.sub("", text) if inj.search(text) else text
        quotes = []
        for i, p in enumerate(pats):
            m = p.search(clean)
            r[FLAG_COLS[i]] = int(m is not None)
            if m:
                quotes.append((i, _quote(clean, m)))
        mm, im = mit.search(clean), inj.search(text)
        r["mitigating"] = int(mm is not None)
        r["injection"] = int(im is not None)
        r["quotes"] = quotes
        r["mit_quotes"] = [_quote(clean, mm)] if mm else []
        r["inj_quotes"] = [_quote(text, im)] if im else []
        rows.append(r)
    return pd.DataFrame(rows)


def tfidf_risk(notes: pd.DataFrame, train_notes_y: pd.Series) -> np.ndarray:
    """Note-level risk from TF-IDF + logistic regression, trained only on training-split notes."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
    tr = train_notes_y.index.to_numpy()
    vec.fit(notes["text"].astype(str).to_numpy()[tr])  # vocabulary from training notes only
    X = vec.transform(notes["text"].astype(str))
    clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000)
    clf.fit(X[tr], train_notes_y.to_numpy())
    return clf.predict_proba(X)[:, 1]
