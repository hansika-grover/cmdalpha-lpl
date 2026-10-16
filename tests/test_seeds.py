import math

from sentinel.eval.run_eval import TYPOLOGIES
from sentinel.eval.seeds import KEYS, METHODS, summarize


def fake_run(recall, typ_recall, as_dict=False):
    def node():
        bt = {t: {"recall": ({"value": r} if as_dict else r), "n": 10} for t, r in typ_recall.items()}
        return {**{k: {"value": recall, "ci95": None} for k in KEYS}, "by_typology": bt}
    return {"methods": {m: ({"natural": node()} if b == ("natural",) else {"budgets": {x: node() for x in b}})
                        for m, b in METHODS.items()}}


def test_by_typology_aggregation():
    t0, t1 = TYPOLOGIES[:2]
    runs = {0: fake_run(0.5, {t0: 0.2, t1: None}), 1: fake_run(0.7, {t0: 0.4}, as_dict=True)}
    S = summarize(runs)
    assert S["seeds"] == [0, 1]
    assert S["methods"]["sentinel"]["1%"]["recall"]["mean"] == 0.6
    for m, budgets in METHODS.items():
        assert set(S["by_typology"][m]) == set(budgets)
    a = S["by_typology"]["sentinel"]["0.25%"]
    assert set(a) == set(TYPOLOGIES)
    assert a[t0]["mean"] == 0.3 and a[t0]["min"] == 0.2 and a[t0]["max"] == 0.4 and math.isclose(a[t0]["sd"], 0.1)
    assert a[t0]["per_seed"] == [0.2, 0.4] and a[t0]["n"] == [10, 10]
    assert a[t1]["mean"] is None and a[t1]["per_seed"] == [None, None] and a[t1]["n"] == [10, 0]
    assert S["by_typology"]["threshold_100k"]["natural"][t0]["mean"] == 0.3
