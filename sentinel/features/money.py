"""Engine A: money movement against the household's own baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Stream, keyed

LONG_HELD_DAYS = 5 * 365
ASSET_BANDS = [0, 100e3, 250e3, 500e3, 1e6, 3e6, np.inf]
PEER_DAYS = 90  # peer baselines use only the first 90 days, before any planted case starts


def enrich_txns(W) -> pd.DataFrame:
    """Per-transaction attributes known at request time (payee kind, payee age, prior baseline)."""
    mm = W["money_movements"].merge(W["payees"][["payee_id", "kind", "person_id", "created_ts"]], on="payee_id", how="left")
    hh = W["households"].set_index("household_id")
    mm["assets"] = hh.loc[mm["household_id"], "assets"].to_numpy()
    mm["band"] = pd.cut(mm["assets"], ASSET_BANDS, labels=False)
    mm["payee_age"] = (mm["ts"] - mm["created_ts"]).dt.total_seconds() / 86400
    mm["young_payee"] = (mm["payee_age"] < 60) & (mm["kind"] != "own_bank")
    mm["key"] = keyed(mm, W["hh_index"])
    mm = mm.sort_values("key", kind="stable").reset_index(drop=True)

    # household baseline over the prior 12 months (strictly before this request)
    g = mm.set_index("ts").groupby("household_id", sort=False)["amount"]
    roll = g.rolling("365D", closed="left")
    mm["base_med"] = roll.median().to_numpy()
    mm["base_q25"] = roll.quantile(0.25).to_numpy()
    mm["base_q75"] = roll.quantile(0.75).to_numpy()
    mm["base_n"] = roll.count().to_numpy()
    mm["prior_max"] = mm.groupby("household_id", sort=False)["amount"].transform(lambda s: s.shift().cummax())
    mm["repeat_n"] = mm.groupby(["household_id", "payee_id", "amount"], sort=False).cumcount()

    # peer baseline by asset band (age is never used)
    early = mm[mm["day"] < PEER_DAYS]
    peer = early.groupby("band")["amount"].agg(p_med="median", p_q25=lambda s: s.quantile(.25),
                                                 p_q75=lambda s: s.quantile(.75), p_max=lambda s: s.quantile(.95))
    mm = mm.join(peer, on="band")
    thin = mm["base_n"].fillna(0) < 3
    med = np.where(thin, mm["p_med"], mm["base_med"])
    iqr = np.where(thin, mm["p_q75"] - mm["p_q25"], mm["base_q75"] - mm["base_q25"])
    scale = np.maximum.reduce([iqr / 1.349, 0.1 * med, np.full(len(mm), 50.0)])
    mm["robust_z"] = np.clip((mm["amount"] - med) / scale, -5, 50)
    pmax = np.where(mm["prior_max"].isna(), mm["p_max"], np.maximum(mm["prior_max"], 1.0))
    mm["ratio_max"] = np.clip(mm["amount"] / pmax, 0, 100)
    mm["thin_history"] = thin.astype(int)
    return mm


TXN_FEATURES = ["is_outflow", "amt_log", "robust_z", "ratio_max", "share_assets", "payee_new", "young_payee",
                "payee_crypto", "payee_individual", "payee_ext_bank", "method_wire", "method_check",
                "method_acat", "method_journal", "sold_long_held", "recurring", "thin_history"]
WINDOW_FEATURES = ["out_cnt_7d", "out_sum_7d", "out_cnt_30d", "out_sum_30d", "out_30d_vs_typical",
                   "young_payee_sum_30d", "crypto_sum_30d", "individual_sum_30d", "nonrecurring_sum_30d",
                   "struct_cnt_60d", "share_out_30d"]
FEATURES = TXN_FEATURES + WINDOW_FEATURES


def money_features(W, P: pd.DataFrame, mm: pd.DataFrame) -> pd.DataFrame:
    F = pd.DataFrame(index=P.index)
    cur = P[["ref_id"]].merge(mm.set_index("txn_id"), left_on="ref_id", right_index=True, how="left")
    is_mm = (P["kind"] == "mm").to_numpy()
    F["is_outflow"] = is_mm.astype(int)
    a = cur["amount"].fillna(0).to_numpy()
    F["amt_log"] = np.log1p(a)
    F["robust_z"] = cur["robust_z"].fillna(0).to_numpy()
    F["ratio_max"] = cur["ratio_max"].fillna(0).to_numpy()
    F["share_assets"] = (a / cur["assets"].fillna(1)).to_numpy()
    F["payee_new"] = cur["payee_is_new"].fillna(False).astype(int).to_numpy()
    F["young_payee"] = cur["young_payee"].fillna(False).astype(int).to_numpy()
    for k, col in [("crypto_exchange", "payee_crypto"), ("individual", "payee_individual"), ("external_bank", "payee_ext_bank")]:
        F[col] = (cur["kind"] == k).astype(int).to_numpy()
    for m in ("wire", "check", "acat", "journal"):
        F[f"method_{m}"] = (cur["method"] == m).astype(int).to_numpy()
    F["sold_long_held"] = (cur["source_holding_days"].fillna(0) >= LONG_HELD_DAYS).astype(int).to_numpy()
    F["recurring"] = (cur["repeat_n"].fillna(0) >= 2).astype(int).to_numpy()
    F["thin_history"] = cur["thin_history"].fillna(0).to_numpy()

    q = P["key"].to_numpy()
    amt = mm["amount"].to_numpy()
    k = mm["key"].to_numpy()
    s_cnt, s_amt = Stream(k), Stream(k, amt)
    F["out_cnt_7d"] = s_cnt.window(q, 7)
    F["out_sum_7d"] = np.log1p(s_amt.window(q, 7))
    F["out_cnt_30d"] = s_cnt.window(q, 30)
    sum30 = s_amt.window(q, 30)
    F["out_sum_30d"] = np.log1p(sum30)
    typical = s_amt.window(q - 30, 335) / 11  # monthly outflow over the prior 11 months
    F["out_30d_vs_typical"] = np.clip(sum30 / np.maximum(typical, 500.0), 0, 200)
    for col, mask in [("young_payee_sum_30d", mm["young_payee"]), ("crypto_sum_30d", mm["kind"] == "crypto_exchange"),
                      ("individual_sum_30d", mm["kind"] == "individual"), ("nonrecurring_sum_30d", mm["repeat_n"] < 2)]:
        F[col] = np.log1p(Stream(k, amt * mask.to_numpy()).window(q, 30))
    F["struct_cnt_60d"] = Stream(k, ((amt >= 8000) & (amt < 10000)).astype(float)).window(q, 60)
    assets = W["households"].set_index("household_id").loc[P["household_id"], "assets"].to_numpy()
    F["share_out_30d"] = np.clip(sum30 / assets, 0, 5)
    return F
