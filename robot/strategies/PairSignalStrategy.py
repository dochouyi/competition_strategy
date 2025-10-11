from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime
import pandas as pd
import logging
import os
from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)



@dataclass
class PairSelectorConfig:
    timeframe: str = "5m"
    form_period: int = 200
    bb_window: int = 30          # 仅用于 beta 估计最小窗长限制
    min_form_bars: int = 60
    recompute_every: int = 20
    min_corr: float = 0.2
    use_log_price: bool = False
    select_pairs_per_window: int = 1
    max_candidates_per_method: int = 20

def estimate_beta_on_window(a: pd.Series, b: pd.Series, use_log_price=True, min_len: int = 30) -> float:
    a = np.log(a.dropna()) if use_log_price else a.dropna().copy()
    b = np.log(b.dropna()) if use_log_price else b.dropna().copy()
    idx = a.index.intersection(b.index)
    a = a.loc[idx]; b = b.loc[idx]
    if len(a) < min_len:
        return 1.0
    cov = np.cov(b.values, a.values)[0, 1]
    var = np.var(b.values)
    beta = cov / var if var > 0 else 1.0
    if not np.isfinite(beta):
        beta = 1.0
    return float(np.clip(beta, 0.1, 10.0))

def unified_scaled_distance(a: pd.Series, b: pd.Series) -> float:
    idx = a.dropna().index.intersection(b.dropna().index)
    if len(idx) == 0:
        return np.inf
    A = a.loc[idx].values.reshape(-1, 1)
    B = b.loc[idx].values.reshape(-1, 1)
    X = np.vstack([A, B])
    mu = X.mean()
    sd = X.std() if X.std() > 0 else 1.0
    As = (A - mu) / sd
    Bs = (B - mu) / sd
    return float(np.linalg.norm(As.flatten() - Bs.flatten()))

def euclidean_distance_method(prices: Dict[str, pd.Series], cfg: PairSelectorConfig, topk: int) -> List[Tuple[str, str, float]]:
    scores = []
    keys = list(prices.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            d = unified_scaled_distance(prices[a], prices[b])
            if not np.isfinite(d):
                continue
            scores.append((a, b, d))
    scores.sort(key=lambda x: x[2])
    pairs = []
    for a, b, _ in scores[:topk]:
        beta = estimate_beta_on_window(prices[a], prices[b], use_log_price=cfg.use_log_price, min_len=max(30, cfg.bb_window))
        pairs.append((a, b, beta))
    return pairs



class PairSignalStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1m'
    minimal_roi = {"0": 0.1}
    stoploss = -0.4
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 50
    can_short = True

    CLOSE_SNAPSHOT_LEN = 100  # 可按需修改，比如 50、100、200

    def informative_pairs(self):
        EXTRA_PAIRS = {
            "ETH/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT", "SOL/USDT:USDT",
            "DOGE/USDT:USDT", "TRX/USDT:USDT", "ADA/USDT:USDT", "LINK/USDT:USDT",
            "SUI/USDT:USDT", "HYPE/USDT:USDT", "XLM/USDT:USDT", "AVAX/USDT:USDT",
            "BCH/USDT:USDT", "HBAR/USDT:USDT", "LTC/USDT:USDT", "TON/USDT:USDT",
            "DOT/USDT:USDT", "XMR/USDT:USDT", "WLFI/USDT:USDT", "UNI/USDT:USDT",
            "AAVE/USDT:USDT", "ENA/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT",
            "TAO/USDT:USDT", "ASTER/USDT:USDT", "ETC/USDT:USDT", "ONDO/USDT:USDT",
            "IP/USDT:USDT", "ZEC/USDT:USDT", "WLD/USDT:USDT", "POL/USDT:USDT",
        }
        return [(p, self.timeframe) for p in sorted(set(self.dp.current_whitelist()) | EXTRA_PAIRS)]

    def get_all_coin_close(self):
        inf_pairs = self.informative_pairs()
        target_pairs = [p for p, tf in inf_pairs]
        closes_list = {}
        df_first = self.dp.get_pair_dataframe(pair=target_pairs[0], timeframe=self.timeframe)
        timestamps = df_first['date'].tail(self.CLOSE_SNAPSHOT_LEN).tolist()
        for p in target_pairs:
            df_p = self.dp.get_pair_dataframe(pair=p, timeframe=self.timeframe)
            closes = df_p['close'].tail(self.CLOSE_SNAPSHOT_LEN).tolist()
            closes_list[p] = closes

        df_closes = pd.DataFrame(closes_list, index=pd.to_datetime(timestamps))
        df_closes.index.name = "date"
        df_closes.to_csv('/freqtrade/user_data/123.csv')
        return df_closes

    def claculate_pairs(self):
        base_pairs = euclidean_distance_method(closes, self.cfg, topk=self.cfg.max_candidates_per_method)

        # 互斥选择 + 二次过滤：相关性与样本量
        used = set()
        filtered: List[Tuple[str, str, float]] = []
        for a, b, beta in base_pairs:
            if a in used or b in used:
                continue
            sA = closes[a].pct_change().dropna()
            sB = closes[b].pct_change().dropna()
            idx = sA.index.intersection(sB.index)
            if len(idx) < max(self.cfg.min_form_bars, 30):
                continue
            corr = sA.loc[idx].corr(sB.loc[idx])
            if (corr is None) or (not np.isfinite(corr)) or (corr < self.cfg.min_corr):
                continue
            filtered.append((a, b, beta))
            used.add(a);
            used.add(b)
            if len(filtered) >= self.cfg.select_pairs_per_window:
                break

        self._current_pairs = filtered
        self._last_recompute_index = cur_len
        if filtered:
            self.logger.info(f"[PAIR-SELECT] Selected pairs: {filtered}")
        else:
            self.logger.info("[PAIR-SELECT] No pairs selected this round.")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe