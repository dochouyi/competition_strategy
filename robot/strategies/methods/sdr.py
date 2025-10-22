from typing import Optional
from metrics import pct_change_series, adf_test_simple
from typing import Dict, List, Tuple, Set
import pandas as pd
from metrics import pearson_returns_correlation, estimate_beta_ols, market_series, sdr_gamma_diff
import numpy as np

Pair = Tuple[str, str, float]

class SDRSelector:
    def __init__(self, **kwargs):
        self.select_pairs_per_window: int = 3   #最多的候选数量,返回的总数量不超过这个值
        self.min_corr: float = 0.2  #皮尔逊相关系数的筛选阈值
        self.use_log_price: bool = True     #计算beta的时候是否使用log尺度

        self.market_mode: str = "mean"  # "mean" 或 "symbol"
        self.market_symbol: Optional[str] = "BTC/USDT:USDT"

    def select_pairs(self, prices: Dict[str, pd.Series]) -> List[Pair]:
        market_r = market_series(prices, self.market_mode, self.market_symbol)
        keys = list(prices.keys())
        scored: List[Pair] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                a_s, b_s = prices[a], prices[b]
                ra, rb = pct_change_series(a_s), pct_change_series(b_s)

                gamma = sdr_gamma_diff(a_s, b_s, market_r)  #计算市场参考下的gamma差值
                gt = ra.values - rb.values - gamma * market_r.values        #构造 gt 序列：两资产收益率之差，再减去 gamma 调整的市场收益率
                t_stat = adf_test_simple(gt, lags=1)    #用 ADF（单位根）检验 gt 是否平稳，返回 t 统计量。
                if not np.isfinite(t_stat):
                    continue
                scored.append((a, b, t_stat))
        scored.sort(key=lambda x: x[2])     #从小到大排列

        out: List[Pair] = []
        for a, b, _ in scored:
            a_s, b_s = prices[a], prices[b]
            c = pearson_returns_correlation(a_s, b_s)
            if not np.isfinite(c) or c < self.min_corr:
                continue
            beta = estimate_beta_ols(a_s, b_s, use_log_price=self.use_log_price)

            out.append((a, b, beta))
            if len(out) >= self.select_pairs_per_window:
                break
        return out