from metrics import engle_granger_beta, adf_test_simple
from typing import Dict, List, Tuple, Set
import pandas as pd
from metrics import pearson_returns_correlation
import numpy as np

Pair = Tuple[str, str, float]

class CointegrationSelector:

    def __init__(self, **kwargs):
        self.select_pairs_per_window: int = 3   #最多的候选数量,返回的总数量不超过这个值
        self.use_log_price: bool = True     #计算beta的时候是否使用log尺度
        self.min_corr: float = 0.2  #皮尔逊相关系数的筛选阈值
        self.adf_lags: int = 1
        self.adf_crit: float = -3.4

    def select_pairs(self, prices: Dict[str, pd.Series]) -> List[Pair]:

        keys = list(prices.keys())
        candidates: List[Tuple[str, str, float, float]] = []

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):

                a, b = keys[i], keys[j]
                a_s, b_s = prices[a], prices[b]

                beta_coint, resid = engle_granger_beta(a_s, b_s, use_log_price=self.use_log_price)
                if resid.empty or resid.isna().all():
                    continue
                t_stat = adf_test_simple(resid.values, lags=self.adf_lags)
                if np.isfinite(t_stat) and t_stat < self.adf_crit:
                    c = pearson_returns_correlation(a_s, b_s)
                    if np.isfinite(c) and c >= self.min_corr:
                        # 使用协整beta
                        candidates.append((a, b, float(beta_coint), float(c)))

        candidates.sort(key=lambda x: -x[3])
        out: List[Pair] = []
        for a, b, beta, _ in candidates:
            out.append((a, b, beta))
            if len(out) >= self.select_pairs_per_window:
                break
        return out