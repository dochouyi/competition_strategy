from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np
from metrics import pearson_returns_correlation, estimate_beta_ols

Pair = Tuple[str, str, float]

class CorrelationSelector:
    def __init__(self, **kwargs):
        self.select_pairs_per_window: int = 3   #最多的候选数量,返回的总数量不超过这个值
        self.min_corr: float = 0.2  #皮尔逊相关系数的筛选阈值
        self.use_log_price: bool = True     #计算beta的时候是否使用log尺度

    def select_pairs(self, prices: Dict[str, pd.Series]) -> List[Pair]:

        keys = list(prices.keys())
        scores: List[Pair] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                c = pearson_returns_correlation(prices[a], prices[b])      #皮尔逊相关系数
                if np.isfinite(c):
                    scores.append((a, b, c))

        scores.sort(key=lambda x: -x[2])         #从大到小排列
        out: List[Pair] = []
        for a, b, c in scores:
            a_s, b_s = prices[a], prices[b]

            if c < self.min_corr:
                continue
            beta = estimate_beta_ols(a_s, b_s, use_log_price=self.use_log_price)
            out.append((a, b, beta))
            if len(out) >= self.select_pairs_per_window:
                break
        return out