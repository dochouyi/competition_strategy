from typing import Dict, List, Tuple, Set
import pandas as pd
from metrics import unified_scaled_distance, estimate_beta_ols
from metrics import pct_change_series, adf_test_simple
import numpy as np, random
from metrics import bollinger_spread_score,market_series, sdr_gamma_diff


Pair = Tuple[str, str, float]

class GASelector:
    def __init__(self, **kwargs):
        self.pairs_per_chrom: int = 5
        self.pop: int = 60
        self.gen: int = 40
        self.cxp: float = 0.8
        self.mutp: float = 0.2
        self.candidate_cap: int = 200
        self.use_log_price: bool = False
        # 评分参数
        self.bb_window: int = 30
        self.takeprofit_z: float = 2.0
        self.close_to_sma: bool = True
        self.fee_bps: float = 0.0
        self.seed: int = 42
        # 候选来源
        self.candidate_source: str = "sdr"  # "distance" 或 "sdr"

        random.seed(self.seed)
        np.random.seed(self.seed)
    def _build_candidates(self, prices: Dict[str, pd.Series]) -> List[Pair]:
        keys = list(prices.keys())
        pool = []
        if self.candidate_source == "distance":
            raw = []
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a, b = keys[i], keys[j]
                    d = unified_scaled_distance(prices[a], prices[b])
                    if np.isfinite(d):
                        raw.append((a, b, d))
            raw.sort(key=lambda x: x[2])
            raw = raw[:self.candidate_cap]
            for a, b, _ in raw:
                a_s, b_s = prices[a], prices[b]

                beta = estimate_beta_ols(a_s, b_s, use_log_price=self.use_log_price)
                pool.append((a, b, beta))
        else:
            market_r = market_series(prices, "symbol", "BTC/USDT:USDT")
            raw = []
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a, b = keys[i], keys[j]
                    a_s, b_s = prices[a], prices[b]
                    ra, rb = pct_change_series(a_s), pct_change_series(b_s)

                    gamma = sdr_gamma_diff(a_s, b_s, market_r)
                    gt = ra.values - rb.values - gamma * market_r.values
                    t_stat = adf_test_simple(gt, lags=1)
                    if np.isfinite(t_stat):
                        raw.append((a, b, t_stat))
            raw.sort(key=lambda x: x[2])    #按照从小到大排列
            raw = raw[:self.candidate_cap]
            for a, b, _ in raw:
                a_s, b_s = prices[a], prices[b]
                beta = estimate_beta_ols(a_s, b_s, use_log_price=self.use_log_price)
                pool.append((a, b, beta))
        return pool

    def select_pairs(self, prices: Dict[str, pd.Series]) -> List[Pair]:

        candidates = self._build_candidates(prices)
        # 验证样本
        if not candidates:
            return []

        K = min(self.pairs_per_chrom, max(1, len(candidates)//5))

        def random_chrom():
            used = set(); chrom = []
            idxs = list(range(len(candidates)))
            random.shuffle(idxs)
            for idx in idxs:
                a, b, beta = candidates[idx]
                if (a not in used) and (b not in used):
                    chrom.append(idx); used.add(a); used.add(b)
                    if len(chrom) >= K:
                        break
            return chrom

        def fitness(ch):
            score_sum = 0.0
            for idx in ch:
                a, b, beta = candidates[idx]
                a_s, b_s = prices[a], prices[b]
                s1, _ = bollinger_spread_score(
                    np.log(a_s) if self.use_log_price else a_s,
                    np.log(b_s) if self.use_log_price else b_s,
                    beta, self.bb_window, self.takeprofit_z, self.fee_bps
                )
                if not np.isfinite(s1): s1 = -1e6
                score_sum += s1
            return score_sum

        pop = [random_chrom() for _ in range(self.pop)]
        fit = [fitness(ch) for ch in pop]

        def tournament():
            k = 3
            cand = random.sample(range(len(pop)), k)
            best = max(cand, key=lambda i: fit[i])
            return pop[best]

        def repair(ch):
            used = set(); out = []
            for idx in ch:
                a, b, _ = candidates[idx]
                if a in used or b in used: continue
                out.append(idx); used.add(a); used.add(b)
                if len(out) >= K: break
            if len(out) < K:
                pool = list(range(len(candidates))); random.shuffle(pool)
                for idx in pool:
                    a, b, _ = candidates[idx]
                    if a in used or b in used or idx in out: continue
                    out.append(idx); used.add(a); used.add(b)
                    if len(out) >= K: break
            return out[:K]

        def crossover(p1, p2):
            if random.random() > self.cxp: return p1[:], p2[:]
            cut = random.randint(1, max(1, K-1))
            c1 = p1[:cut] + [g for g in p2 if g not in p1[:cut]]
            c2 = p2[:cut] + [g for g in p1 if g not in p2[:cut]]
            return repair(c1), repair(c2)

        def mutate(ch):
            if random.random() > self.mutp: return ch
            used_assets = set()
            for idx in ch:
                a, b, _ = candidates[idx]
                used_assets.add(a); used_assets.add(b)
            pool = []
            for i, (a, b, _) in enumerate(candidates):
                if (a not in used_assets) and (b not in used_assets):
                    pool.append(i)
            if not pool: return ch
            pos = random.randrange(len(ch))
            ch2 = ch[:]; ch2[pos] = random.choice(pool)
            return repair(ch2)

        for _ in range(self.gen):
            new_pop = []
            while len(new_pop) < self.pop:
                p1 = tournament(); p2 = tournament()
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1); c2 = mutate(c2)
                new_pop.extend([c1, c2])
            pop = new_pop[:self.pop]
            fit = [fitness(ch) for ch in pop]

        best_idx = max(range(len(pop)), key=lambda i: fit[i])
        best = pop[best_idx]
        out = []
        for idx in best:
            a, b, beta = candidates[idx]
            out.append((a, b, beta))
        return out