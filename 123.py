import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

# -----------------------------
# 配置与算法函数
# -----------------------------
@dataclass
class PairSelectorConfig:
    bb_window: int = 30          # 仅用于 beta 估计最小窗长限制
    min_corr: float = 0.2
    use_log_price: bool = False
    select_pairs_per_window: int = 1
    max_candidates_per_method: int = 20

def estimate_beta_on_window(a: pd.Series, b: pd.Series, use_log_price=True) -> float:
    a = np.log(a) if use_log_price else a.copy()
    b = np.log(b) if use_log_price else b.copy()

    x = b.values
    y = a.values
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (vx ** 2).sum()
    if denom <= 0:
        return 1.0
    beta = (vx * vy).sum() / denom
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
    scores: List[Tuple[str, str, float]] = []
    keys = list(prices.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            d = unified_scaled_distance(prices[a], prices[b])
            if not np.isfinite(d):
                continue
            scores.append((a, b, d))
    scores.sort(key=lambda x: x[2])
    pairs: List[Tuple[str, str, float]] = []
    # 前 topk 个距离最小的对，计算 beta
    for a, b, _ in scores[:topk]:
        beta = estimate_beta_on_window(
            prices[a],
            prices[b],
            use_log_price=cfg.use_log_price,
        )
        pairs.append((a, b, beta))
    return pairs

# -----------------------------
# 数据加载与主流程
# -----------------------------
def load_prices_from_csv(path: str) -> Dict[str, pd.Series]:
    """
    期望 CSV:
    - 第一列为时间或索引（可为字符串/时间/整数），其余列为各币对收盘价
    - 若没有时间列，也可以仅有数值列，将使用行号作为索引
    """
    df = pd.read_csv(path)
    # 若第一列看起来是时间列，尝试解析为 datetime 索引
    if df.shape[1] >= 2:
        first_col = df.columns[0]
        # 判断是否多数值列，如果第一列非纯数值，视作索引列
        if not pd.api.types.is_numeric_dtype(df[first_col]):
            # 尝试解析为时间
            ts = pd.to_datetime(df[first_col], errors='coerce')
            if ts.notna().mean() > 0.7:
                df[first_col] = ts
            df = df.set_index(first_col)
        else:
            # 第一列是数值，可能并非时间索引。直接用默认行号索引。
            pass
    else:
        # 只有一列时，无法组成多资产，直接报错
        raise ValueError("CSV 至少需要两列（索引列 + 一个资产列），建议为：time, SYMBOL1, SYMBOL2, ...")

    # 去除全空列
    df = df.dropna(axis=1, how='all')
    # 只保留数值列
    value_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    df = df[value_cols]
    df = df.astype(float)
    prices = {col: df[col].dropna() for col in df.columns}
    return prices

def select_pairs_from_prices(
    prices: Dict[str, pd.Series],
    cfg: PairSelectorConfig
) -> List[Tuple[str, str, float]]:
    """
    主流程:
    - 通过欧式距离候选 topK
    - 互斥选择 + 二次过滤（收益相关性 + 样本量）
    """
    # 基于距离的候选
    base_pairs = euclidean_distance_method(
        prices,
        cfg,
        topk=cfg.max_candidates_per_method
    )

    # 互斥与二次过滤
    used = set()
    filtered: List[Tuple[str, str, float]] = []

    # 预先对齐所有序列的时间索引（仅在需要时交集）
    # 这里按 pair 局部交集即可，维持灵活性
    for a, b, beta in base_pairs:
        if a in used or b in used:
            continue
        sA = prices[a].pct_change().dropna()
        sB = prices[b].pct_change().dropna()
        idx = sA.index.intersection(sB.index)

        corr = sA.loc[idx].corr(sB.loc[idx])
        if (corr is None) or (not np.isfinite(corr)) or (corr < cfg.min_corr):
            continue
        filtered.append((a, b, beta))
        used.add(a)
        used.add(b)
        if len(filtered) >= cfg.select_pairs_per_window:
            break
    return filtered

def main():
    csv_path = "123.csv"  # 与脚本同级目录
    cfg = PairSelectorConfig(
        bb_window=30,
        min_corr=0.2,
        use_log_price=False,         # 若币价跨数量级大，可改为 True
        select_pairs_per_window=1,   # 每轮选择的最终对数
        max_candidates_per_method=20 # 候选搜索宽度
    )

    prices = load_prices_from_csv(csv_path)

    selected = select_pairs_from_prices(prices, cfg)
    if selected:
        print("[PAIR-SELECT] Selected pairs:", selected)
    else:
        print("[PAIR-SELECT] No pairs selected this round.")

if __name__ == "__main__":
    main()