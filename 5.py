import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import math
import random

# -----------------------------
# 配置与数据工具
# -----------------------------
@dataclass
class PairSelectorConfig:
  bb_window: int = 30                # 用于布林策略评分/回测的窗口（SMA/STD）
  min_form_bars: int = 60            # 最小样本数
  min_corr: float = 0.2              # 二次过滤的最小收益相关性
  use_log_price: bool = False
  select_pairs_per_window: int = 5   # 最终要输出的对数（除GA/NSGA-II外）
  max_candidates_per_method: int = 100  # 候选集合规模上限（用于距离/相关等）
  # Cointegration 参数
  adf_lags: int = 1                  # 简化ADF滞后
  adf_crit: float = -3.3             # 简化ADF阈值（越小越容易拒绝单位根）；注意仅作近似
  # SDR 参数
  sdr_market_mode: str = "mean"      # "mean" 或 "symbol"
  sdr_market_symbol: Optional[str] = None
  # GA/NSGA 参数
  ga_pop: int = 60
  ga_gen: int = 40
  ga_cxp: float = 0.8
  ga_mutp: float = 0.2
  ga_pairs_per_chrom: int = 5        # 染色体中对数
  nsga_pop: int = 80
  nsga_gen: int = 60
  nsga_pairs_per_chrom: int = 5
  # 简化评分回测参数
  score_takeprofit_z: float = 2.0    # 开仓阈值（布林带上/下轨）
  score_close_to_sma: bool = True    # 回归至SMA即平仓
  score_fee_bps: float = 0.0         # 简化手续费（基点）
  # 随机种子（可复现）
  seed: int = 42

def load_prices_from_csv(path: str) -> Dict[str, pd.Series]:
  df = pd.read_csv(path)
  if df.shape[1] >= 2:
    first_col = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[first_col]):
      ts = pd.to_datetime(df[first_col], errors='coerce')
      if ts.notna().mean() > 0.7:
        df[first_col] = ts
      df = df.set_index(first_col)
  else:
    raise ValueError("CSV 至少需要两列（索引列 + 一个资产列）")
  df = df.dropna(axis=1, how='all')
  value_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
  if len(value_cols) < 2:
    raise ValueError("有效的数值资产列不足 2 列")
  df = df[value_cols].astype(float)
  prices = {col: df[col].dropna() for col in df.columns}
  return prices

# -----------------------------
# 通用工具函数
# -----------------------------
def intersect_align(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
  idx = a.dropna().index.intersection(b.dropna().index)
  return a.loc[idx], b.loc[idx]

def estimate_beta_ols(a: pd.Series, b: pd.Series, use_log_price=True, min_len: int = 30) -> float:
  a = np.log(a.dropna()) if use_log_price else a.dropna().copy()
  b = np.log(b.dropna()) if use_log_price else b.dropna().copy()
  a, b = intersect_align(a, b)
  if len(a) < min_len:
    return 1.0
  x = b.values
  y = a.values
  vx = x - x.mean()
  vy = y - y.mean()
  denom = (vx**2).sum()
  if denom <= 0:
    return 1.0
  beta = (vx * vy).sum() / denom
  if not np.isfinite(beta):
    beta = 1.0
  return float(np.clip(beta, 0.1, 10.0))

def series_returns(s: pd.Series) -> pd.Series:
  return s.pct_change().dropna()

def corr_returns(a: pd.Series, b: pd.Series) -> float:
  ra, rb = series_returns(a), series_returns(b)
  idx = ra.index.intersection(rb.index)
  if len(idx) < 3:
    return np.nan
  return float(ra.loc[idx].corr(rb.loc[idx]))

# -----------------------------
# 方法1：欧氏距离
# -----------------------------
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

def select_distance_pairs(prices: Dict[str, pd.Series], cfg: PairSelectorConfig) -> List[Tuple[str, str, float]]:
  keys = list(prices.keys())
  scores: List[Tuple[str, str, float]] = []
  for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
      a, b = keys[i], keys[j]
      d = unified_scaled_distance(prices[a], prices[b])
      if np.isfinite(d):
        scores.append((a, b, d))
  scores.sort(key=lambda x: x[2])
  # 二次过滤 + 互斥
  used: Set[str] = set()
  out: List[Tuple[str, str, float]] = []
  for a, b, _ in scores[: cfg.max_candidates_per_method]:
    beta = estimate_beta_ols(prices[a], prices[b], use_log_price=cfg.use_log_price, min_len=max(30, cfg.bb_window))
    if a in used or b in used:
      continue
    a_s, b_s = intersect_align(prices[a], prices[b])
    if len(a_s) < max(cfg.min_form_bars, 30):
      continue
    c = corr_returns(a_s, b_s)
    if not np.isfinite(c) or c < cfg.min_corr:
      continue
    out.append((a, b, beta))
    used.add(a); used.add(b)
    if len(out) >= cfg.select_pairs_per_window:
      break
  return out

# -----------------------------
# 方法2：相关性
# -----------------------------
def select_correlation_pairs(prices: Dict[str, pd.Series], cfg: PairSelectorConfig) -> List[Tuple[str, str, float]]:
  keys = list(prices.keys())
  scores: List[Tuple[str, str, float]] = []
  for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
      a, b = keys[i], keys[j]
      c = corr_returns(prices[a], prices[b])
      if np.isfinite(c):
        scores.append((a, b, c))
  scores.sort(key=lambda x: -x[2])  # 高相关优先
  used: Set[str] = set()
  out: List[Tuple[str, str, float]] = []
  for a, b, _ in scores[: cfg.max_candidates_per_method]:
    beta = estimate_beta_ols(prices[a], prices[b], use_log_price=cfg.use_log_price, min_len=max(30, cfg.bb_window))
    if a in used or b in used:
      continue
    a_s, b_s = intersect_align(prices[a], prices[b])
    if len(a_s) < max(cfg.min_form_bars, 30):
      continue
    out.append((a, b, beta))
    used.add(a); used.add(b)
    if len(out) >= cfg.select_pairs_per_window:
      break
  return out

# -----------------------------
# 方法3：协整（Engle-Granger 简化）
# -----------------------------
def adf_test_simple(x: np.ndarray, lags: int = 1) -> float:
  """
  返回简化的 tau 统计量（越小越显著拒绝单位根）
  过程：
  - Δx_t = x_t - x_{t-1}
  - 回归 Δx_t ~ c + phi * x_{t-1} + sum gamma_i * Δx_{t-i}
  - 以 phi 的 t 值作为ADF统计量近似（负值大→拒绝单位根）
  注意：此实现是近似版，非严格替代 statsmodels。
  """
  x = x.astype(float)
  dx = np.diff(x)
  x_1 = x[:-1]
  n = len(dx)
  if n - lags - 1 <= 3:
    return np.nan
  # 构建设计矩阵
  Y = dx[lags:]
  X_cols = []
  # 常数项
  X_cols.append(np.ones(len(Y)))
  # 滞后水平 x_{t-1}
  X_cols.append(x_1[lags:])
  # 滞后差分项
  for i in range(1, lags + 1):
    X_cols.append(dx[lags - i: -i])
  X = np.vstack(X_cols).T  # shape: [m, k]
  # OLS
  XtX = X.T @ X
  try:
    XtX_inv = np.linalg.inv(XtX)
  except np.linalg.LinAlgError:
    return np.nan
  beta_hat = XtX_inv @ (X.T @ Y)
  residuals = Y - X @ beta_hat
  sigma2 = (residuals @ residuals) / (len(Y) - X.shape[1])
  var_beta = sigma2 * XtX_inv
  # 目标系数：phi 的 t 值（第二个系数）
  phi = beta_hat[1]
  se_phi = math.sqrt(var_beta[1, 1]) if var_beta[1, 1] > 0 else np.inf
  t_phi = phi / se_phi if se_phi > 0 else np.nan
  return float(t_phi)

def engle_granger_beta(a: pd.Series, b: pd.Series, use_log_price=True) -> Tuple[float, pd.Series]:
  # 回归 a ~ alpha + beta * b，取残差做ADF
  a = np.log(a.dropna()) if use_log_price else a.dropna().copy()
  b = np.log(b.dropna()) if use_log_price else b.dropna().copy()
  a, b = intersect_align(a, b)
  x = b.values
  y = a.values
  if len(x) < 30:
    return 1.0, pd.Series(index=a.index, dtype=float)
  X = np.vstack([np.ones(len(x)), x]).T
  XtX = X.T @ X
  try:
    XtX_inv = np.linalg.inv(XtX)
  except np.linalg.LinAlgError:
    return 1.0, pd.Series(index=a.index, dtype=float)
  beta_hat = XtX_inv @ (X.T @ y)
  alpha, beta = beta_hat[0], beta_hat[1]
  resid = y - (alpha + beta * x)
  return float(np.clip(beta, 0.1, 10.0)), pd.Series(resid, index=a.index)

def select_cointegration_pairs(prices: Dict[str, pd.Series], cfg: PairSelectorConfig) -> List[Tuple[str, str, float]]:
  keys = list(prices.keys())
  candidates: List[Tuple[str, str, float]] = []  # (a,b,beta) 通过协整检验者
  for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
      a, b = keys[i], keys[j]
      a_s, b_s = intersect_align(prices[a], prices[b])
      if len(a_s) < max(cfg.min_form_bars, 30):
        continue
      beta, resid = engle_granger_beta(a_s, b_s, use_log_price=cfg.use_log_price)
      if resid.empty or resid.isna().all():
        continue
      t_stat = adf_test_simple(resid.values, lags=cfg.adf_lags)
      # ADF：t_stat 越小（更负）越显著，这里用阈值判断
      if np.isfinite(t_stat) and t_stat < cfg.adf_crit:
        candidates.append((a, b, beta))
  # 二次过滤：收益相关性 + 互斥 + 裁剪
  used: Set[str] = set()
  out: List[Tuple[str, str, float]] = []
  # 简单评分：相关性越高先选
  scored = []
  for a, b, beta in candidates:
    c = corr_returns(prices[a], prices[b])
    scored.append((a, b, beta, c if np.isfinite(c) else -1))
  scored.sort(key=lambda x: -x[3])
  for a, b, beta, c in scored:
    if a in used or b in used:
      continue
    out.append((a, b, beta))
    used.add(a); used.add(b)
    if len(out) >= cfg.select_pairs_per_window:
      break
  return out

# -----------------------------
# 方法4：SDR
# -----------------------------
def _market_series(prices: Dict[str, pd.Series], mode: str, symbol: Optional[str]) -> pd.Series:
  # 返回市场代理收益序列
  dfs = pd.concat(prices.values(), axis=1, join="inner")
  dfs.columns = list(prices.keys())
  if mode == "symbol" and symbol is not None and symbol in prices:
    base = prices[symbol]
    return series_returns(base)
  # 默认：横截面平均价格 -> 再取收益
  px_mean = dfs.mean(axis=1)
  return series_returns(px_mean)

def sdr_gamma_diff(a: pd.Series, b: pd.Series, market_r: pd.Series) -> float:
  # 以简单CAPM式：R_i = alpha + gamma * R_m + eps，估计各自 gamma，再取差
  ra, rb = series_returns(a), series_returns(b)
  idx = ra.index.intersection(rb.index).intersection(market_r.index)
  if len(idx) < 30:
    return 0.0
  ri_a = ra.loc[idx].values
  ri_b = rb.loc[idx].values
  rm = market_r.loc[idx].values
  def ols_beta(y, x):
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (vx**2).sum()
    if denom <= 0:
      return 0.0
    beta = (vx * vy).sum() / denom
    return float(beta)
  gamma_a = ols_beta(ri_a, rm)
  gamma_b = ols_beta(ri_b, rm)
  return float(gamma_a - gamma_b)

def sdr_score(a: pd.Series, b: pd.Series, market_r: pd.Series) -> float:
  # 根据论文 G_t = R^A_t - R^B_t - Γ r^m_t，计算残差序列，衡量其“均值回复强度”
  ra, rb = series_returns(a), series_returns(b)
  idx = ra.index.intersection(rb.index).intersection(market_r.index)
  if len(idx) < 30:
    return -np.inf
  gamma = sdr_gamma_diff(a, b, market_r)
  gt = ra.loc[idx].values - rb.loc[idx].values - gamma * market_r.loc[idx].values
  # 均值回复评分：-单位根倾向 -> 使用ADF t统计，越负越好
  t_stat = adf_test_simple(gt, lags=1)
  if not np.isfinite(t_stat):
    return -np.inf
  # 也可以结合方差小/自相关负等指标，这里直接用 -t_stat 作为分数（越大越好）
  return -t_stat

def select_sdr_pairs(prices: Dict[str, pd.Series], cfg: PairSelectorConfig) -> List[Tuple[str, str, float]]:
  market_r = _market_series(prices, cfg.sdr_market_mode, cfg.sdr_market_symbol)
  keys = list(prices.keys())
  scores: List[Tuple[str, str, float]] = []
  for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
      a, b = keys[i], keys[j]
      a_s, b_s = intersect_align(prices[a], prices[b])
      if len(a_s) < max(cfg.min_form_bars, 30):
        continue
      sc = sdr_score(a_s, b_s, market_r)
      scores.append((a, b, sc))
  scores = [x for x in scores if np.isfinite(x[2])]
  scores.sort(key=lambda x: -x[2])
  # 二次过滤 + 互斥
  used: Set[str] = set()
  out: List[Tuple[str, str, float]] = []
  for a, b, _ in scores[: cfg.max_candidates_per_method]:
    if a in used or b in used:
      continue
    beta = estimate_beta_ols(prices[a], prices[b], use_log_price=cfg.use_log_price, min_len=max(30, cfg.bb_window))
    c = corr_returns(prices[a], prices[b])
    if not np.isfinite(c) or c < cfg.min_corr:
      continue
    out.append((a, b, beta))
    used.add(a); used.add(b)
    if len(out) >= cfg.select_pairs_per_window:
      break
  return out

# -----------------------------
# 简化布林回测评分（用于 GA/NSGA）
# -----------------------------
def bollinger_spread_score(a: pd.Series, b: pd.Series, beta: float, window: int, take_z: float,
                           close_to_sma: bool, fee_bps: float) -> Tuple[float, float]:
  """
  基于价差 S_t = a_t - beta * b_t（价格或log价格都可，但要一致）
  策略：超上轨做空S（空a多b），破下轨做多S（多a空b）；回到SMA平仓。
  返回：(收益评分, 最大回撤近似)
  说明：这是轻量评分器，用于遗传算法排名，非完整回测。
  """
  a, b = a.dropna(), b.dropna()
  idx = a.index.intersection(b.index)
  if len(idx) < window + 10:
    return -np.inf, np.inf
  A = a.loc[idx].values
  B = b.loc[idx].values
  S = A - beta * B
  # 滑窗SMA/STD
  sma = pd.Series(S).rolling(window).mean().values
  std = pd.Series(S).rolling(window).std(ddof=0).values
  up = sma + take_z * std
  dn = sma - take_z * std

  pnl = 0.0
  equity = 0.0
  peak = 0.0
  maxdd = 0.0
  pos = 0  # 1=long spread, -1=short spread, 0=flat
  entry = 0.0
  fee = fee_bps / 10000.0

  for t in range(window, len(S)):
    s = S[t]; mu = sma[t]; u = up[t]; d = dn[t]
    if np.isnan(mu) or np.isnan(u) or np.isnan(d):
      continue
    # 开仓逻辑
    if pos == 0:
      if s > u:
        pos = -1
        entry = s
        pnl -= abs(s) * fee
      elif s < d:
        pos = 1
        entry = s
        pnl -= abs(s) * fee
    else:
      # 平仓逻辑：回到SMA或越界反向
      close_cond = (abs(s - mu) < 1e-12) if close_to_sma else (d < s < u)
      if close_cond:
        pnl += (s - entry) * pos
        pnl -= abs(s) * fee
        pos = 0
      else:
        # 浮动盈亏用于估算回撤
        float_pnl = pnl + (s - entry) * pos
        equity = float_pnl
        peak = max(peak, equity)
        dd = (peak - equity)
        maxdd = max(maxdd, dd)

  # 若尾仓未平，进行结算（按SMA）
  if pos != 0 and not np.isnan(sma[-1]):
    pnl += (s - sma[-1]) * pos
    pnl -= abs(s) * fee
    pos = 0

  # 归一化评分：以std均值缩放
  std_mean = np.nanmean(std)
  norm = std_mean if (std_mean is not None and std_mean > 1e-12) else 1.0
  score = pnl / norm
  maxdd_n = maxdd / norm if norm > 0 else maxdd
  return float(score), float(maxdd_n)

# -----------------------------
# 方法5：GA
# -----------------------------
def build_candidate_pairs(prices: Dict[str, pd.Series], cfg: PairSelectorConfig, base_method: str = "distance",
                          candidate_cap: int = 100) -> List[Tuple[str, str, float]]:
  # 使用某个基础方法先生成候选对集合，GA/NSGA在其上组合优化
  if base_method == "distance":
    base = []
    keys = list(prices.keys())
    for i in range(len(keys)):
      for j in range(i + 1, len(keys)):
        a, b = keys[i], keys[j]
        d = unified_scaled_distance(prices[a], prices[b])
        if np.isfinite(d):
          base.append((a, b, d))
    base.sort(key=lambda x: x[2])
    base = base[:candidate_cap]
    out = []
    for a, b, _ in base:
      beta = estimate_beta_ols(prices[a], prices[b], use_log_price=cfg.use_log_price, min_len=max(30, cfg.bb_window))
      out.append((a, b, beta))
    return out
  elif base_method == "sdr":
    # SDR分数高的先进入候选
    market_r = _market_series(prices, cfg.sdr_market_mode, cfg.sdr_market_symbol)
    keys = list(prices.keys())
    scores = []
    for i in range(len(keys)):
      for j in range(i + 1, len(keys)):
        a, b = keys[i], keys[j]
        a_s, b_s = intersect_align(prices[a], prices[b])
        if len(a_s) < max(cfg.min_form_bars, 30):
          continue
        sc = sdr_score(a_s, b_s, market_r)
        if np.isfinite(sc):
          scores.append((a, b, sc))
    scores.sort(key=lambda x: -x[2])
    base = scores[:candidate_cap]
    out = []
    for a, b, _ in base:
      beta = estimate_beta_ols(prices[a], prices[b], use_log_price=cfg.use_log_price, min_len=max(30, cfg.bb_window))
      out.append((a, b, beta))
    return out
  else:
    raise ValueError("未知 base_method")

def ga_select_pairs(prices: Dict[str, pd.Series], cfg: PairSelectorConfig) -> List[Tuple[str, str, float]]:
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  candidates = build_candidate_pairs(prices, cfg, base_method="distance",
                                     candidate_cap=max(cfg.max_candidates_per_method, 50))
  # 过滤：确保每个候选对有足够样本
  valid = []
  for a, b, beta in candidates:
    a_s, b_s = intersect_align(prices[a], prices[b])
    if len(a_s) >= max(cfg.min_form_bars, 30):
      valid.append((a, b, beta))
  candidates = valid
  if not candidates:
    return []

  # 基因表示：长度K的对列表（不共享资产）
  K = min(cfg.ga_pairs_per_chrom, max(1, len(candidates)//5))
  # 工具：生成互斥资产的K对
  def random_chrom():
    used = set()
    chrom = []
    idxs = list(range(len(candidates)))
    random.shuffle(idxs)
    for idx in idxs:
      a, b, beta = candidates[idx]
      if (a not in used) and (b not in used):
        chrom.append(idx)
        used.add(a); used.add(b)
        if len(chrom) >= K:
          break
    return chrom

  # 适应度：累积评分（布林回测分数之和）
  def fitness(chrom):
    score_sum = 0.0
    for idx in chrom:
      a, b, beta = candidates[idx]
      a_s, b_s = intersect_align(prices[a], prices[b])
      s1, _ = bollinger_spread_score(
        np.log(a_s) if cfg.use_log_price else a_s,
        np.log(b_s) if cfg.use_log_price else b_s,
        beta, cfg.bb_window, cfg.score_takeprofit_z, cfg.score_close_to_sma, cfg.score_fee_bps
      )
      if not np.isfinite(s1):
        s1 = -1e6
      score_sum += s1
    return score_sum

  # 初始种群
  pop = [random_chrom() for _ in range(cfg.ga_pop)]
  # 评估
  fit = [fitness(ch) for ch in pop]

  def tournament():
    k = 3
    cand = random.sample(range(len(pop)), k)
    best = max(cand, key=lambda i: fit[i])
    return pop[best]

  def crossover(p1, p2):
    if random.random() > cfg.ga_cxp:
      return p1[:], p2[:]
    # 单点交叉 + 修正互斥
    cut = random.randint(1, max(1, K-1))
    c1 = p1[:cut] + [g for g in p2 if g not in p1[:cut]]
    c2 = p2[:cut] + [g for g in p1 if g not in p2[:cut]]
    # 修正长度与互斥资产
    def repair(ch):
      used = set()
      out = []
      for idx in ch:
        a, b, _ = candidates[idx]
        if a in used or b in used:
          continue
        out.append(idx)
        used.add(a); used.add(b)
        if len(out) >= K:
          break
      if len(out) < K:
        # 补齐
        pool = list(range(len(candidates)))
        random.shuffle(pool)
        for idx in pool:
          if idx in out:
            continue
          a, b, _ = candidates[idx]
          if a in used or b in used:
            continue
          out.append(idx); used.add(a); used.add(b)
          if len(out) >= K:
            break
      return out[:K]
    return repair(c1), repair(c2)

  def mutate(ch):
    if random.random() > cfg.ga_mutp:
      return ch
    # 随机替换一个基因
    used_assets = set()
    for idx in ch:
      a, b, _ = candidates[idx]
      used_assets.add(a); used_assets.add(b)
    pool = []
    for i, (a, b, _) in enumerate(candidates):
      if (a not in used_assets) and (b not in used_assets):
        pool.append(i)
    if not pool:
      return ch
    pos = random.randrange(len(ch))
    ch2 = ch[:]
    ch2[pos] = random.choice(pool)
    return ch2

  for _gen in range(cfg.ga_gen):
    new_pop = []
    while len(new_pop) < cfg.ga_pop:
      p1 = tournament()
      p2 = tournament()
      c1, c2 = crossover(p1, p2)
      c1 = mutate(c1)
      c2 = mutate(c2)
      new_pop.extend([c1, c2])
    pop = new_pop[:cfg.ga_pop]
    fit = [fitness(ch) for ch in pop]

  # 取最优个体，输出其 pair 列表
  best_idx = max(range(len(pop)), key=lambda i: fit[i])
  best = pop[best_idx]
  out = []
  used = set()
  for idx in best:
    a, b, beta = candidates[idx]
    if a in used or b in used:
      continue
    out.append((a, b, beta))
    used.add(a); used.add(b)
  return out

# -----------------------------
# 方法6：NSGA-II（收益最大 + 回撤最小）
# -----------------------------
def nsga_select_pairs(prices: Dict[str, pd.Series], cfg: PairSelectorConfig) -> List[Tuple[str, str, float]]:
  random.seed(cfg.seed)
  np.random.seed(cfg.seed)

  candidates = build_candidate_pairs(prices, cfg, base_method="distance",
                                     candidate_cap=max(cfg.max_candidates_per_method, 80))
  valid = []
  for a, b, beta in candidates:
    a_s, b_s = intersect_align(prices[a], prices[b])
    if len(a_s) >= max(cfg.min_form_bars, 30):
      valid.append((a, b, beta))
  candidates = valid
  if not candidates:
    return []

  K = min(cfg.nsga_pairs_per_chrom, max(1, len(candidates)//5))

  def random_chrom():
    used = set()
    chrom = []
    idxs = list(range(len(candidates)))
    random.shuffle(idxs)
    for idx in idxs:
      a, b, beta = candidates[idx]
      if (a not in used) and (b not in used):
        chrom.append(idx)
        used.add(a); used.add(b)
        if len(chrom) >= K:
          break
    return chrom

  def obj_pair(idx):
    a, b, beta = candidates[idx]
    a_s, b_s = intersect_align(prices[a], prices[b])
    s1, dd = bollinger_spread_score(
      np.log(a_s) if cfg.use_log_price else a_s,
      np.log(b_s) if cfg.use_log_price else b_s,
      beta, cfg.bb_window, cfg.score_takeprofit_z, cfg.score_close_to_sma, cfg.score_fee_bps
    )
    if not np.isfinite(s1):
      s1 = -1e6
    if not np.isfinite(dd):
      dd = 1e6
    return s1, dd

  # 预先缓存每个候选对的两个目标
  cache_obj = [obj_pair(i) for i in range(len(candidates))]

  def objectives(ch):
    # 聚合：总收益 = sum(scores)；总回撤 = sum(maxdd) 或 max(maxdd)
    tot_score = sum(cache_obj[i][0] for i in ch)
    tot_dd = sum(max(0.0, cache_obj[i][1]) for i in ch)
    return (tot_score, tot_dd)

  # 非支配关系：最大化 score，最小化 dd
  def dominates(a_obj, b_obj):
    better_or_equal = (a_obj[0] >= b_obj[0]) and (a_obj[1] <= b_obj[1])
    strictly_better = (a_obj[0] > b_obj[0]) or (a_obj[1] < b_obj[1])
    return better_or_equal and strictly_better

  def fast_nondominated_sort(pop_objs):
    S = [[] for _ in pop_objs]
    n = [0] * len(pop_objs)
    fronts = [[]]
    for p in range(len(pop_objs)):
      for q in range(len(pop_objs)):
        if p == q:
          continue
        if dominates(pop_objs[p], pop_objs[q]):
          S[p].append(q)
        elif dominates(pop_objs[q], pop_objs[p]):
          n[p] += 1
      if n[p] == 0:
        fronts[0].append(p)
    i = 0
    while fronts[i]:
      Q = []
      for p in fronts[i]:
        for q in S[p]:
          n[q] -= 1
          if n[q] == 0:
            Q.append(q)
      i += 1
      fronts.append(Q)
    fronts.pop()
    return fronts

  def crowding_distance(front, pop_objs):
    if not front:
      return []
    distances = [0.0 for _ in front]
    # 对每个目标计算拥挤距离
    for m in range(2):
      vals = [pop_objs[i][m] for i in front]
      order = np.argsort(vals)
      distances[order[0]] = distances[order[-1]] = float('inf')
      vmin, vmax = vals[order[0]], vals[order[-1]]
      if vmax == vmin:
        continue
      for k in range(1, len(front) - 1):
        prev_v = vals[order[k - 1]]
        next_v = vals[order[k + 1]]
        distances[order[k]] += (next_v - prev_v) / (vmax - vmin)
    return distances

  def selection(pop, pop_objs):
    fronts = fast_nondominated_sort(pop_objs)
    new_pop = []
    for front in fronts:
      if len(new_pop) + len(front) > cfg.nsga_pop:
        # 根据拥挤距离选择
        dist = crowding_distance(front, pop_objs)
        order = np.argsort([-dist[i] for i in range(len(front))])
        for idx in order:
          if len(new_pop) >= cfg.nsga_pop:
            break
          new_pop.append(pop[front[idx]])
        break
      else:
        for i in front:
          new_pop.append(pop[i])
    return new_pop

  def crossover(p1, p2):
    cut = random.randint(1, max(1, K-1))
    c1 = p1[:cut] + [g for g in p2 if g not in p1[:cut]]
    c2 = p2[:cut] + [g for g in p1 if g not in p2[:cut]]
    def repair(ch):
      used = set()
      out = []
      for idx in ch:
        a, b, _ = candidates[idx]
        if a in used or b in used:
          continue
        out.append(idx)
        used.add(a); used.add(b)
        if len(out) >= K:
          break
      # 补齐
      if len(out) < K:
        pool = list(range(len(candidates)))
        random.shuffle(pool)
        for idx in pool:
          a, b, _ = candidates[idx]
          if a in used or b in used or idx in out:
            continue
          out.append(idx)
          used.add(a); used.add(b)
          if len(out) >= K:
            break
      return out[:K]
    return repair(c1), repair(c2)

  def mutate(ch):
    if random.random() > 0.2:
      return ch
    used_assets = set()
    for idx in ch:
      a, b, _ = candidates[idx]
      used_assets.add(a); used_assets.add(b)
    pool = []
    for i, (a, b, _) in enumerate(candidates):
      if (a not in used_assets) and (b not in used_assets):
        pool.append(i)
    if not pool:
      return ch
    pos = random.randrange(len(ch))
    ch2 = ch[:]
    ch2[pos] = random.choice(pool)
    return ch2

  # 初始化
  pop = [random_chrom() for _ in range(cfg.nsga_pop)]
  pop_objs = [objectives(ch) for ch in pop]

  for _ in range(cfg.nsga_gen):
    # 生成子代
    children = []
    while len(children) < cfg.nsga_pop:
      p1 = random.choice(pop)
      p2 = random.choice(pop)
      c1, c2 = crossover(p1, p2)
      c1 = mutate(c1); c2 = mutate(c2)
      children.extend([c1, c2])
    children = children[:cfg.nsga_pop]
    children_objs = [objectives(ch) for ch in children]
    # 合并选择
    pop = pop + children
    pop_objs = pop_objs + children_objs
    pop = selection(pop, pop_objs)
    pop_objs = [objectives(ch) for ch in pop]

  # 最终从第一前沿选择拥挤距离最大的解或收益最高的解
  fronts = fast_nondominated_sort(pop_objs)
  f0 = fronts[0] if fronts else list(range(len(pop_objs)))
  # 选择收益最高的个体
  best_idx = max(f0, key=lambda i: pop_objs[i][0])
  best = pop[best_idx]

  out = []
  used = set()
  for idx in best:
    a, b, beta = candidates[idx]
    if a in used or b in used:
      continue
    out.append((a, b, beta))
    used.add(a); used.add(b)
  return out

# -----------------------------
# 统一入口
# -----------------------------
def select_pairs_all_methods(prices: Dict[str, pd.Series], cfg: PairSelectorConfig):
  return {
    "Distance": select_distance_pairs(prices, cfg),
    "Correlation": select_correlation_pairs(prices, cfg),
    "Cointegration": select_cointegration_pairs(prices, cfg),
    "SDR": select_sdr_pairs(prices, cfg),
    "GA": ga_select_pairs(prices, cfg),
    "NSGA-II": nsga_select_pairs(prices, cfg),
  }

# -----------------------------
# 示例主程序
# -----------------------------
def main():
  csv_path = "123.csv"  # 请替换为你的数据路径
  cfg = PairSelectorConfig(
    bb_window=30,
    min_form_bars=60,
    min_corr=0.2,
    use_log_price=False,
    select_pairs_per_window=5,
    max_candidates_per_method=200,
    adf_lags=1,
    adf_crit=-3.3,
    sdr_market_mode="mean",
    sdr_market_symbol=None,
    ga_pop=60,
    ga_gen=40,
    ga_cxp=0.8,
    ga_mutp=0.2,
    ga_pairs_per_chrom=5,
    nsga_pop=80,
    nsga_gen=60,
    nsga_pairs_per_chrom=5,
    score_takeprofit_z=2.0,
    score_close_to_sma=True,
    score_fee_bps=0.0,
    seed=42
  )
  prices = load_prices_from_csv(csv_path)
  results = select_pairs_all_methods(prices, cfg)
  for k, v in results.items():
    print(f"[{k}]")
    for a, b, beta in v:
      print(f"  ({a}, {b}) beta={beta:.3f}")
    print()

if __name__ == "__main__":
  main()