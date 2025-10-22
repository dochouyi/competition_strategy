import numpy as np
import pandas as pd
import math
from typing import Tuple
from typing import Dict, List, Tuple, Set
from typing import Optional

def pct_change_series(s: pd.Series) -> pd.Series:
    return s.pct_change().dropna()

# 皮尔逊相关系数，值越接近 1，说明涨跌同步性越强；越接近 -1，说明走势相反；接近 0，则相关性弱。
def pearson_returns_correlation(a: pd.Series, b: pd.Series) -> float:
    ra, rb = pct_change_series(a), pct_change_series(b)
    return float(ra.corr(rb))


# 用于计算 OLS（普通最小二乘法）回归的 Beta 系数，
# a: 目标资产的价格（或收益）
# b: 基准资产的价格（或收益）
# use_log_price: 是否对价格取对数，默认为 True
# Beta 是衡量a相对于b的波动性的指标，Beta = 1.5 表示，当b上涨1%，a上涨1.5%
def estimate_beta_ols(a: pd.Series, b: pd.Series, use_log_price=True) -> float:
    # 取对数或原值
    a = np.log(a) if use_log_price else a.copy()
    b = np.log(b) if use_log_price else b.copy()

    # 去除NaN和inf
    mask = np.isfinite(a.values) & np.isfinite(b.values)
    x = b.values[mask]
    y = a.values[mask]
    if x.size < 2:
        return 1.0

    # 标准化数据（中心化）
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (vx**2).sum()   #计算分母
    if denom <= 0:
        return 1.0

    beta = (vx * vy).sum() / denom  #计算beta值
    if not np.isfinite(beta):
        beta = 1.0
    return float(np.clip(beta, 0.1, 10.0))  #   防止极端值


#度量两个序列的相似性，消除了量纲和尺度的影响
def unified_scaled_distance(a: pd.Series, b: pd.Series) -> float:

    A = a.values.reshape(-1, 1)
    B = b.values.reshape(-1, 1)
    X = np.vstack([A, B])
    mu = X.mean()
    sd = X.std() if X.std() > 0 else 1.0

    #对 A 和 B 都用同一个均值和标准差做标准化
    As = (A - mu) / sd
    Bs = (B - mu) / sd

    return float(np.linalg.norm(As.flatten() - Bs.flatten()))   # 计算L2范数


# ADF（Augmented Dickey-Fuller）单位根检验的核心部分，用于判断一个时间序列是否平稳
# 输入：一组时间序列数据 x，以及滞后阶数 lags。
# 输出：该序列的ADF检验t统计量（float类型），用于后续判断序列是否平稳。
# 输出的数值越小（越负），越倾向于拒绝单位根假设，即序列更可能是平稳的。

def adf_test_simple(x: np.ndarray, lags: int = 1) -> float:
    x = x.astype(float)
    dx = np.diff(x)
    x_1 = x[:-1]
    n = len(dx)
    if n - lags - 1 <= 3:
        return np.nan
    Y = dx[lags:]
    X_cols = []
    X_cols.append(np.ones(len(Y)))
    X_cols.append(x_1[lags:])
    for i in range(1, lags + 1):
        X_cols.append(dx[lags - i: -i])
    X = np.vstack(X_cols).T
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return np.nan
    beta_hat = XtX_inv @ (X.T @ Y)
    residuals = Y - X @ beta_hat
    sigma2 = (residuals @ residuals) / (len(Y) - X.shape[1])
    var_beta = sigma2 * XtX_inv
    phi = beta_hat[1]
    se_phi = math.sqrt(var_beta[1, 1]) if var_beta[1, 1] > 0 else np.inf
    t_phi = phi / se_phi if se_phi > 0 else np.nan
    return float(t_phi) #返回ADF检验的t统计量（不是p值）


# Engle-Granger 协整回归，如果 a、b 是两个资产的价格序列，beta 反映了它们长期均衡关系的比例
# 用最小二乘法对两个时间序列做线性回归（通常用于协整检验中的 Engle-Granger 方法）。
# 返回回归斜率 beta（代表协整关系的强度），以及残差序列。
# 支持对数变换，并对异常情况做了容错处理（数据太少、矩阵不可逆等）。
# 残差序列反映了 a 和 b 在协整关系下的“偏离程度”。如果两者协整，残差应该是均值回归的、平稳的。
def engle_granger_beta(a: pd.Series, b: pd.Series, use_log_price=True) -> Tuple[float, pd.Series]:

    a = np.log(a) if use_log_price else a.copy()
    b = np.log(b) if use_log_price else b.copy()
    x = b.values; y = a.values
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


#mode：市场参考模式。mean 表示所有资产价格的均值，symbol 表示用指定资产作为市场参考。
#生成市场参考收益率序列。
def market_series(prices: Dict[str, pd.Series], mode: str, symbol: Optional[str]) -> pd.Series:
    dfs = pd.concat(prices.values(), axis=1, join="inner")
    dfs.columns = list(prices.keys())
    if mode == "symbol" and symbol is not None and symbol in prices:
        base = prices[symbol]
        return pct_change_series(base)
    else:
        px_mean = dfs.mean(axis=1)
        return pct_change_series(px_mean)


# 计算两个资产对在市场参考下的“gamma差”。计算两个资产对市场参考收益率的“gamma差”。
# 这里的 gamma 实际上就是 OLS（最小二乘法）回归得到的 beta 系数，表示资产对市场的敏感度。最终返回的是两个资产对市场敏感度的差值。
def sdr_gamma_diff(a: pd.Series, b: pd.Series, market_r: pd.Series) -> float:
    ra, rb = pct_change_series(a), pct_change_series(b)

    ri_a = ra.values
    ri_b = rb.values
    rm = market_r.values
    vx = rm - rm.mean()   #对市场参考收益率序列做中心化（减去均值），让回归更稳健。
    def ols_beta(y, x_centered):
        vy = y - y.mean()
        denom = (x_centered**2).sum()
        if denom <= 0:
            return 0.0
        return float((x_centered * vy).sum() / denom)

    #分别对 a、b 的收益率序列和市场参考收益率做回归，得到各自的 beta（敏感度）。
    gamma_a = ols_beta(ri_a, vx)
    gamma_b = ols_beta(ri_b, vx)
    #返回两个资产对市场敏感度的差值。
    return float(gamma_a - gamma_b)



# 这个函数用于评估一对资产（a, b）在某个beta配对下的布林带套利表现，输出两个指标：
# score：标准化后的累计收益（pnl/标准差均值）；maxdd_n：标准化后的最大回撤

def bollinger_spread_score(a: pd.Series, b: pd.Series, beta: float, window: int, take_z: float,
                            fee_bps: float) -> Tuple[float, float]:
    A = a.values
    B = b.values
    S = A - beta * B
    sma = pd.Series(S).rolling(window).mean().values
    std = pd.Series(S).rolling(window).std(ddof=0).values
    up = sma + take_z * std
    dn = sma - take_z * std

    pnl = 0.0   #资金
    peak = 0.0  #曾获得的最大未平仓利润值
    maxdd = 0.0    #最大回撤

    pos = 0     #持仓pos
    entry = 0.0     #入场价格
    fee = fee_bps / 10000.0 #手续费

    for t in range(window, len(S)):
        s = S[t]; mu = sma[t]; u = up[t]; d = dn[t]
        if np.isnan(mu) or np.isnan(u) or np.isnan(d):
            continue
        if pos == 0:
            if s > u:   #上轨的情况
                pos = -1; entry = s; pnl -= abs(s) * fee
            elif s < d:    #下轨的情况
                pos = 1; entry = s; pnl -= abs(s) * fee
        else:
            close_cond = abs(s - mu) < 1e-12
            if close_cond:
                pnl += (s - entry) * pos
                pnl -= abs(s) * fee
                pos = 0
            else:
                float_pnl = pnl + (s - entry) * pos
                peak = max(peak, float_pnl)
                maxdd = max(maxdd, peak - float_pnl)

    if pos != 0 and not np.isnan(sma[-1]):
        pnl += (S[-1] - entry) * pos
        pnl -= abs(S[-1]) * fee
        pos = 0

    std_mean = np.nanmean(std)
    norm = std_mean if (std_mean is not None and std_mean > 1e-12) else 1.0
    score = pnl / norm
    maxdd_n = maxdd / norm if norm > 0 else maxdd
    return float(score), float(maxdd_n)



# 返回四个Series：中轨、上轨、下轨、标准差
# bb_k是标准差倍数
# std_clip标准差的下限（clip），用于防止标准差过小导致布林带过于收窄
def compute_bb(spread: pd.Series, bb_window: int, bb_k: float):
    ma = spread.rolling(bb_window).mean()
    std = spread.rolling(bb_window).std().clip(lower=1e-6)
    upper = ma + bb_k * std
    lower = ma - bb_k * std
    return ma, upper, lower, std