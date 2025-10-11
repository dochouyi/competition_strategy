from freqtrade.strategy import IStrategy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from pandas import DataFrame
from ta.volatility import BollingerBands
import logging

logger = logging.getLogger(__name__)

# ============= 配置数据类（集中配置） =============
@dataclass
class SignalConfig:
    timeframe: str = "5m"
    form_period: int = 200
    bb_window: int = 30
    bb_k: float = 3.0
    min_form_bars: int = 60
    recompute_every: int = 20
    min_corr: float = 0.2
    use_log_price: bool = False

    # 信号参数（保留以便需要调试指标，但不再用于发布）
    z_stop: float = 6.0
    z_exit_to_sma: bool = True
    takeprofit_exit_buffer_z: float = 0.2
    enforce_profit_on_cross: bool = True
    min_take_profit_pct: float = 0.2

    select_pairs_per_window: int = 1
    max_candidates_per_method: int = 20

    # 冷却与去抖（当前仅保留字段，不用于发布）
    cooldown_bars_open: int = 10
    cooldown_bars_close: int = 3


# ============= 工具函数（配对、距离、beta等） =============
def estimate_beta_on_window(a: pd.Series, b: pd.Series, use_log_price=True, min_len: int = 30) -> float:
    a = np.log(a.dropna()) if use_log_price else a.dropna().copy()
    b = np.log(b.dropna()) if use_log_price else b.dropna().copy()
    idx = a.index.intersection(b.index)
    a = a.loc[idx]; b = b.loc[idx]
    if len(a) < min_len:
        return 1.0
    # 简化 OLS 斜率 ≈ cov / var
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


def euclidean_distance_method(prices: Dict[str, pd.Series], cfg: SignalConfig, topk: int) -> List[Tuple[str, str, float]]:
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


def compute_bb(spread: pd.Series, window: int, k: float):
    bb = BollingerBands(close=spread, window=window, window_dev=k, fillna=False)
    sma = bb.bollinger_mavg()
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    sd = ((upper - sma) / k).clip(lower=1e-6)
    return sma, upper, lower, sd


# ============= 策略类（仅重算并打印，不发布、不下单） =============
class PairSignalStrategy2(IStrategy):
    """
    使用方法：
    1) 将本文件放在 user_data/strategies/ 下
    2) 启动：freqtrade trade --strategy PairSignalStrategy --config user_data/config.json --dry-run
       本策略不会下单，也不会向Redis发布，只会在周期性重算后打印候选配对
    """
    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 260  # >= form_period + bb_window
    can_short = True

    # 禁用ROI/止损（无实际下单）
    minimal_roi = {"0": 10}
    stoploss = -0.99

    # 内部状态
    cfg = SignalConfig()
    _last_recompute_index: int = 0
    _current_pairs: List[Tuple[str, str, float]] = []  # [(A,B,beta)]
    _main_leg_map: Dict[str, Dict] = {}               # A-> {mate, beta}, B-> {mate, beta_adj}
    _lock = threading.Lock()

    # 冷却控制：记录每个组最后一次动作所在bar索引（当前不使用）
    _pair_last_action_bar: Dict[str, int] = {}

    def informative_pairs(self):
        # 允许白名单中的所有对，确保可跨pair取数据
        pairs = [(p, self.timeframe) for p in self.dp.current_whitelist()]
        return pairs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 覆盖timeframe
        self.timeframe = self.cfg.timeframe

    @staticmethod
    def _pair_key(a: str, b: str) -> str:
        return "|".join(sorted([a, b]))

    def _recompute_candidates(self, df_map: Dict[str, DataFrame]) -> List[Tuple[str, str, float]]:
        # 取最近 form_period 的close
        closes: Dict[str, pd.Series] = {}
        for pair, df in df_map.items():
            if "close" not in df.columns:
                continue
            s = df["close"].dropna()
            if s.shape[0] >= self.cfg.min_form_bars:
                closes[pair] = s.iloc[-self.cfg.form_period:]
        if len(closes) < 2:
            return []

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
            if len(idx) < max(self.cfg.min_form_bars, self.cfg.bb_window):
                continue
            corr = sA.loc[idx].corr(sB.loc[idx])
            if (corr is None) or (not np.isfinite(corr)) or (corr < self.cfg.min_corr):
                continue
            filtered.append((a, b, beta))
            used.add(a); used.add(b)
            if len(filtered) >= self.cfg.select_pairs_per_window:
                break
        return filtered

    def _build_maps(self, pairs: List[Tuple[str, str, float]]):
        m = {}
        for a, b, beta in pairs:
            m[a] = {"mate": b, "beta": float(beta)}
            # 反向beta（简化，避免除零）
            bbeta = (1.0 / beta) if (beta and abs(beta) > 1e-9) else 1.0
            m[b] = {"mate": a, "beta": float(bbeta)}
        self._main_leg_map = m

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        在每个pair上都会被调用。我们只在一个“主循环”里执行重算逻辑以避免重复。
        这里选择按字母序第一个pair作为主循环触发点。
        """
        pair = metadata["pair"]

        # 选择一个锚定pair作为主循环触发器
        wl = sorted(self.dp.current_whitelist())
        if not wl or pair != wl[0]:
            return dataframe

        # 收集所有pair的df
        df_map: Dict[str, DataFrame] = {}
        for p in wl:
            try:
                dfp = self.dp.get_pair_dataframe(p, self.timeframe)
                df_map[p] = dfp.copy()
            except Exception:
                continue

        cur_len = len(dataframe)
        if self._last_recompute_index == 0:
            self._last_recompute_index = cur_len

        # 周期性重算候选
        pairs = self._recompute_candidates(df_map)
        self._current_pairs = pairs
        self._build_maps(pairs)
        self._last_recompute_index = cur_len
        # 仅打印输出选择结果
        if pairs:
            self.logger.info(f"[PAIR] Recomputed candidates ({len(pairs)}): {pairs}")
        else:
            self.logger.info("[PAIR] Recomputed candidates but none selected.")

        # 不再生成开/平信号，也不发布
        return dataframe

    # 禁用下单
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe
