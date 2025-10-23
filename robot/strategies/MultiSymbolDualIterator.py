import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# =========================
# 工具函数
# =========================
def load_freqtrade_file(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        raw = json.load(f)
    arr = np.array(raw, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Unexpected data shape in {path}: {arr.shape}")
    df = pd.DataFrame(arr, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce").astype("Int64"), unit="ms", utc=True)
    df = df.dropna(subset=["ts"])
    df.set_index("ts", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[["open","high","low","close","volume"]]


def parse_symbol_from_filename(fn: str, timeframe: str) -> str:
    base = os.path.basename(fn)
    if base.endswith(".json"):
        base = base[:-5]
    flag = f"-{timeframe}-futures"
    if flag in base:
        return base.split(flag)[0]
    return base


def load_freqtrade_dir(dir_path: str, timeframe: str) -> Dict[str, pd.DataFrame]:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Data directory not found: {dir_path}")
    suffix = f"-{timeframe}-futures.json"
    files = [fn for fn in os.listdir(dir_path) if fn.endswith(suffix)]
    files.sort()

    data = {}
    for fn in files:
        full = os.path.join(dir_path, fn)
        sym = parse_symbol_from_filename(full, timeframe)
        try:
            df = load_freqtrade_file(full)
            data[sym] = df
        except Exception as e:
            print(f"Skip {fn}: {e}")
    if not data:
        raise FileNotFoundError("No json found.")
    return data




# =========================
# 新增：多符号双时间框流（每次返回所有符号）
# =========================
class MultiSymbolDualStreamer:
    """
    多符号统一迭代逻辑：
    - 使用全局锚定时间：取所有符号的 5m 第 101 根时间戳的最大值作为 anchor_ts，以确保所有符号有足够的 5m 历史。
    - current_ts 从 anchor_ts 开始，每次推进 1 分钟。
    - 每次调用返回一个 dict: {symbol: (win_1m, win_5m)}。
    - 5m 窗口仅在 5m 边界更新各符号的缓存，否则沿用缓存。
    - 只要所有符号都能提供各自的窗口（1m 必须 100 根；5m 边界处必须 100 根；非边界处缓存有效）就可继续。
    """
    def __init__(self,data_dir):

        data_1m = load_freqtrade_dir(data_dir, timeframe="1m")
        data_5m = load_freqtrade_dir(data_dir, timeframe="5m")

        self.syms =sorted(list(set(data_1m.keys()) & set(data_5m.keys())))
        self.data_1m = {s: data_1m[s] for s in self.syms}
        self.data_5m = {s: data_5m[s] for s in self.syms}

        # 计算每个符号的 5m 第 101 根时间戳
        per_sym_anchor = {}
        for s in self.syms:
            df5 = self.data_5m[s]
            if len(df5.index) < 101:
                raise ValueError(f"符号 {s} 的 5m 数据不足 101 根。")
            per_sym_anchor[s] = df5.index[100]
        # 全局 anchor_ts 取最大值，保证所有符号在该时刻之前都能有至少 100 根 5m
        self.anchor_ts = max(per_sym_anchor.values())
        self.current_ts = self.anchor_ts

        # 初始化每个符号的 5m 缓存窗口（截至 anchor_ts 的最近 100 根）
        self.current_5m_cache: Dict[str, pd.DataFrame] = {}
        for s in self.syms:
            df5 = self.data_5m[s]
            win5 = self._slice_last_n_until_ts(df5, self.anchor_ts, n=100)
            if win5 is None or len(win5) < 100:
                raise ValueError(f"符号 {s} 无法在 anchor_ts 处获得完整的 5m 初始窗口。")
            self.current_5m_cache[s] = win5

    @staticmethod
    def _slice_last_n_until_ts(df: pd.DataFrame, ts: pd.Timestamp, n: int) -> Optional[pd.DataFrame]:
        sub = df.loc[df.index <= ts]
        if len(sub) < n:
            return None
        return sub.iloc[-n:]

    @staticmethod
    def _is_5m_boundary(ts: pd.Timestamp) -> bool:
        minute = ts.minute
        return (minute % 5) == 0

    def get_next(self) -> Optional[Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]]:
        # 准备当次所有符号 1m 窗口
        win_1m_all: Dict[str, pd.DataFrame] = {}
        for s in self.syms:
            df1 = self.data_1m[s]
            win1 = self._slice_last_n_until_ts(df1, self.current_ts, n=100)
            if win1 is None or len(win1) < 100:
                return None
            win_1m_all[s] = win1

        # 5m 更新逻辑：在边界统一更新所有符号，否则沿用缓存
        if self._is_5m_boundary(self.current_ts):
            for s in self.syms:
                df5 = self.data_5m[s]
                win5_new = self._slice_last_n_until_ts(df5, self.current_ts, n=100)
                if win5_new is None or len(win5_new) < 100:
                    return None
                self.current_5m_cache[s] = win5_new

        # 组装输出
        win_5m_all: Dict[str, pd.DataFrame] = {}
        for s in self.syms:
            win5 = self.current_5m_cache.get(s, None)
            if win5 is None or len(win5) < 100:
                return None
            win_5m_all[s] = win5

        # 推进时间
        self.current_ts = self.current_ts + pd.Timedelta(minutes=1)
        return win_1m_all, win_5m_all


if __name__ == "__main__":

    multi_streamer = MultiSymbolDualStreamer()

    for i in range(12):
        result = multi_streamer.get_next()
        if result is None:
            print("No more windows available.")
            break
        win_1m_all, win_5m_all = result
