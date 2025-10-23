import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import warnings
import os
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


def parse_symbol_from_filename(fn: str, timeframe: str = "5m") -> str:

    base = os.path.basename(fn)
    if base.endswith(".json"):
        base = base[:-5]
    flag = f"-{timeframe}-futures"
    if flag in base:
        return base.split(flag)[0]
    return base


def load_freqtrade_dir(dir_path: str, timeframe: str = "5m") -> Dict[str, pd.DataFrame]:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Data directory not found: {dir_path}")
    files = [fn for fn in os.listdir(dir_path) if fn.endswith("-5m-futures.json")]
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


if __name__ == "__main__":
    data_dir = "/home/houyi/crypto/pair_trading/data/"
    print(f"Loading data from: {data_dir}")
    data = load_freqtrade_dir(data_dir)
    pass