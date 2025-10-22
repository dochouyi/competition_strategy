from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas as pd
import logging
from methods.sdr import SDRSelector
from metrics import compute_bb
import numpy as np

logger = logging.getLogger(__name__)


class PairSignalStrategy_op(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1m'
    minimal_roi = {"0": 1}
    stoploss = -0.5
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = False
    can_short = True

    CLOSE_SNAPSHOT_LEN = 100
    startup_candle_count = CLOSE_SNAPSHOT_LEN

    selector = SDRSelector()

    def informative_pairs(self):
        EXTRA_PAIRS = {
            "ETH/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT", "SOL/USDT:USDT",
            "DOGE/USDT:USDT", "TRX/USDT:USDT", "ADA/USDT:USDT", "LINK/USDT:USDT",
            "SUI/USDT:USDT", "HYPE/USDT:USDT", "XLM/USDT:USDT", "AVAX/USDT:USDT",
            "BCH/USDT:USDT", "HBAR/USDT:USDT", "LTC/USDT:USDT", "TON/USDT:USDT",
            "DOT/USDT:USDT", "XMR/USDT:USDT", "WLFI/USDT:USDT", "UNI/USDT:USDT",
            "AAVE/USDT:USDT", "ENA/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT",
            "TAO/USDT:USDT", "ASTER/USDT:USDT", "ETC/USDT:USDT", "ONDO/USDT:USDT",
            "IP/USDT:USDT", "ZEC/USDT:USDT", "WLD/USDT:USDT", "POL/USDT:USDT",
        }
        pairs = sorted(set(self.dp.current_whitelist()) | EXTRA_PAIRS)
        timeframes = ['1m', '5m']
        return [(p, tf) for p in pairs for tf in timeframes]

    def get_all_coin_close_5m(self):
        inf_pairs = self.informative_pairs()
        target_pairs = [p for p, tf in inf_pairs if tf == '5m']
        closes_list = {}
        df_first = self.dp.get_pair_dataframe(pair=target_pairs[0], timeframe='5m')
        timestamps = df_first['date'].tail(self.CLOSE_SNAPSHOT_LEN).tolist()
        for p in target_pairs:
            df_p = self.dp.get_pair_dataframe(pair=p, timeframe='5m')
            closes = df_p['close'].tail(self.CLOSE_SNAPSHOT_LEN).tolist()
            closes_list[p] = closes
        df_closes = pd.DataFrame(closes_list, index=pd.to_datetime(timestamps))
        df_closes.index.name = "date"
        close_prices = {col: df_closes[col] for col in df_closes.columns}

        return close_prices

    def get_all_coin_close_1m(self):
        inf_pairs = self.informative_pairs()
        target_pairs = [p for p, tf in inf_pairs if tf == '1m']
        closes_list = {}
        df_first = self.dp.get_pair_dataframe(pair=target_pairs[0], timeframe='1m')
        timestamps = df_first['date'].tail(self.CLOSE_SNAPSHOT_LEN).tolist()
        for p in target_pairs:
            df_p = self.dp.get_pair_dataframe(pair=p, timeframe='1m')
            closes = df_p['close'].tail(self.CLOSE_SNAPSHOT_LEN).tolist()
            closes_list[p] = closes
        df_closes = pd.DataFrame(closes_list, index=pd.to_datetime(timestamps))
        df_closes.index.name = "date"
        close_prices = {col: df_closes[col] for col in df_closes.columns}
        return close_prices

    def _try_open(
            self,
            A_5m: pd.Series,
            B_5m: pd.Series,
            beta: float,
            label: str,
            latest_A_1m: float,
            latest_B_1m: float,
            bb_window: int,
            bb_k: float
    ):
        # 计算5m级别spread和布林带
        spread_5m = A_5m - beta * B_5m
        sma_5m, upper_5m, lower_5m, sd = compute_bb(spread_5m, bb_window, bb_k)

        # 用1m最新收盘价计算当前spread
        sp = latest_A_1m - beta * latest_B_1m
        ma = sma_5m.iloc[-1]
        up = upper_5m.iloc[-1]
        lo = lower_5m.iloc[-1]

        if np.isnan(ma) or np.isnan(up) or np.isnan(lo):
            return None
        if sp >= up:
            side = "shortA_longB"
        elif sp <= lo:
            side = "longA_shortB"
        else:
            return None

        # 你可以返回side或者其他状态
        return side

    def _try_close(
            self,
            A_5m: pd.Series,
            B_5m: pd.Series,
            beta: float,
            label: str,
            latest_A_1m: float,
            latest_B_1m: float,
            bb_window: int,
            bb_k: float,
            side: str  # 当前持仓方向
    ):
        # 计算5m级别spread和布林带
        spread_5m = A_5m - beta * B_5m
        sma_5m, upper_5m, lower_5m, sd = compute_bb(spread_5m, bb_window, bb_k)

        # 用1m最新收盘价计算当前spread
        sp = latest_A_1m - beta * latest_B_1m
        ma = sma_5m.iloc[-1]

        if np.isnan(ma):
            return None

        # 平仓逻辑：spread到达中轨
        if side == "shortA_longB":
            # 空A多B，spread向下回归，遇到中轨平仓
            if sp <= ma:
                return "close"
        elif side == "longA_shortB":
            # 多A空B，spread向上回归，遇到中轨平仓
            if sp >= ma:
                return "close"
        return None



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        close_prices_5m=self.get_all_coin_close_5m()
        close_prices_1m = self.get_all_coin_close_1m()  # 1m收盘价

        matching_pairs = self.selector.select_pairs(close_prices_5m)
        logger.info(matching_pairs)

        for (a, b, beta) in matching_pairs:
            A_5m = close_prices_5m[a]
            B_5m = close_prices_5m[b]
            label = f"{a}-{b}"

            # 获取最新的1m收盘价（Series最后一个值）
            A_1m = close_prices_1m[a].iloc[-1]
            B_1m = close_prices_1m[b].iloc[-1]

            state = self._try_open(A_5m, B_5m, beta, label,
                                   latest_A_1m=A_1m, latest_B_1m=B_1m,
                                   bb_window=30, bb_k=3.5)


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe