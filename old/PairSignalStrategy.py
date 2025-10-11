from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


class AlphaTrend(IStrategy):
    INTERFACE_VERSION = 3

    # 参数设置
    timeframe = '1m'
    minimal_roi = {"0": 0.1}  # 止盈10%
    stoploss = -0.4           # 止损40%


    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 50



    can_short = True


    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:

        return 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe
