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


    # 用户可调参数
    coeff = 1.0      # Multiplier
    AP = 14          # Common Period
    novolumedata = True  # True: 用RSI, False: 用MFI


    can_short = True


    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:

        return 10

    def alpha_trend(self, df: DataFrame) -> DataFrame:
        # 计算ATR
        tr = ta.TRANGE(df)
        atr = ta.SMA(tr, timeperiod=self.AP)
        # 计算upT和downT
        upT = df['low'] - atr * self.coeff
        downT = df['high'] + atr * self.coeff

        # 计算信号条件
        if self.novolumedata:
            signal = ta.RSI(df['close'], timeperiod=self.AP) >= 50
        else:
            hlc3 = (df['high'] + df['low'] + df['close']) / 3
            signal = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=self.AP) >= 50

        # 初始化AlphaTrend
        alpha = np.zeros(len(df))
        for i in range(1, len(df)):
            prev = alpha[i-1]
            if signal[i]:
                alpha[i] = prev if upT[i] < prev else upT[i]
            else:
                alpha[i] = prev if downT[i] > prev else downT[i]



        df['AlphaTrend'] = alpha
        df['AlphaTrend_2'] = df['AlphaTrend'].shift(2)
        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.alpha_trend(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 多头开仓：AlphaTrend上穿AlphaTrend_2
        dataframe.loc[
            (dataframe['AlphaTrend'] > dataframe['AlphaTrend_2']) &
            (dataframe['AlphaTrend'].shift(1) <= dataframe['AlphaTrend_2'].shift(1)),
            'enter_long'
        ] = 1

        # 空头开仓：AlphaTrend下穿AlphaTrend_2
        dataframe.loc[
            (dataframe['AlphaTrend'] < dataframe['AlphaTrend_2']) &
            (dataframe['AlphaTrend'].shift(1) >= dataframe['AlphaTrend_2'].shift(1)),
            'enter_short'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 多头平仓：AlphaTrend下穿AlphaTrend_2
        dataframe.loc[
            (dataframe['AlphaTrend'] < dataframe['AlphaTrend_2']) &
            (dataframe['AlphaTrend'].shift(1) >= dataframe['AlphaTrend_2'].shift(1)),
            'exit_long'
        ] = 1

        # # 空头平仓：AlphaTrend上穿AlphaTrend_2
        dataframe.loc[
            (dataframe['AlphaTrend'] > dataframe['AlphaTrend_2']) &
            (dataframe['AlphaTrend'].shift(1) <= dataframe['AlphaTrend_2'].shift(1)),
            'exit_short'
        ] = 1

        return dataframe
