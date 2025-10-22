from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame


class DummyStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1h'
    minimal_roi = {"0": 1}
    stoploss = -0.5
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = False
    can_short = True


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe
