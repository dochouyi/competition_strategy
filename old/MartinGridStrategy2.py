from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence import Trade
from pandas import DataFrame
from datetime import datetime
import talib

class MartinGridStrategy2(IStrategy):
    position_adjustment_enable = True  # 启用自动加仓
    max_entry_position_adjustment = 3
    max_dca_multiplier = 5.5

    timeframe = '1m'
    minimal_roi = {"0": 0.1}
    stoploss = -0.4
    trailing_stop = False
    process_only_new_candles = False
    use_exit_signal = False
    startup_candle_count = 50
    can_short = True

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        return 10


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = talib.RSI(dataframe['close'], timeperiod=14)
        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe['rsi'] < 30, 'enter_long'] = 1
        dataframe.loc[dataframe['rsi'] > 70, 'enter_short'] = 1
        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe['rsi'] > 70, 'exit_long'] = 1
        dataframe.loc[dataframe['rsi'] < 30, 'exit_short'] = 1
        return dataframe



    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:

        return proposed_stake / self.max_dca_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs
                              ) -> float | None | tuple[float | None, str | None]:
        if trade.has_open_orders:
            return None

        entry_count = trade.nr_of_successful_entries
        if entry_count >= self.max_entry_position_adjustment:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        count_of_entries = trade.nr_of_successful_entries

        try:
            # This returns first order stake size
            stake_amount = filled_entries[0].stake_amount_filled
            # This then calculates current safety order size
            stake_amount = stake_amount * (1 + (count_of_entries * 0.025))
            return stake_amount, "1/nrd_increase"
        except Exception as exception:
            return None

        return None


