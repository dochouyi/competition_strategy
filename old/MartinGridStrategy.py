from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from datetime import datetime
from freqtrade.persistence import Trade
import logging

logger = logging.getLogger(__name__)

class MartinGridStrategy(IStrategy):
    """
    Freqtrade 策略：启动后直接入场，最多补仓5次，每次下跌0.5%加仓，每次加仓翻倍。
    """

    # ROI和止损设置
    minimal_roi = {"0": 0.1}
    stoploss = -0.2
    timeframe = '1m'
    startup_candle_count = 20
    process_only_new_candles = True
    use_exit_signal = False
    position_adjustment_enable = True

    # DCA相关参数
    max_entry_position_adjustment = 5  # 最多补仓5次（共6次入场）
    dca_trigger = -0.005  # 每-0.5%触发一次加仓
    dca_multiplier = 1    # 每次加仓

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        return 10

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:
        """
        初始仓位要足够小，预留后续加仓资金
        """
        max_multiplier = sum([self.dca_multiplier ** i for i in range(self.max_entry_position_adjustment + 1)])
        return proposed_stake / max_multiplier

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs
                              ) -> float | None | tuple[float | None, str | None]:
        if trade.has_open_orders:
            return None

        entries = trade.nr_of_successful_entries
        if entries > self.max_entry_position_adjustment:
            return None

        next_trigger = self.dca_trigger * (entries + 1)
        if current_profit > next_trigger:
            return None

        try:
            filled_entries = trade.select_filled_orders(trade.entry_side)
            if not filled_entries:
                logger.error("No filled entries found for trade.")
                return None
            initial_stake = filled_entries[0].stake_amount_filled
            stake_amount = initial_stake * (self.dca_multiplier ** entries)
            return stake_amount, f"dca_{entries}_at_{current_profit:.2%}"
        except Exception as e:
            logger.error(f"Error in adjust_trade_position: {e}")
            return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 无需指标，直接返回
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        启动后直接入场：所有K线都给入场信号
        """
        dataframe['enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        出场信号：这里可以选择永不出场，或自定义出场条件。
        若想永不主动平仓，可全部为0；如需止盈止损交由minimal_roi和stoploss控制。
        """
        dataframe['exit_long'] = 0
        return dataframe
