from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas as pd
import logging
from methods.sdr import SDRSelector

logger = logging.getLogger(__name__)


class PairSignalStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    minimal_roi = {"0": 1}
    stoploss = -0.5
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = False
    can_short = True

    CLOSE_SNAPSHOT_LEN = 100  # 可按需修改，比如 50、100、200
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
        return [(p, self.timeframe) for p in sorted(set(self.dp.current_whitelist()) | EXTRA_PAIRS)]

    def get_all_coin_close(self):
        inf_pairs = self.informative_pairs()
        target_pairs = [p for p, tf in inf_pairs]
        closes_list = {}
        df_first = self.dp.get_pair_dataframe(pair=target_pairs[0], timeframe=self.timeframe)
        timestamps = df_first['date'].tail(self.CLOSE_SNAPSHOT_LEN).tolist()
        for p in target_pairs:
            df_p = self.dp.get_pair_dataframe(pair=p, timeframe=self.timeframe)
            closes = df_p['close'].tail(self.CLOSE_SNAPSHOT_LEN).tolist()
            closes_list[p] = closes

        df_closes = pd.DataFrame(closes_list, index=pd.to_datetime(timestamps))
        df_closes.index.name = "date"

        close_prices = {col: df_closes[col] for col in df_closes.columns}

        return close_prices

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        close_prices=self.get_all_coin_close()


        matching_pairs = self.selector.select_pairs(close_prices)
        logger.info(matching_pairs)

        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe