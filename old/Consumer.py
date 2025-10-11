from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame

class ConsumerStrategy(IStrategy):
    timeframe = "5m"
    process_only_new_candles = False  # 消费者必需

    # 预期从 producer 合并来的列（用于空数据时的占位）
    _columns_to_expect = ['rsi_default', 'tema_default', 'bb_middleband_default']

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        timeframe = self.timeframe

        producer_dataframe, _ = self.dp.get_producer_df(pair)

        if producer_dataframe is not None and not producer_dataframe.empty:
            merged = merge_informative_pair(
                base_dataframe=dataframe,
                informative_dataframe=producer_dataframe,
                base_timeframe=timeframe,
                informative_timeframe=timeframe,
                append_timeframe=False,
                suffix="default",
                ffill=False  # 若想直接用 producer 的信号，建议 ffill=False
            )
            return merged
        else:
            # 没拿到数据时，创建占位列
            for col in self._columns_to_expect:
                if col not in dataframe.columns:
                    dataframe[col] = 0
            return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 使用 producer 的指标进行决策（与 producer 等价的条件示例）
        cond = (
            (dataframe['rsi_default'] > 50) &
            (dataframe['tema_default'] <= dataframe['bb_middleband_default']) &
            (dataframe['tema_default'] > dataframe['tema_default'].shift(1)) &
            (dataframe['volume'] > 0)
        )
        dataframe.loc[cond, 'enter_long'] = 1

        # 如果你想直接用上游信号（需在 merge 时 ffill=False 且 producer 产生 enter_long）
        # dataframe['enter_long'] = dataframe.get('enter_long_default', 0)

        return dataframe
