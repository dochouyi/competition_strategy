from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timezone, timedelta
from enum import Enum
from methods.sdr import SDRSelector
from metrics import compute_bb
# SQLAlchemy ORM
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Index, and_, select
)
from sqlalchemy.orm import declarative_base, sessionmaker


logger = logging.getLogger(__name__)
Base = declarative_base()

# --------------------------
# 枚举：信号状态
# --------------------------
class PairSignalStatus(Enum):
    OBSERVE = "observe"  # 观察中（未入场、未超时）
    TIMEOUT = "timeout"  # 观察超时（未入场即超时）
    OPENED = "opened"    # 已入场、未出场
    CLOSED = "closed"    # 已出场

# --------------------------
# ORM Model
# --------------------------
class PairSignal(Base):
    __tablename__ = "pair_signals"
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 信号（5m生成时间，UTC对齐）
    created_time = Column(DateTime, index=True)     # 5m产生时间（UTC，对齐到5m）
    a = Column(String, index=True)
    b = Column(String, index=True)
    beta = Column(Float)

    # 策略参数快照
    bb_window = Column(Integer)
    bb_k = Column(Float)
    OPEN_TIMEOUT_MIN = Column(Integer)              # 生成时策略配置：观察期分钟数
    CLOSE_TIMEOUT_MIN = Column(Integer)             # 生成时策略配置：持仓期分钟数

    # 状态机：observe / timeout / opened / closed
    status = Column(String, index=True, default=PairSignalStatus.OBSERVE.value)

    # 入场信息
    entry_trade_time = Column(DateTime, index=True, nullable=True)
    side = Column(String, nullable=True)  # "shortA_longB" or "longA_shortB"
    a_entry_price = Column(Float, nullable=True)   # 1m
    b_entry_price = Column(Float, nullable=True)   # 1m
    bb_entry_mid = Column(Float, nullable=True)    # 5m中轨
    bb_entry_up = Column(Float, nullable=True)     # 5m上轨
    bb_entry_lo = Column(Float, nullable=True)     # 5m下轨

    # 新增：组合名义成本（入场时基于价格与 beta 计算）
    notional = Column(Float, nullable=True)

    # 出场信息
    exit_trade_time = Column(DateTime, index=True, nullable=True)
    a_exit_price = Column(Float, nullable=True)    # 1m
    b_exit_price = Column(Float, nullable=True)    # 1m
    bb_exit_mid = Column(Float, nullable=True)     # 5m中轨
    exit_reason = Column(String, nullable=True)    # "to_mid" or "force_exit"

    # 绩效（注意：以下字段语义已改为“百分比（小数）”）
    pnl_percent = Column(Float, nullable=True)         # 最终百分比收益（如 0.012 表示 1.2%）
    max_drawdown_percent = Column(Float, nullable=True)  # 持仓期最大回撤百分比（小数）
    peak_pnl_percent = Column(Float, nullable=True)          # 持仓期最大浮盈百分比（小数）

    # 新增：持仓持续时间（单位小时）
    hold_duration_hours = Column(Float, nullable=True)  #从入场到出场的时间

    __table_args__ = (
        Index('ix_pair_signal_ab_time', 'a', 'b', 'created_time'),
        Index('ix_pair_signal_entry', 'entry_trade_time'),
        Index('ix_pair_signal_exit', 'exit_trade_time'),
        Index('ix_pair_signal_status', 'status'),
    )

# --------------------------
# Strategy
# --------------------------
class PairSignalStrategy(IStrategy):
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

    # ORM
    _engine = None
    _SessionFactory = None
    _db_session = None

    _last_5m_bucket = None

    # 配置参数
    BB_WINDOW = 30
    BB_K = 3
    OPEN_TIMEOUT_5M = 50   # 50个5m
    CLOSE_TIMEOUT_5M = 100 # 100个5m

    # 换算为分钟
    OPEN_TIMEOUT_MIN = OPEN_TIMEOUT_5M * 5
    CLOSE_TIMEOUT_MIN = CLOSE_TIMEOUT_5M * 5

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

    # --------------------------
    # ORM init
    # --------------------------
    def _init_db(self):
        if self._engine is not None:
            return
        base_dir = "/freqtrade/user_data"
        os.makedirs(base_dir, exist_ok=True)
        db_path = os.path.join(base_dir, "pair_trading.db")
        self._engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
        Base.metadata.create_all(self._engine)
        self._SessionFactory = sessionmaker(bind=self._engine)
        self._db_session = self._SessionFactory()
        logger.info(f"SQLite DB initialized at: {db_path}")

    # --------------------------
    # Helpers
    # --------------------------
    def _get_5m_bucket(self, dt: datetime) -> datetime:
        minute = dt.minute - (dt.minute % 5)
        bucket = dt.replace(second=0, microsecond=0, minute=minute)
        return bucket

    def _get_last_date_utc_1m(self, dataframe: DataFrame) -> datetime:
        dt = pd.to_datetime(dataframe['date'].iloc[-1])
        if getattr(dt, 'tzinfo', None) is not None:
            dt = dt.tz_convert('UTC').tz_localize(None)
        return dt.to_pydatetime()

    def _get_all_coin_close_5m(self):
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
        return {col: df_closes[col] for col in df_closes.columns}

    def _get_all_coin_close_1m(self):
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
        return {col: df_closes[col] for col in df_closes.columns}

    def _calc_spread_bb_5m(self, A_5m: pd.Series, B_5m: pd.Series, beta: float):
        spread_5m = A_5m - beta * B_5m
        sma_5m, upper_5m, lower_5m, sd = compute_bb(spread_5m, self.BB_WINDOW, self.BB_K)
        return spread_5m, sma_5m, upper_5m, lower_5m

    def _try_open(self, A_5m: pd.Series, B_5m: pd.Series, beta: float,
                  latest_A_1m: float, latest_B_1m: float):
        spread_5m, sma_5m, up_5m, lo_5m = self._calc_spread_bb_5m(A_5m, B_5m, beta)
        sp = latest_A_1m - beta * latest_B_1m
        ma = sma_5m.iloc[-1]
        up = up_5m.iloc[-1]
        lo = lo_5m.iloc[-1]
        if np.isnan(ma) or np.isnan(up) or np.isnan(lo):
            return None, None, None, None
        if sp >= up:
            return "shortA_longB", ma, up, lo
        elif sp <= lo:
            return "longA_shortB", ma, up, lo
        else:
            return None, None, None, None

    def _try_close(self, A_5m: pd.Series, B_5m: pd.Series, beta: float,
                   latest_A_1m: float, latest_B_1m: float, side: str):
        spread_5m, sma_5m, _, _ = self._calc_spread_bb_5m(A_5m, B_5m, beta)
        sp = latest_A_1m - beta * latest_B_1m
        ma = sma_5m.iloc[-1]
        if np.isnan(ma):
            return None, ma
        if side == "shortA_longB":
            if sp <= ma:
                return "close", ma
        elif side == "longA_shortB":
            if sp >= ma:
                return "close", ma
        return None, ma

    # 组合名义成本（百分比分母）
    def _notional(self, side: str, beta: float, a_in: float, b_in: float) -> float:
        # 统一用两腿名义和作为基准
        return float(a_in) + float(beta) * float(b_in)

    def _pnl_abs(self, side: str, beta: float, a_in: float, b_in: float, a_out: float, b_out: float):
        if side == "longA_shortB":
            return (a_out - a_in) - beta * (b_out - b_in)
        elif side == "shortA_longB":
            return -(a_out - a_in) + beta * (b_out - b_in)
        return None

    def _pnl_pct(self, side: str, beta: float, a_in: float, b_in: float, a_out: float, b_out: float):
        pnl_abs = self._pnl_abs(side, beta, a_in, b_in, a_out, b_out)
        if pnl_abs is None:
            return None
        notional = self._notional(side, beta, a_in, b_in)
        if notional == 0 or np.isnan(notional):
            return None
        return float(pnl_abs) / float(notional)

    # 计算是否观察期超时
    def _is_observe_timeout(self, sig: PairSignal, now_utc: datetime) -> bool:
        deadline = sig.created_time + timedelta(minutes=int(sig.OPEN_TIMEOUT_MIN))
        return now_utc >= deadline

    # 计算是否持仓期超时
    def _is_hold_timeout(self, sig: PairSignal, now_utc: datetime) -> bool:
        deadline = sig.entry_trade_time + timedelta(minutes=int(sig.CLOSE_TIMEOUT_MIN))
        return now_utc >= deadline

    # --------------------------
    # Strategy main
    # --------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self._init_db()
        session = self._db_session

        # 最新时间（UTC）
        now_utc = self._get_last_date_utc_1m(dataframe)
        now_5m_bucket = self._get_5m_bucket(now_utc)

        # 读取价格
        close_prices_5m = self._get_all_coin_close_5m()
        close_prices_1m = self._get_all_coin_close_1m()

        # 每5分钟生成新的matching_pairs并入库（去重）
        if self._last_5m_bucket is None or now_5m_bucket > self._last_5m_bucket:
            self._last_5m_bucket = now_5m_bucket
            matching_pairs = self.selector.select_pairs(close_prices_5m)  # [(a,b,beta),...]

            logger.info(f"New 5m bucket {now_5m_bucket}, pairs: {matching_pairs}")

            for (a, b, beta) in matching_pairs:
                sig = PairSignal(
                    created_time=now_5m_bucket,
                    a=a, b=b, beta=float(beta),
                    bb_window=self.BB_WINDOW, bb_k=self.BB_K,
                    OPEN_TIMEOUT_MIN=self.OPEN_TIMEOUT_MIN,
                    CLOSE_TIMEOUT_MIN=self.CLOSE_TIMEOUT_MIN,
                    status=PairSignalStatus.OBSERVE.value,
                )
                session.add(sig)
            session.commit()

        # 逐分钟处理（纯 ORM）
        # 1) 观察阶段 -> 入场 or 超时
        pending_signals = session.execute(
            select(PairSignal).where(
                PairSignal.status == PairSignalStatus.OBSERVE.value
            )
        ).scalars().all()

        for sig in pending_signals:
            a, b, beta = sig.a, sig.b, sig.beta

            # 超时判断（通过时间推导）
            if self._is_observe_timeout(sig, now_utc):
                # 若尚未入场则超时
                sig.status = PairSignalStatus.TIMEOUT.value
                continue

            # 尝试入场
            A_5m = close_prices_5m[a]
            B_5m = close_prices_5m[b]
            A_1m_latest = float(close_prices_1m[a].iloc[-1])
            B_1m_latest = float(close_prices_1m[b].iloc[-1])

            side, ma, up, lo = self._try_open(A_5m, B_5m, beta, A_1m_latest, B_1m_latest)
            if side is None:
                continue
            else:
                # 入场
                sig.status = PairSignalStatus.OPENED.value
                sig.entry_trade_time = now_utc
                sig.side = side
                sig.a_entry_price = A_1m_latest
                sig.b_entry_price = B_1m_latest
                sig.bb_entry_mid = float(ma)
                sig.bb_entry_up = float(up)
                sig.bb_entry_lo = float(lo)
                # 计算并保存名义成本
                entry_notional = self._notional(side, beta, sig.a_entry_price, sig.b_entry_price)
                sig.notional = float(entry_notional)
                # 初始化百分比峰值与回撤
                sig.peak_pnl_percent = 0.0            # 百分比（小数）
                sig.max_drawdown_percent = 0.0    # 百分比（小数）

        session.commit()

        # 2) 持仓阶段 -> 正常/强制出场
        opened_signals = session.execute(
            select(PairSignal).where(
                PairSignal.status == PairSignalStatus.OPENED.value
            )
        ).scalars().all()

        for sig in opened_signals:
            a, b, beta = sig.a, sig.b, sig.beta

            A_5m = close_prices_5m[a]
            B_5m = close_prices_5m[b]
            A_1m_latest = float(close_prices_1m[a].iloc[-1])
            B_1m_latest = float(close_prices_1m[b].iloc[-1])

            # 更新浮盈百分比 -> peak -> 回撤百分比
            curr_pnl_pct = self._pnl_pct(sig.side, beta, sig.a_entry_price, sig.b_entry_price, A_1m_latest, B_1m_latest)
            if curr_pnl_pct is not None:
                prev_peak_pct = float(sig.peak_pnl_percent or 0.0)
                sig.peak_pnl_percent = max(prev_peak_pct, float(curr_pnl_pct))   # 百分比
                dd_pct = float(sig.peak_pnl_percent) - float(curr_pnl_pct)       # 百分比
                sig.max_drawdown_percent = max(float(sig.max_drawdown_percent or 0.0), dd_pct)  # 百分比


            # 正常出场判定
            res, ma_mid = self._try_close(A_5m, B_5m, beta, A_1m_latest, B_1m_latest, sig.side)
            reason = "to_mid" if res == "close" else None

            if curr_pnl_pct<0.0015:
                reason=None

            # 强制退出判定（通过时间推导）
            if self._is_hold_timeout(sig, now_utc):
                reason = "force_exit"

            # 触发退出
            if reason is not None and sig.status == PairSignalStatus.OPENED.value:
                sig.status = PairSignalStatus.CLOSED.value
                sig.exit_trade_time = now_utc
                sig.a_exit_price = A_1m_latest
                sig.b_exit_price = B_1m_latest
                sig.bb_exit_mid = float(ma_mid)
                sig.exit_reason = reason
                # 存储最终百分比盈亏
                final_pnl_pct = self._pnl_pct(sig.side, beta, sig.a_entry_price, sig.b_entry_price, A_1m_latest, B_1m_latest)
                sig.pnl_percent = float(final_pnl_pct) if final_pnl_pct is not None else None  # 百分比

                delta_sec = (sig.exit_trade_time - sig.entry_trade_time).total_seconds()
                sig.hold_duration_hours = float(delta_sec) / 3600.0


        session.commit()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe
