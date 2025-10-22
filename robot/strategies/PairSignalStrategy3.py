from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timezone

from methods.sdr import SDRSelector
from metrics import compute_bb

# SQLAlchemy ORM
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

logger = logging.getLogger(__name__)
Base = declarative_base()

# --------------------------
# ORM Models
# --------------------------
class PairSignal(Base):
    __tablename__ = "pair_signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, index=True)            # 5m产生时间（UTC，对齐到5m）
    a = Column(String, index=True)
    b = Column(String, index=True)
    beta = Column(Float)
    bb_window = Column(Integer)
    bb_k = Column(Float)
    alive_minutes = Column(Integer, default=100)         # 20个5m = 100个1m
    opened = Column(Boolean, default=False)              # 是否最终入场
    expired = Column(Boolean, default=False)             # 是否入场超时
    comment = Column(String, default="")

    trades = relationship("PairTrade", back_populates="signal", cascade="all, delete-orphan")

    __table_args__ = (
        Index('ix_pair_signal_ab_time', 'a', 'b', 'created_at'),
    )


class PairTrade(Base):
    __tablename__ = "pair_trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer, ForeignKey("pair_signals.id", ondelete="CASCADE"), index=True)

    # 入场信息
    entry_at = Column(DateTime, index=True, nullable=True)
    side = Column(String)  # "shortA_longB" or "longA_shortB"
    a_entry_price = Column(Float, nullable=True)   # 1m
    b_entry_price = Column(Float, nullable=True)   # 1m
    bb_entry_mid = Column(Float, nullable=True)    # 5m中轨
    bb_entry_up = Column(Float, nullable=True)     # 5m上轨
    bb_entry_lo = Column(Float, nullable=True)     # 5m下轨

    # 出场信息
    exit_at = Column(DateTime, index=True, nullable=True)
    a_exit_price = Column(Float, nullable=True)    # 1m
    b_exit_price = Column(Float, nullable=True)    # 1m
    bb_exit_mid = Column(Float, nullable=True)     # 5m中轨
    exit_reason = Column(String, nullable=True)    # "to_mid" or "force_exit"

    # 绩效
    pnl_abs = Column(Float, nullable=True)         # 绝对收益（基于单位权重 1 和 beta）
    max_drawdown_abs = Column(Float, default=0.0)  # 持仓期绝对最大回撤
    alive_minutes = Column(Integer, default=500)   # 100个5m = 500个1m

    signal = relationship("PairSignal", back_populates="trades")

    __table_args__ = (
        Index('ix_pair_trade_signal_entry', 'signal_id', 'entry_at'),
    )

# --------------------------
# Strategy
# --------------------------
class PairSignalStrategy3(IStrategy):
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

    # ORM & in-memory state
    _engine = None
    _Session = None
    _session = None

    # 最近一次5m边界时间与匹配结果缓存
    _last_5m_bucket = None
    _last_matching_pairs = []  # [(a,b,beta), ...]
    # 活跃未入场信号: key=(a,b,created_at_iso) -> dict state
    _active_pairs = {}
    # 活跃持仓交易: key=trade_id -> dict state
    _active_trades = {}

    # 配置参数
    BB_WINDOW = 30
    BB_K = 3.5
    OPEN_TIMEOUT_5M = 20   # 20个5m
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
        # 固定将数据库写入 /freqtrade/user_data
        base_dir = "/freqtrade/user_data"
        os.makedirs(base_dir, exist_ok=True)
        db_path = os.path.join(base_dir, "pair_trading.db")
        self._engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        self._session = self._Session()
        logger.info(f"SQLite DB initialized at: {db_path}")

    # --------------------------
    # Helpers
    # --------------------------
    def _is_5m_bucket(self, dt: datetime) -> datetime:
        # 返回该分钟对应的5m桶开始时间（UTC）
        minute = dt.minute - (dt.minute % 5)
        bucket = dt.replace(second=0, microsecond=0, minute=minute)
        return bucket

    def _now_from_df(self, dataframe: DataFrame) -> datetime:
        # 以最新bar时间为“当前时间”（UTC）
        return pd.to_datetime(dataframe['date'].iloc[-1]).to_pydatetime().replace(tzinfo=timezone.utc)

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
            return None, ma, up, lo

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

    def _pnl_abs(self, side: str, beta: float, a_in: float, b_in: float, a_out: float, b_out: float):
        if None in (a_in, b_in, a_out, b_out):
            return None
        if side == "longA_shortB":
            return (a_out - a_in) - beta * (b_out - b_in)
        elif side == "shortA_longB":
            return -(a_out - a_in) + beta * (b_out - b_in)
        return None

    # 更新持仓期最大回撤（绝对值）
    def _update_max_drawdown(self, trade_state: dict, curr_pnl: float):
        peak = trade_state.get("peak_pnl", 0.0)
        if curr_pnl > peak:
            trade_state["peak_pnl"] = curr_pnl
        drawdown = trade_state.get("peak_pnl", 0.0) - curr_pnl
        if drawdown > trade_state.get("max_drawdown_abs", 0.0):
            trade_state["max_drawdown_abs"] = drawdown

    # --------------------------
    # Strategy main
    # --------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self._init_db()
        session = self._session

        # 最新时间（UTC）
        now_utc = self._now_from_df(dataframe)
        now_bucket = self._is_5m_bucket(now_utc)

        # 读取价格
        close_prices_5m = self._get_all_coin_close_5m()
        close_prices_1m = self._get_all_coin_close_1m()

        # 每5分钟生成新的matching_pairs并入库
        if self._last_5m_bucket is None or now_bucket > self._last_5m_bucket:
            self._last_5m_bucket = now_bucket
            matching_pairs = self.selector.select_pairs(close_prices_5m)  # [(a,b,beta),...]
            self._last_matching_pairs = matching_pairs
            logger.info(f"New 5m bucket {now_bucket}, pairs: {matching_pairs}")

            # 将所有新matching_pair加入数据库与活跃队列
            for (a, b, beta) in matching_pairs:
                sig = PairSignal(
                    created_at=now_bucket,
                    a=a, b=b, beta=float(beta),
                    bb_window=self.BB_WINDOW, bb_k=self.BB_K,
                    alive_minutes=self.OPEN_TIMEOUT_MIN,
                    opened=False, expired=False, comment=""
                )
                session.add(sig)
                session.flush()  # 获取sig.id

                key = (a, b, now_bucket.isoformat())
                self._active_pairs[key] = {
                    "signal_id": sig.id,
                    "a": a, "b": b, "beta": float(beta),
                    "created_at": now_bucket,
                    "minutes_alive": 0
                }
            session.commit()

        # 逐分钟处理：尝试入场
        to_remove_active_pairs = []
        for key, st in list(self._active_pairs.items()):
            a, b = st["a"], st["b"]
            beta = st["beta"]
            st["minutes_alive"] = st.get("minutes_alive", 0) + 1

            # 超时未入场，标记expired
            if st["minutes_alive"] > self.OPEN_TIMEOUT_MIN:
                sig = session.get(PairSignal, st["signal_id"])
                if sig and not sig.opened:
                    sig.expired = True
                    sig.comment = "open_timeout"
                    session.add(sig)
                to_remove_active_pairs.append(key)
                continue

            # 数据可用性检查
            if (a not in close_prices_5m) or (b not in close_prices_5m) or (a not in close_prices_1m) or (b not in close_prices_1m):
                continue

            A_5m = close_prices_5m[a]
            B_5m = close_prices_5m[b]
            A_1m_latest = close_prices_1m[a].iloc[-1]
            B_1m_latest = close_prices_1m[b].iloc[-1]

            side, ma, up, lo = self._try_open(A_5m, B_5m, beta, A_1m_latest, B_1m_latest)
            if side is None:
                # 还未触发
                continue

            # 入场，建立PairTrade
            sig = session.get(PairSignal, st["signal_id"])
            if sig is None or sig.opened or sig.expired:
                # 已被其他流程处理
                to_remove_active_pairs.append(key)
                continue

            trade = PairTrade(
                signal_id=sig.id,
                entry_at=now_utc,
                side=side,
                a_entry_price=float(A_1m_latest),
                b_entry_price=float(B_1m_latest),
                bb_entry_mid=float(ma) if ma is not None else None,
                bb_entry_up=float(up) if up is not None else None,
                bb_entry_lo=float(lo) if lo is not None else None,
                alive_minutes=self.CLOSE_TIMEOUT_MIN,
                max_drawdown_abs=0.0
            )
            session.add(trade)
            session.flush()

            # 更新signal状态
            sig.opened = True
            session.add(sig)

            # 注册活跃持仓状态
            self._active_trades[trade.id] = {
                "signal_id": sig.id,
                "a": a, "b": b, "beta": beta,
                "side": side,
                "a_entry": float(A_1m_latest),
                "b_entry": float(B_1m_latest),
                "entry_at": now_utc,
                "minutes_alive": 0,
                "max_drawdown_abs": 0.0,
                "peak_pnl": 0.0
            }

            to_remove_active_pairs.append(key)

        if to_remove_active_pairs:
            for key in to_remove_active_pairs:
                self._active_pairs.pop(key, None)
            session.commit()

        # 逐分钟处理：尝试出场或强制出场
        to_remove_trades = []
        for trade_id, ts in list(self._active_trades.items()):
            a, b, beta = ts["a"], ts["b"], ts["beta"]
            ts["minutes_alive"] = ts.get("minutes_alive", 0) + 1

            # 数据可用性检查
            if (a not in close_prices_5m) or (b not in close_prices_5m) or (a not in close_prices_1m) or (b not in close_prices_1m):
                continue

            A_5m = close_prices_5m[a]
            B_5m = close_prices_5m[b]
            A_1m_latest = close_prices_1m[a].iloc[-1]
            B_1m_latest = close_prices_1m[b].iloc[-1]

            # 浮动盈亏，更新回撤
            curr_pnl = self._pnl_abs(ts["side"], beta, ts["a_entry"], ts["b_entry"], float(A_1m_latest), float(B_1m_latest))
            if curr_pnl is not None:
                self._update_max_drawdown(ts, curr_pnl)

            reason = None
            ma_mid = None

            # 正常出场判定
            res, ma_mid = self._try_close(A_5m, B_5m, beta, A_1m_latest, B_1m_latest, ts["side"])
            if res == "close":
                reason = "to_mid"

            # 强制出场判定
            if reason is None and ts["minutes_alive"] > self.CLOSE_TIMEOUT_MIN:
                reason = "force_exit"

            if reason is not None:
                # 写回数据库
                trade = session.get(PairTrade, trade_id)
                if trade and trade.exit_at is None:
                    trade.exit_at = now_utc
                    trade.a_exit_price = float(A_1m_latest)
                    trade.b_exit_price = float(B_1m_latest)
                    trade.bb_exit_mid = float(ma_mid) if ma_mid is not None else None
                    trade.exit_reason = reason
                    # 计算最终pnl与最大回撤
                    trade.pnl_abs = self._pnl_abs(ts["side"], beta, trade.a_entry_price, trade.b_entry_price,
                                                  trade.a_exit_price, trade.b_exit_price)
                    trade.max_drawdown_abs = float(ts.get("max_drawdown_abs", 0.0))
                    session.add(trade)
                to_remove_trades.append(trade_id)

        if to_remove_trades:
            for tid in to_remove_trades:
                self._active_trades.pop(tid, None)
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