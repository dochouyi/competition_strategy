import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime

Base = declarative_base()

class PairSignal(Base):
    __tablename__ = "pair_signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_time = Column(DateTime)
    a = Column(String)
    b = Column(String)
    beta = Column(Float)
    bb_window = Column(Integer)
    bb_k = Column(Float)
    OPEN_TIMEOUT_MIN = Column(Integer)
    CLOSE_TIMEOUT_MIN = Column(Integer)
    status = Column(String)
    entry_trade_time = Column(DateTime)
    side = Column(String)
    a_entry_price = Column(Float)
    b_entry_price = Column(Float)
    bb_entry_mid = Column(Float)
    bb_entry_up = Column(Float)
    bb_entry_lo = Column(Float)
    notional = Column(Float)
    exit_trade_time = Column(DateTime)
    a_exit_price = Column(Float)
    b_exit_price = Column(Float)
    bb_exit_mid = Column(Float)
    exit_reason = Column(String)
    pnl_percent = Column(Float)
    max_drawdown_percent = Column(Float)
    peak_pnl_percent = Column(Float)
    hold_duration_hours = Column(Float)

def main():
    db_path = "./pair_trading_data/pair_trading.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    closed_signals = session.query(PairSignal).filter(PairSignal.status == "closed").all()
    print(f"Number of closed trades: {len(closed_signals)}")

    df = pd.DataFrame([{
        "pnl_percent": sig.pnl_percent,
        "max_drawdown_percent": sig.max_drawdown_percent,
        "hold_duration_hours": sig.hold_duration_hours
    } for sig in closed_signals])

    # 将收益率低于-2%的情况设为-2%
    df.loc[df["pnl_percent"] < -0.01, "pnl_percent"] = -0.01

    df = df.dropna(subset=["pnl_percent"])

    total_trades = len(df)
    max_profit = df["pnl_percent"].max()
    max_loss = df["pnl_percent"].min()
    win_rate = (df["pnl_percent"] > 0).mean()
    avg_pnl = df["pnl_percent"].mean()
    max_drawdown = df["max_drawdown_percent"].max() if "max_drawdown_percent" in df else None

    # Simple total return
    total_return = df["pnl_percent"].sum()

    if total_trades > 1:
        pnl_series = df["pnl_percent"]
        sharpe_ratio = pnl_series.mean() / pnl_series.std() * (total_trades ** 0.5)
    else:
        sharpe_ratio = None

    print(f"Total trades: {total_trades}")
    print(f"Max profit: {max_profit*100:.4f}%")
    print(f"Max loss: {max_loss*100:.4f}%")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Average PnL: {avg_pnl*100:.4f}%")
    print(f"Max drawdown: {max_drawdown*100:.4f}%")
    print(f"Simple total return: {total_return*100:.4f}%")
    print(f"Sharpe ratio: {sharpe_ratio:.4f}" if sharpe_ratio is not None else "Sharpe ratio: N/A")

    plt.figure(figsize=(10,6))
    plt.hist(df["pnl_percent"], bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Trade Returns (pnl_percent)")
    plt.xlabel("Return")
    plt.ylabel("Number of Trades")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
