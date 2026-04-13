"""
Formats and outputs TrendSignal results to console and/or CSV.
"""

from __future__ import annotations

import datetime
from analyzer import TrendSignal

import pandas as pd


_COLS_CONSOLE = [
    "Symbol", "Name", "Type", "Price", "SMA50", "SMA200",
    "RSI", "Score", "Prior Trend", "Direction", "Signals",
]

_DIR_ORDER = {
    "BULLISH BREAK": 0,
    "WEAK BULLISH":  1,
    "NEUTRAL":       2,
    "WEAK BEARISH":  3,
    "BEARISH BREAK": 4,
}


def _to_df(signals: list[TrendSignal]) -> pd.DataFrame:
    rows = []
    for s in signals:
        rows.append({
            "Symbol":      s.symbol,
            "Name":        s.asset_name[:28],
            "Type":        s.asset_type,
            "Ticker":      s.ticker,
            "Price":       s.price,
            "SMA20":       s.sma20 or "",
            "SMA50":       s.sma50 or "",
            "SMA200":      s.sma200 or "",
            "RSI":         s.rsi or "",
            "EMA Cross":   s.ema_cross,
            "Price/MA":    s.price_ma_break,
            "Donchian":    s.donchian_break,
            "Score":       s.score,
            "Direction":   s.direction,
            "Prior Trend": s.prior_trend,
            "Signals":     s.signals_desc,
        })
    return pd.DataFrame(rows)


def print_report(
    signals: list[TrendSignal],
    min_score: int = 1,
) -> None:
    """Print a grouped console report of flagged symbols."""
    flagged = [s for s in signals if abs(s.score) >= min_score]
    if not flagged:
        print(f"\nNo signals found (min_score={min_score}).")
        return

    flagged.sort(key=lambda s: (-abs(s.score), _DIR_ORDER.get(s.direction, 2)))

    bullish = [s for s in flagged if s.score > 0]
    bearish = [s for s in flagged if s.score < 0]

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    width = 110
    print(f"\n{'═' * width}")
    print(f"  TREND BREAK SCANNER  |  {now}  |  {len(flagged)} signals  (min_score ≥ {min_score})")
    print(f"{'═' * width}")

    if bullish:
        print(f"\n  ▲  BULLISH BREAKS / WEAK BULLISH  ({len(bullish)})\n")
        df = _to_df(bullish)[_COLS_CONSOLE]
        print(df.to_string(index=False))

    if bearish:
        print(f"\n  ▼  BEARISH BREAKS / WEAK BEARISH  ({len(bearish)})\n")
        df = _to_df(bearish)[_COLS_CONSOLE]
        print(df.to_string(index=False))

    print(f"\n{'═' * width}\n")


def save_csv(signals: list[TrendSignal], path: str) -> None:
    """Save full signal table (all symbols) to CSV."""
    df = _to_df(signals)
    df.sort_values(["Score"], ascending=False, inplace=True)
    df.to_csv(path, index=False)
    print(f"Results saved → {path}")
