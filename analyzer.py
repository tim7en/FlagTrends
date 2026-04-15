"""
Trend-break detection logic.

Three independent signals are checked for each symbol:

  1. EMA-cross   – EMA20 crossed EMA50 in the last *signal_window* trading days.
  2. Price/SMA   – Close crossed the SMA50 in the last *signal_window* days.
  3. Donchian    – Today's close is a new N-day high or low (channel breakout).

Each signal returns +1 (bullish) / -1 (bearish) / 0 (none).
The composite *score* is the sum (-3 … +3).

A prior-trend context is also determined so you can judge whether a bullish
signal is a reversal of a prior downtrend (more significant) or a continuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Timeframe definitions (trading bars) ─────────────────────────────────────

TIMEFRAMES: dict[str, int] = {
    "2M":  42,
    "4M":  84,
    "6M": 126,
    "1Y": 252,
}


@dataclass
class TimeframeTrend:
    label:      str
    bars:       int
    direction:  str    # BULLISH / BEARISH / NEUTRAL / N/A
    change_pct: float  # price change % over the window
    above_sma:  bool | None  # close > mean(close) for the window


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class TrendSignal:
    symbol:            str
    asset_name:        str
    asset_type:        str
    ticker:            str
    price:             float
    sma20:             float | None
    sma50:             float | None
    sma200:            float | None
    rsi:               float | None
    ema_cross:         int    # +1 / -1 / 0
    price_ma_break:    int    # +1 / -1 / 0
    donchian_break:    int    # +1 / -1 / 0
    score:             int    # sum of the three signals
    direction:         str    # BULLISH BREAK / BEARISH BREAK / WEAK BULLISH / WEAK BEARISH / NEUTRAL
    prior_trend:       str    # UP / DOWN / SIDEWAYS / UNKNOWN
    signals_desc:      str    # human-readable list of triggered signals
    week52_high:       float | None = None
    week52_low:        float | None = None
    pct_from_52w_high: float | None = None  # ≤0: how far below the high
    pct_from_52w_low:  float | None = None  # ≥0: how far above the low
    ytd_change_pct:    float | None = None  # % change since Jan 1 of current year
    timeframes:        dict[str, TimeframeTrend] = field(default_factory=dict)


# ── Indicators ────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _sign_cross(series: pd.Series, window: int) -> int:
    """
    Check whether *series* (a spread / difference) changed sign within the
    last *window* bars.

    Returns:
        +1  if the most-recent value is **positive** AND a sign change occurred
        -1  if the most-recent value is **negative** AND a sign change occurred
         0  otherwise (no recent cross or not enough data)
    """
    tail = series.iloc[-(window + 1):].dropna()
    if len(tail) < 2:
        return 0
    signs = np.sign(tail)
    if (signs != signs.shift(1)).iloc[1:].any():  # at least one sign change
        last = signs.iloc[-1]
        if last > 0:
            return 1
        if last < 0:
            return -1
    return 0


# ── Multi-timeframe trend ─────────────────────────────────────────────────────

def analyze_timeframes(close: pd.Series) -> dict[str, TimeframeTrend]:
    """
    For each timeframe window (2M / 4M / 6M / 1Y), compute a simple trend
    direction based on the percentage price change and whether the current
    close is above the window's SMA.
    """
    results: dict[str, TimeframeTrend] = {}
    for label, bars in TIMEFRAMES.items():
        if len(close) < bars:
            results[label] = TimeframeTrend(
                label=label, bars=bars,
                direction="N/A", change_pct=0.0, above_sma=None,
            )
            continue
        segment   = close.iloc[-bars:]
        current   = float(segment.iloc[-1])
        start     = float(segment.iloc[0])
        sma_val   = float(segment.mean())
        chg       = (current - start) / start * 100.0 if start != 0 else 0.0
        above_sma = current > sma_val
        if chg > 2.0:
            direction = "BULLISH"
        elif chg < -2.0:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        results[label] = TimeframeTrend(
            label=label, bars=bars,
            direction=direction, change_pct=round(chg, 2), above_sma=above_sma,
        )
    return results


# ── Signal detectors ─────────────────────────────────────────────────────────

def detect_ema_cross(
    close: pd.Series,
    short: int = 20,
    medium: int = 50,
    window: int = 5,
) -> int:
    """EMA20 crossed EMA50 in the last *window* days."""
    ema_s = close.ewm(span=short, adjust=False).mean()
    ema_m = close.ewm(span=medium, adjust=False).mean()
    return _sign_cross(ema_s - ema_m, window)


def detect_price_ma_break(
    close: pd.Series,
    ma_period: int = 50,
    window: int = 5,
) -> int:
    """Price crossed SMA50 in the last *window* days."""
    sma = close.rolling(ma_period).mean()
    return _sign_cross(close - sma, window)


def detect_donchian_break(close: pd.Series, period: int = 20) -> int:
    """
    Today's close exceeds the *period*-day high / low (excluding today).

    Returns +1 / -1 / 0.
    """
    if len(close) < period + 1:
        return 0
    # Use the [period] bars immediately before the current bar
    prior = close.iloc[-(period + 1):-1]
    current = close.iloc[-1]
    if pd.isna(current):
        return 0
    if current > prior.max():
        return 1
    if current < prior.min():
        return -1
    return 0


def determine_prior_trend(
    close: pd.Series,
    sma50_period: int = 50,
    sma200_period: int = 200,
    lookback: int = 10,
) -> str:
    """
    Determine the trend that was in place *before* the recent signal window.
    Uses the bar at index -(lookback + signal_window) as the reference point.
    """
    sma50 = close.rolling(sma50_period).mean()
    sma200 = close.rolling(sma200_period).mean()

    ref = -(lookback + 5)
    if len(close) < abs(ref):
        return "UNKNOWN"

    p = float(close.iloc[ref])
    s50 = float(sma50.iloc[ref]) if not pd.isna(sma50.iloc[ref]) else None
    s200 = float(sma200.iloc[ref]) if not pd.isna(sma200.iloc[ref]) else None

    if s50 is None or s200 is None:
        return "UNKNOWN"

    if p > s50 > s200:
        return "UP"
    if p < s50 < s200:
        return "DOWN"
    return "SIDEWAYS"


# ── Main entry point ─────────────────────────────────────────────────────────

def analyze(
    symbol: str,
    asset_name: str,
    asset_type: str,
    ticker: str,
    df: pd.DataFrame,
    short_ma: int = 20,
    medium_ma: int = 50,
    long_ma: int = 200,
    signal_window: int = 5,
    donchian_period: int = 20,
) -> TrendSignal:
    """Compute all indicators and signals for a single symbol."""
    close = df["Close"].squeeze()  # ensure 1-D Series

    # ── Indicators ──
    sma20_val  = close.rolling(short_ma).mean().iloc[-1]
    sma50_val  = close.rolling(medium_ma).mean().iloc[-1]
    sma200_val = close.rolling(long_ma).mean().iloc[-1]
    price_val  = close.iloc[-1]
    rsi_val    = _rsi(close).iloc[-1]

    def _fmt(v: float | None) -> float | None:
        return round(float(v), 4) if v is not None and not pd.isna(v) else None

    # ── Signals ──
    ema_cross      = detect_ema_cross(close, short_ma, medium_ma, signal_window)
    price_ma_break = detect_price_ma_break(close, medium_ma, signal_window)
    donchian       = detect_donchian_break(close, donchian_period)
    score          = ema_cross + price_ma_break + donchian
    prior_trend    = determine_prior_trend(close, medium_ma, long_ma)

    # ── 52-week high / low ──
    w52_bars = min(252, len(close))
    w52_slice = close.iloc[-w52_bars:]
    week52_high_val = float(w52_slice.max())
    week52_low_val  = float(w52_slice.min())
    price_f         = _fmt(price_val) or 0.0
    pct_from_high   = round((price_f - week52_high_val) / week52_high_val * 100, 2) if week52_high_val else None
    pct_from_low    = round((price_f - week52_low_val)  / week52_low_val  * 100, 2) if week52_low_val  else None

    # ── YTD return ──
    import datetime as _dt
    ytd_chg: float | None = None
    if close.index.dtype.kind == 'M':
        try:
            jan1 = pd.Timestamp(_dt.date.today().year, 1, 1)
            # Match timezone of the index to avoid tz-naive vs tz-aware comparison
            if close.index.tz is not None:
                jan1 = jan1.tz_localize(close.index.tz)
            ytd_slice = close[close.index >= jan1]
            if len(ytd_slice) >= 2:
                ytd_start = float(ytd_slice.iloc[0])
                if ytd_start:
                    ytd_chg = round((price_f - ytd_start) / ytd_start * 100, 2)
        except Exception:
            pass

    # ── Multi-timeframe trends ──
    tf_trends = analyze_timeframes(close)

    # ── Direction label ──
    if score >= 2:
        direction = "BULLISH BREAK"
    elif score <= -2:
        direction = "BEARISH BREAK"
    elif score == 1:
        direction = "WEAK BULLISH"
    elif score == -1:
        direction = "WEAK BEARISH"
    else:
        direction = "NEUTRAL"

    # ── Human-readable signal list ──
    parts: list[str] = []
    if ema_cross != 0:
        parts.append(f"EMA{short_ma}/EMA{medium_ma} cross {'UP' if ema_cross > 0 else 'DOWN'}")
    if price_ma_break != 0:
        parts.append(f"Price/SMA{medium_ma} break {'UP' if price_ma_break > 0 else 'DOWN'}")
    if donchian != 0:
        parts.append(f"Donchian{donchian_period} break {'UP' if donchian > 0 else 'DOWN'}")
    signals_desc = " | ".join(parts) if parts else "-"

    return TrendSignal(
        symbol=symbol,
        asset_name=asset_name,
        asset_type=asset_type,
        ticker=ticker,
        price=price_f,
        sma20=_fmt(sma20_val),
        sma50=_fmt(sma50_val),
        sma200=_fmt(sma200_val),
        rsi=round(float(rsi_val), 1) if not pd.isna(rsi_val) else None,
        ema_cross=ema_cross,
        price_ma_break=price_ma_break,
        donchian_break=donchian,
        score=score,
        direction=direction,
        prior_trend=prior_trend,
        signals_desc=signals_desc,
        week52_high=round(week52_high_val, 4),
        week52_low=round(week52_low_val, 4),
        pct_from_52w_high=pct_from_high,
        pct_from_52w_low=pct_from_low,
        ytd_change_pct=ytd_chg,
        timeframes=tf_trends,
    )
