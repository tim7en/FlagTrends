"""
Tori Trades Strategy – Safety Line & Breakout Structure Detection
=================================================================

Safety Line:
    A horizontal price level confirmed by ≥ MIN_TOUCHES independent touch
    events. Each "touch" is a discrete approach-and-reaction group (consecutive
    bars in the zone count as ONE touch). The more touches, the stronger the
    level. Acts as the entry trigger / invalidation reference for a breakout.

Breakout Structure:
    Consolidation building near a safety line, evidenced by ATR compression
    (current ATR below its historical baseline). Price is coiling; energy is
    building for a directional break. Each structure is scored 0–10.

Touch:
    A group of consecutive bars whose high–low range overlaps within TOUCH_PCT
    of the level price. Deduplication ensures a multi-day sideways grind near
    the level counts as a single touch, not dozens.

Forming Soon:
    True when price is within APPROACH_PCT of a safety line. Higher scores
    (via ATR compression + touch recency) indicate more imminent structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# ── Strategy parameters ───────────────────────────────────────────────────────
PIVOT_WINDOW      = 5      # bars each side for swing-high / swing-low detection
CLUSTER_PCT       = 1.0    # merge levels within 1 % into one zone  (percent)
TOUCH_PCT         = 0.4    # within 0.4 % of the level = a "touch"  (percent)
MIN_TOUCHES       = 2      # minimum touches to qualify as a safety line
ATR_PERIOD        = 14
HIST_ATR_BARS     = 50     # bars used for historical ATR baseline
APPROACH_PCT      = 3.0    # "forming soon" if price within 3 % of level (percent)
COMPRESSION_RATIO = 0.85   # ATR_current / ATR_hist < this → compressed
MAX_LEVELS        = 8      # keep top-N levels per symbol

__all__ = [
    "KeyLevel",
    "ToriAnalysis",
    "analyze",
    "MIN_TOUCHES",
    "APPROACH_PCT",
    "COMPRESSION_RATIO",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class KeyLevel:
    price:               float
    level_type:          Literal["support", "resistance"]
    touches:             int
    last_touch_bars_ago: int
    atr_ratio:           float   # current_ATR / historical_ATR
    pct_from_price:      float   # signed %; negative = level is BELOW current price
    is_safety_line:      bool    # touches ≥ MIN_TOUCHES
    is_compressed:       bool    # atr_ratio < COMPRESSION_RATIO
    score:               int     # 0–10 composite quality

    @property
    def abs_pct(self) -> float:
        return abs(self.pct_from_price)

    @property
    def forming_soon(self) -> bool:
        """Price is within APPROACH_PCT of a confirmed safety line."""
        return self.is_safety_line and self.abs_pct <= APPROACH_PCT


@dataclass
class ToriAnalysis:
    symbol:     str
    asset_name: str
    asset_type: str
    ticker:     str
    price:      float
    atr:        float
    atr_ratio:  float
    key_levels: list[KeyLevel] = field(default_factory=list)

    @property
    def support_levels(self) -> list[KeyLevel]:
        return [l for l in self.key_levels if l.level_type == "support"]

    @property
    def resistance_levels(self) -> list[KeyLevel]:
        return [l for l in self.key_levels if l.level_type == "resistance"]

    @property
    def best_forming(self) -> KeyLevel | None:
        forming = [l for l in self.key_levels if l.forming_soon]
        return max(forming, key=lambda l: l.score) if forming else None

    @property
    def forming_direction(self) -> str:
        bf = self.best_forming
        if bf is None:
            return ""
        return "BULLISH BREAKOUT" if bf.level_type == "support" else "BEARISH BREAKOUT"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _find_pivot_prices(
    high: pd.Series,
    low:  pd.Series,
    window: int = PIVOT_WINDOW,
) -> tuple[list[float], list[float]]:
    """Return (pivot_highs, pivot_lows) price lists."""
    h_arr = high.values
    l_arr = low.values
    ph: list[float] = []
    pl: list[float] = []
    for i in range(window, len(h_arr) - window):
        if h_arr[i] == h_arr[i - window: i + window + 1].max():
            ph.append(float(h_arr[i]))
        if l_arr[i] == l_arr[i - window: i + window + 1].min():
            pl.append(float(l_arr[i]))
    return ph, pl


def _cluster(prices: list[float], pct: float = CLUSTER_PCT) -> list[float]:
    """Merge nearby prices (within pct %) into representative zone means."""
    if not prices:
        return []
    arr = sorted(prices)
    groups: list[list[float]] = [[arr[0]]]
    ratio = 1 + pct / 100
    for p in arr[1:]:
        if p <= groups[-1][-1] * ratio:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [float(np.mean(g)) for g in groups]


def _count_touches(
    level: float,
    high:  pd.Series,
    low:   pd.Series,
    touch_pct: float = TOUCH_PCT,
) -> tuple[int, int]:
    """
    Count discrete touch events (deduplicated) and bars since the latest one.

    Returns (touch_count, bars_since_last_touch).
    Consecutive bars inside the zone count as a single touch.
    """
    ratio = touch_pct / 100
    zone_hi = level * (1 + ratio)
    zone_lo = level * (1 - ratio)
    in_zone   = (low <= zone_hi) & (high >= zone_lo)
    new_touch = in_zone & (~in_zone.shift(1).infer_objects(copy=False).fillna(False))
    count = int(new_touch.sum())
    if count == 0:
        return 0, 9_999
    idxs = np.where(new_touch.values)[0]
    bars_ago = len(high) - 1 - int(idxs[-1])
    return count, bars_ago


def _score(
    touches:   int,
    bars_ago:  int,
    atr_ratio: float,
    abs_pct:   float,   # already in percent
) -> int:
    """Composite quality score 0–10.

    Points breakdown:
      Touches  (max 4):  touches-1, capped at 4   (2T=1, 3T=2, 4T=3, 5T+=4)
      Recency  (max 2):  ≤5 bars=2, ≤20 bars=1
      ATR comp (max 3):  <0.70=3, <0.85=2, <1.00=1
      Proximity(max 1):  within APPROACH_PCT
    """
    s = 0
    s += min(max(touches - 1, 0), 4)                      # touches
    s += 2 if bars_ago <= 5 else (1 if bars_ago <= 20 else 0)  # recency
    s += 3 if atr_ratio < 0.70 else (2 if atr_ratio < 0.85 else (1 if atr_ratio < 1.00 else 0))
    s += 1 if abs_pct <= APPROACH_PCT else 0              # proximity
    return min(s, 10)


# ── Public API ────────────────────────────────────────────────────────────────

def analyze(
    symbol:     str,
    asset_name: str,
    asset_type: str,
    ticker:     str,
    df:         pd.DataFrame,
) -> ToriAnalysis:
    """Run Tori Trades analysis on a single symbol's OHLCV DataFrame."""
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    close = df["Close"].squeeze()
    price = float(close.iloc[-1])

    atr_s       = _atr(high, low, close, ATR_PERIOD)
    current_atr = float(atr_s.iloc[-ATR_PERIOD:].mean())
    hist_atr    = float(atr_s.iloc[-HIST_ATR_BARS:].mean()) if len(atr_s) >= HIST_ATR_BARS else current_atr
    atr_ratio   = (current_atr / hist_atr) if hist_atr > 0 else 1.0

    ph, pl = _find_pivot_prices(high, low)
    supports    = [p for p in _cluster(pl) if p < price]
    resistances = [p for p in _cluster(ph) if p > price]

    levels: list[KeyLevel] = []

    for lp in supports:
        touches, bars_ago = _count_touches(lp, high, low)
        pct = (lp - price) / price * 100
        levels.append(KeyLevel(
            price               = round(lp, 4),
            level_type          = "support",
            touches             = touches,
            last_touch_bars_ago = bars_ago,
            atr_ratio           = round(atr_ratio, 3),
            pct_from_price      = round(pct, 2),
            is_safety_line      = touches >= MIN_TOUCHES,
            is_compressed       = atr_ratio < COMPRESSION_RATIO,
            score               = _score(touches, bars_ago, atr_ratio, abs(pct)),
        ))

    for lp in resistances:
        touches, bars_ago = _count_touches(lp, high, low)
        pct = (lp - price) / price * 100
        levels.append(KeyLevel(
            price               = round(lp, 4),
            level_type          = "resistance",
            touches             = touches,
            last_touch_bars_ago = bars_ago,
            atr_ratio           = round(atr_ratio, 3),
            pct_from_price      = round(pct, 2),
            is_safety_line      = touches >= MIN_TOUCHES,
            is_compressed       = atr_ratio < COMPRESSION_RATIO,
            score               = _score(touches, bars_ago, atr_ratio, abs(pct)),
        ))

    levels.sort(key=lambda l: -l.score)
    levels = levels[:MAX_LEVELS]

    return ToriAnalysis(
        symbol     = symbol,
        asset_name = asset_name,
        asset_type = asset_type,
        ticker     = ticker,
        price      = round(price, 4),
        atr        = round(current_atr, 4),
        atr_ratio  = round(atr_ratio, 3),
        key_levels = levels,
    )
