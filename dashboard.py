"""
FlagTrends — Tori Trades Dashboard
====================================
Run:  streamlit run dashboard.py
"""

from __future__ import annotations

import csv
import datetime
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

import config
from fetcher import fetch_all
from analyzer import (
    TrendSignal as TrendSignalBreak,
    analyze as analyze_trend,
    TIMEFRAMES,
)
from tori import (
    APPROACH_PCT,
    COMPRESSION_RATIO,
    MIN_TOUCHES,
    ToriAnalysis,
    analyze,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FlagTrends",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* forming-soon cards */
.ft-card {
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
    border-left: 4px solid #555;
    background: #161b22;
    font-family: monospace;
}
.ft-bull { border-left-color: #26a69a; }
.ft-bear { border-left-color: #ef5350; }
.ft-sym  { font-size: 1.05rem; font-weight: 700; letter-spacing: 0.03em; }
.ft-sub  { font-size: 0.78rem; color: #9e9e9e; line-height: 1.7; }
.ft-score-badge {
    float: right;
    font-size: 0.8rem;
    background: #2a2a2a;
    border-radius: 4px;
    padding: 1px 7px;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_symbols(path: str = "symbols.csv") -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _normalize_symbols(symbols: list[dict]) -> list[dict]:
    """Ensure each symbol dict has asset_type/asset_name, derive yfinance ticker."""
    for s in symbols:
        if "asset_type" not in s:
            url = s.get("url", "")
            parts = url.strip("/").split("/")
            s["asset_type"] = parts[1].capitalize() if len(parts) > 2 else "Stock"
        if "asset_name" not in s:
            s["asset_name"] = s.get("name", "") or s["symbol"]
    return symbols


@st.cache_data(ttl=300, show_spinner="Fetching & analysing market data…")
def _run(asset_type: str) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """
    Returns:
        rows     – flat list of dicts (one per key level, all symbols)
        raw      – OHLCV DataFrames keyed by Libertex symbol
    """
    symbols = _normalize_symbols(_load_symbols())
    if asset_type != "All":
        symbols = [s for s in symbols if s["asset_type"].lower() == asset_type.lower()]

    sym_ticker = {
        s["symbol"]: config.SYMBOL_MAP.get(s["symbol"], s["symbol"])
        for s in symbols
    }
    name_map = {s["symbol"]: s["asset_name"] for s in symbols}
    type_map = {s["symbol"]: s["asset_type"]  for s in symbols}

    raw = fetch_all(
        sym_ticker,
        period_days=config.LOOKBACK_DAYS,
        use_cache=True,
        workers=config.FETCH_WORKERS,
    )

    rows: list[dict] = []
    for sym, df in raw.items():
        if df is None or len(df) < 220:
            continue
        try:
            ta: ToriAnalysis = analyze(
                symbol     = sym,
                asset_name = name_map.get(sym, sym),
                asset_type = type_map.get(sym, "?"),
                ticker     = sym_ticker[sym],
                df         = df,
            )
        except Exception:
            continue

        for lv in ta.key_levels:
            rows.append({
                "symbol":           ta.symbol,
                "name":             ta.asset_name,
                "asset_type":       ta.asset_type,
                "ticker":           ta.ticker,
                "price":            ta.price,
                "atr":              ta.atr,
                "atr_ratio":        ta.atr_ratio,
                "level":            lv.price,
                "level_type":       lv.level_type,
                "touches":          lv.touches,
                "last_touch":       lv.last_touch_bars_ago,
                "pct_from_price":   lv.pct_from_price,
                "is_safety_line":   lv.is_safety_line,
                "is_compressed":    lv.is_compressed,
                "score":            lv.score,
                "forming_soon":     lv.forming_soon,
                "direction":        ta.forming_direction if lv.forming_soon else "",
            })

    return rows, raw


@st.cache_data(ttl=300, show_spinner="Running multi-timeframe trend scan…")
def _run_trend_scan(asset_type: str) -> list[dict]:
    """
    Returns a flat list of dicts — one row per symbol — containing:
    price, 52w high/low, trend direction for each timeframe (2M/4M/6M/1Y),
    and the daily break signal score.
    """
    symbols = _normalize_symbols(_load_symbols())
    if asset_type != "All":
        symbols = [s for s in symbols if s["asset_type"].lower() == asset_type.lower()]

    sym_ticker = {
        s["symbol"]: config.SYMBOL_MAP.get(s["symbol"], s["symbol"])
        for s in symbols
    }
    name_map = {s["symbol"]: s["asset_name"] for s in symbols}
    type_map = {s["symbol"]: s["asset_type"]  for s in symbols}

    raw = fetch_all(
        sym_ticker,
        period_days=config.LOOKBACK_DAYS,
        use_cache=True,
        workers=config.FETCH_WORKERS,
    )

    rows: list[dict] = []
    for sym, df in raw.items():
        if df is None or len(df) < 60:
            continue
        try:
            ts: TrendSignalBreak = analyze_trend(
                symbol     = sym,
                asset_name = name_map.get(sym, sym),
                asset_type = type_map.get(sym, "?"),
                ticker     = sym_ticker[sym],
                df         = df,
            )
        except Exception:
            continue

        row: dict = {
            "symbol":            ts.symbol,
            "name":              ts.asset_name,
            "asset_type":        ts.asset_type,
            "sector":            config.SECTOR_MAP.get(ts.symbol, "Other"),
            "price":             ts.price,
            "week52_high":       ts.week52_high,
            "week52_low":        ts.week52_low,
            "pct_from_52w_high": ts.pct_from_52w_high,
            "pct_from_52w_low":  ts.pct_from_52w_low,
            "ytd_chg":           ts.ytd_change_pct,
            "daily_score":       ts.score,
            "daily_direction":   ts.direction,
            "rsi":               ts.rsi,
            "prior_trend":       ts.prior_trend,
        }
        for label in TIMEFRAMES:
            tf = ts.timeframes.get(label)
            row[f"tf_{label}_dir"] = tf.direction    if tf else "N/A"
            row[f"tf_{label}_chg"] = tf.change_pct   if tf else 0.0
        rows.append(row)

    return rows


@st.cache_data(ttl=3600, show_spinner="Fetching earnings & analyst estimates…")
def _run_earnings_scan(asset_type: str) -> list[dict]:
    """
    Returns a flat list of dicts — one row per symbol — containing analyst
    price targets, recommendation summary, upcoming earnings date, and EPS
    estimates as reported by yfinance.  Only symbols with at least one data
    point are included.
    """
    symbols = _normalize_symbols(_load_symbols())
    if asset_type != "All":
        symbols = [s for s in symbols if s["asset_type"].lower() == asset_type.lower()]

    sym_ticker = {
        s["symbol"]: config.SYMBOL_MAP.get(s["symbol"], s["symbol"])
        for s in symbols
    }
    name_map = {s["symbol"]: s.get("asset_name", s["symbol"]) for s in symbols}

    def _fetch_one(sym: str) -> dict | None:
        ticker_str = sym_ticker[sym]
        try:
            t = yf.Ticker(ticker_str)

            # ── current price ───────────────────────────────────────────────
            current_price: float | None = None
            try:
                fi = t.fast_info
                current_price = float(fi.last_price)
            except Exception:
                pass

            # ── analyst price targets ───────────────────────────────────────
            mean_target = low_target = high_target = num_analysts = None
            try:
                pt = t.analyst_price_targets
                if pt is not None:
                    if hasattr(pt, "get"):           # dict-like
                        mean_target  = pt.get("mean")
                        low_target   = pt.get("low")
                        high_target  = pt.get("high")
                        num_analysts = pt.get("numberOfAnalysts")
                    elif hasattr(pt, "__iter__"):    # Series / mapping
                        d = dict(pt)
                        mean_target  = d.get("mean")
                        low_target   = d.get("low")
                        high_target  = d.get("high")
                        num_analysts = d.get("numberOfAnalysts")
                # coerce to Python floats
                mean_target  = float(mean_target)  if mean_target  is not None else None
                low_target   = float(low_target)   if low_target   is not None else None
                high_target  = float(high_target)  if high_target  is not None else None
                num_analysts = int(num_analysts)   if num_analysts is not None else None
            except Exception:
                mean_target = low_target = high_target = num_analysts = None

            # ── upside to mean target ───────────────────────────────────────
            upside_pct: float | None = None
            if mean_target and current_price and current_price > 0:
                upside_pct = round((mean_target - current_price) / current_price * 100, 2)

            # ── analyst recommendations summary ─────────────────────────────
            strong_buy = buy = hold = sell = strong_sell = None
            try:
                rec = t.recommendations
                if rec is not None and not rec.empty:
                    row0 = rec.iloc[0]
                    strong_buy  = int(row0.get("strongBuy",  row0.get("Strong Buy",  0)))
                    buy         = int(row0.get("buy",        row0.get("Buy",         0)))
                    hold        = int(row0.get("hold",       row0.get("Hold",        0)))
                    sell        = int(row0.get("sell",       row0.get("Sell",        0)))
                    strong_sell = int(row0.get("strongSell", row0.get("Strong Sell", 0)))
            except Exception:
                strong_buy = buy = hold = sell = strong_sell = None

            # ── next earnings date ──────────────────────────────────────────
            next_earnings: datetime.date | None = None
            try:
                ed = t.earnings_dates
                if ed is not None and not ed.empty:
                    tz = ed.index.tz
                    today_ts = pd.Timestamp.now(tz=tz) if tz else pd.Timestamp.now()
                    future = ed[ed.index > today_ts]
                    if not future.empty:
                        next_earnings = future.index.min().date()
            except Exception:
                pass

            # ── EPS & revenue estimates ─────────────────────────────────────
            eps_curr_yr = eps_next_yr = None
            try:
                ee = t.earnings_estimate
                if ee is not None and not ee.empty and "avg" in ee.columns:
                    if "0y" in ee.index:
                        v = ee.loc["0y", "avg"]
                        eps_curr_yr = float(v) if pd.notna(v) else None
                    if "+1y" in ee.index:
                        v = ee.loc["+1y", "avg"]
                        eps_next_yr = float(v) if pd.notna(v) else None
            except Exception:
                pass

            # skip symbols with no useful data at all
            if all(v is None for v in [mean_target, strong_buy, next_earnings]):
                return None

            return {
                "symbol":       sym,
                "name":         name_map.get(sym, sym),
                "sector":       config.SECTOR_MAP.get(sym, "Other"),
                "price":        current_price,
                "mean_target":  mean_target,
                "low_target":   low_target,
                "high_target":  high_target,
                "num_analysts": num_analysts,
                "upside_pct":   upside_pct,
                "strong_buy":   strong_buy,
                "buy":          buy,
                "hold":         hold,
                "sell":         sell,
                "strong_sell":  strong_sell,
                "next_earnings": next_earnings,
                "eps_curr_yr":  eps_curr_yr,
                "eps_next_yr":  eps_next_yr,
            }
        except Exception:
            return None

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=config.FETCH_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, sym): sym for sym in sym_ticker}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                rows.append(result)

    return rows


# ── Table helpers ─────────────────────────────────────────────────────────────

def _build_table(
    rows:        list[dict],
    min_touches: int,
    max_pct:     float,
    min_score:   int,
    safety_only: bool,
) -> pd.DataFrame:
    filtered = [
        r for r in rows
        if r["touches"]    >= min_touches
        and abs(r["pct_from_price"]) <= max_pct
        and r["score"]     >= min_score
        and (not safety_only or r["is_safety_line"])
    ]
    if not filtered:
        return pd.DataFrame()

    df = pd.DataFrame(filtered)
    df = df.rename(columns={
        "symbol":         "Symbol",
        "name":           "Name",
        "asset_type":     "Type",
        "price":          "Price",
        "level":          "Level",
        "level_type":     "Level Type",
        "touches":        "Touches",
        "last_touch":     "Last Touch (bars)",
        "pct_from_price": "% Away",
        "atr_ratio":      "ATR Ratio",
        "is_safety_line": "Safety Line",
        "is_compressed":  "Compressed",
        "score":          "Score",
        "forming_soon":   "Forming",
        "direction":      "Direction",
    })
    df["Safety Line"] = df["Safety Line"].map({True: "✅", False: ""})
    df["Compressed"]  = df["Compressed"].map({True: "✅", False: ""})
    df["Forming"]     = df["Forming"].map({True: "⚡", False: ""})
    df["Level Type"]  = df["Level Type"].str.capitalize()
    df["Name"]        = df["Name"].str[:24]
    df = df.sort_values(["Score", "Forming"], ascending=[False, True])
    return df[[
        "Symbol", "Name", "Type", "Price", "Level", "Level Type",
        "% Away", "Touches", "Last Touch (bars)", "ATR Ratio",
        "Safety Line", "Compressed", "Score", "Forming", "Direction",
    ]]


def _style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    return (
        df.style
        .format({
            "% Away":    "{:+.2f}%",
            "ATR Ratio": "{:.2f}",
            "Price":     "{:.4f}",
            "Level":     "{:.4f}",
        })
    )


# ── Trend-scan table helpers ──────────────────────────────────────────────────

_DIR_EMOJI = {
    "BULLISH":       "🟢 Bull",
    "BEARISH":       "🔴 Bear",
    "NEUTRAL":       "⚪ Neutral",
    "N/A":           "—",
    "BULLISH BREAK": "🟢🟢 Bull Break",
    "BEARISH BREAK": "🔴🔴 Bear Break",
    "WEAK BULLISH":  "🟡 Weak Bull",
    "WEAK BEARISH":  "🟠 Weak Bear",
}


def _build_trend_table(scan_rows: list[dict]) -> pd.DataFrame:
    if not scan_rows:
        return pd.DataFrame()
    records = []
    for r in scan_rows:
        records.append({
            "Symbol":        r["symbol"],
            "Name":          r["name"][:22],
            "Sector":        r.get("sector", "Other"),
            "Type":          r["asset_type"],
            "Price":         r["price"],
            "YTD%":          r.get("ytd_chg"),
            "2M Trend":      _DIR_EMOJI.get(r["tf_2M_dir"], r["tf_2M_dir"]),
            "2M Chg%":       r["tf_2M_chg"],
            "4M Trend":      _DIR_EMOJI.get(r["tf_4M_dir"], r["tf_4M_dir"]),
            "4M Chg%":       r["tf_4M_chg"],
            "6M Trend":      _DIR_EMOJI.get(r["tf_6M_dir"], r["tf_6M_dir"]),
            "6M Chg%":       r["tf_6M_chg"],
            "1Y Trend":      _DIR_EMOJI.get(r["tf_1Y_dir"], r["tf_1Y_dir"]),
            "1Y Chg%":       r["tf_1Y_chg"],
            "Daily Signal":  _DIR_EMOJI.get(r["daily_direction"], r["daily_direction"]),
            "Score":         r["daily_score"],
            "RSI":           r["rsi"],
            "52W High":      r["week52_high"],
            "52W Low":       r["week52_low"],
            "% vs 52W High": r["pct_from_52w_high"],
            "% vs 52W Low":  r["pct_from_52w_low"],
        })
    df = pd.DataFrame(records)
    df = df.sort_values("1Y Chg%", ascending=False)
    return df


def _build_sector_table(scan_rows: list[dict]) -> pd.DataFrame:
    """Aggregate sector statistics across all timeframes + YTD."""
    if not scan_rows:
        return pd.DataFrame()

    rows_with_sector = [r for r in scan_rows if r.get("sector")]
    if not rows_with_sector:
        return pd.DataFrame()

    import statistics

    def _safe_mean(vals: list) -> float | None:
        clean = [v for v in vals if v is not None]
        return round(statistics.mean(clean), 2) if clean else None

    def _bull_pct(dirs: list[str]) -> float:
        if not dirs:
            return 0.0
        return round(sum(1 for d in dirs if d == "BULLISH") / len(dirs) * 100, 1)

    def _bear_pct(dirs: list[str]) -> float:
        if not dirs:
            return 0.0
        return round(sum(1 for d in dirs if d == "BEARISH") / len(dirs) * 100, 1)

    sectors: dict[str, list[dict]] = {}
    for r in rows_with_sector:
        s = r["sector"]
        sectors.setdefault(s, []).append(r)

    records = []
    for sector, items in sorted(sectors.items()):
        n = len(items)
        records.append({
            "Sector":         sector,
            "# Symbols":      n,
            "Avg YTD%":       _safe_mean([r.get("ytd_chg") for r in items]),
            "Avg 2M%":        _safe_mean([r.get("tf_2M_chg") for r in items]),
            "Avg 4M%":        _safe_mean([r.get("tf_4M_chg") for r in items]),
            "Avg 6M%":        _safe_mean([r.get("tf_6M_chg") for r in items]),
            "Avg 1Y%":        _safe_mean([r.get("tf_1Y_chg") for r in items]),
            "1Y Bull%":       _bull_pct([r.get("tf_1Y_dir", "") for r in items]),
            "1Y Bear%":       _bear_pct([r.get("tf_1Y_dir", "") for r in items]),
            "Avg RSI":        _safe_mean([r.get("rsi") for r in items]),
            "Avg Score":      _safe_mean([r.get("daily_score") for r in items]),
            "Near 52W High":  sum(1 for r in items
                                  if r.get("pct_from_52w_high") is not None
                                  and r["pct_from_52w_high"] >= -5),
            "Near 52W Low":   sum(1 for r in items
                                  if r.get("pct_from_52w_low") is not None
                                  and r["pct_from_52w_low"] <= 5),
        })

    df = pd.DataFrame(records)
    df = df.sort_values("Avg 1Y%", ascending=False)
    return df


def _style_sector_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def chg_color(val: object) -> str:
        if not isinstance(val, (int, float)):
            return ""
        if val > 0:
            return "color: #1a7f3c"
        if val < 0:
            return "color: #c0392b"
        return ""

    chg_cols = ["Avg YTD%", "Avg 2M%", "Avg 4M%", "Avg 6M%", "Avg 1Y%"]
    fmt: dict = {c: "{:+.2f}%" for c in chg_cols}
    fmt["1Y Bull%"] = "{:.1f}%"
    fmt["1Y Bear%"] = "{:.1f}%"
    fmt["Avg RSI"]  = "{:.1f}"
    fmt["Avg Score"] = "{:+.2f}"
    return df.style.map(chg_color, subset=chg_cols).format(fmt, na_rep="—")


def _style_trend_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def cell_color(val: str) -> str:
        if isinstance(val, str):
            if "Bull" in val:
                return "color: #1a7f3c; font-weight: bold"
            if "Bear" in val:
                return "color: #c0392b; font-weight: bold"
        return ""

    def chg_color(val: float) -> str:
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: #1a7f3c"
            if val < 0:
                return "color: #c0392b"
        return ""

    cols = set(df.columns)
    trend_cols = [c for c in ["2M Trend", "4M Trend", "6M Trend", "1Y Trend", "Daily Signal"] if c in cols]
    chg_cols   = [c for c in ["2M Chg%", "4M Chg%", "6M Chg%", "1Y Chg%", "YTD%"] if c in cols]
    fmt_all = {
        "Price":         "{:.4f}",
        "YTD%":          "{:+.2f}%",
        "2M Chg%":       "{:+.2f}%",
        "4M Chg%":       "{:+.2f}%",
        "6M Chg%":       "{:+.2f}%",
        "1Y Chg%":       "{:+.2f}%",
        "52W High":      "{:.4f}",
        "52W Low":       "{:.4f}",
        "% vs 52W High": "{:+.2f}%",
        "% vs 52W Low":  "{:+.2f}%",
    }
    fmt = {k: v for k, v in fmt_all.items() if k in cols}

    styler = df.style
    if trend_cols:
        styler = styler.map(cell_color, subset=trend_cols)
    if chg_cols:
        styler = styler.map(chg_color, subset=chg_cols)
    return styler.format(fmt, na_rep="—")


# ── Earnings table helpers ────────────────────────────────────────────────────

def _consensus_label(row: dict) -> str:
    sb = row.get("strong_buy")  or 0
    b  = row.get("buy")         or 0
    h  = row.get("hold")        or 0
    s  = row.get("sell")        or 0
    ss = row.get("strong_sell") or 0
    total = sb + b + h + s + ss
    if total == 0:
        return "—"
    bull = sb + b
    bear = s + ss
    if bull / total > 0.60:
        return "Strong Buy" if sb >= b else "Buy"
    elif bear / total > 0.40:
        return "Sell" if ss >= s else "Weak Sell"
    else:
        return "Hold"


def _build_earnings_table(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        records.append({
            "Symbol":        r["symbol"],
            "Name":          r["name"][:22],
            "Sector":        r.get("sector", "Other"),
            "Price":         r.get("price"),
            "Mean Target":   r.get("mean_target"),
            "Low Target":    r.get("low_target"),
            "High Target":   r.get("high_target"),
            "Upside%":       r.get("upside_pct"),
            "# Analysts":    r.get("num_analysts"),
            "Strong Buy":    r.get("strong_buy"),
            "Buy":           r.get("buy"),
            "Hold":          r.get("hold"),
            "Sell":          r.get("sell"),
            "Strong Sell":   r.get("strong_sell"),
            "Consensus":     _consensus_label(r),
            "Next Earnings": r.get("next_earnings"),
            "EPS Curr Yr":   r.get("eps_curr_yr"),
            "EPS Next Yr":   r.get("eps_next_yr"),
        })
    df = pd.DataFrame(records)
    df = df.sort_values("Upside%", ascending=False, na_position="last")
    return df


def _style_earnings_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def upside_color(val: object) -> str:
        if not isinstance(val, (int, float)):
            return ""
        if val > 10:
            return "color: #1a7f3c; font-weight: bold"
        if val > 0:
            return "color: #1a7f3c"
        if val < 0:
            return "color: #c0392b"
        return ""

    def consensus_color(val: str) -> str:
        if not isinstance(val, str):
            return ""
        v = val.lower()
        if "strong buy" in v or v == "buy":
            return "color: #1a7f3c; font-weight: bold"
        if "sell" in v:
            return "color: #c0392b; font-weight: bold"
        return ""

    cols = set(df.columns)
    fmt: dict = {}
    if "Price"       in cols: fmt["Price"]       = "{:.2f}"
    if "Mean Target" in cols: fmt["Mean Target"] = "{:.2f}"
    if "Low Target"  in cols: fmt["Low Target"]  = "{:.2f}"
    if "High Target" in cols: fmt["High Target"] = "{:.2f}"
    if "Upside%"     in cols: fmt["Upside%"]     = "{:+.2f}%"
    if "EPS Curr Yr" in cols: fmt["EPS Curr Yr"] = "{:.2f}"
    if "EPS Next Yr" in cols: fmt["EPS Next Yr"] = "{:.2f}"

    styler = df.style.format(fmt, na_rep="—")
    if "Upside%"   in cols: styler = styler.map(upside_color,    subset=["Upside%"])
    if "Consensus" in cols: styler = styler.map(consensus_color, subset=["Consensus"])
    return styler


# ── Chart ─────────────────────────────────────────────────────────────────────

def _make_chart(
    symbol: str,
    rows:   list[dict],
    raw:    dict[str, pd.DataFrame],
) -> go.Figure:
    sym_rows = [r for r in rows if r["symbol"] == symbol]
    df       = raw.get(symbol)
    if df is None or not sym_rows:
        return go.Figure()

    df_plot = df.tail(90).copy()
    open_s  = df_plot["Open"].squeeze()
    high_s  = df_plot["High"].squeeze()
    low_s   = df_plot["Low"].squeeze()
    close_s = df_plot["Close"].squeeze()

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=open_s, high=high_s, low=low_s, close=close_s,
        name=symbol,
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ))

    # Safety lines
    for r in sym_rows:
        if not r["is_safety_line"]:
            continue
        is_sup  = r["level_type"] == "support"
        color   = "#26a69a" if is_sup else "#ef5350"
        dash    = "solid"   if r["forming_soon"] else "dot"
        width   = 2         if r["forming_soon"] else 1

        touch_dots = "●" * min(r["touches"], 8)
        compressed = " 📉" if r["is_compressed"] else ""
        forming    = " ⚡" if r["forming_soon"]  else ""
        label = (
            f"{'SUP' if is_sup else 'RES'}  {r['level']:.4f}  "
            f"|  {r['touches']}T {touch_dots}  "
            f"|  score {r['score']}/10"
            f"{compressed}{forming}"
        )

        fig.add_hline(
            y=r["level"],
            line_color=color,
            line_dash=dash,
            line_width=width,
            annotation_text=label,
            annotation_position="right",
            annotation_font_color=color,
            annotation_font_size=10,
        )

    meta     = sym_rows[0]
    fig.update_layout(
        title=(
            f"<b>{symbol}</b>  —  {meta['name']}  "
            f"|  Price: {meta['price']}  "
            f"|  ATR ratio: {meta['atr_ratio']:.2f}"
            f"{'  📉 COMPRESSED' if meta['atr_ratio'] < COMPRESSION_RATIO else ''}"
        ),
        xaxis_rangeslider_visible=False,
        height=540,
        template="plotly_dark",
        margin=dict(l=40, r=180, t=55, b=40),
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


# ── Forming-soon card HTML ────────────────────────────────────────────────────

def _card_html(r: dict) -> str:
    is_bull   = "BULLISH" in r["direction"]
    css_cls   = "ft-bull" if is_bull else "ft-bear"
    arrow     = "▲" if is_bull else "▼"
    pct_str   = f"{abs(r['pct_from_price']):.2f}%"
    touch_bar = "●" * min(r["touches"], 8)
    compress  = "  📉 compressing" if r["is_compressed"] else ""
    return (
        f'<div class="ft-card {css_cls}">'
        f'<span class="ft-sym">{arrow} {r["symbol"]}</span>'
        f'<span class="ft-score-badge">Score {r["score"]}/10</span><br>'
        f'<span class="ft-sub">'
        f'{r["name"][:22]}<br>'
        f'Level&nbsp;<b>{r["level"]}</b>&nbsp;|&nbsp;{pct_str} away&nbsp;|&nbsp;'
        f'{r["level_type"].capitalize()}<br>'
        f'Touches&nbsp;{r["touches"]}&nbsp;{touch_bar}<br>'
        f'ATR ratio&nbsp;{r["atr_ratio"]:.2f}{compress}'
        f'</span></div>'
    )


# ── App ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📊 FlagTrends — Tori Trades Dashboard")
    st.caption(
        "Detects safety lines, breakout structures, and forming patterns "
        "using Tori Trades methodology across the Libertex symbol universe."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Filters")
        asset_type  = st.selectbox("Asset Type", ["All", "Stock", "Commodity"], index=0)
        min_touches = st.slider(
            f"Min Touches  (safety line = ≥{MIN_TOUCHES})",
            min_value=1, max_value=10, value=MIN_TOUCHES)
        max_pct     = st.slider(
            "Max % Away from Level",
            min_value=0.5, max_value=15.0, value=5.0, step=0.5)
        min_score   = st.slider("Min Score (0–10)", 0, 10, 3)
        safety_only = st.checkbox("Safety Lines only", value=True)

        st.divider()
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption(f"Cache TTL: 5 min  |  Updated: {datetime.datetime.now():%H:%M:%S}")
        st.divider()
        st.markdown(
            "**Scoring guide**\n"
            "- **Touches** — more = stronger level\n"
            "- **Recency** — fresh touches score higher\n"
            "- **ATR ratio** — <0.85 = compression ✅\n"
            "- **Proximity** — within 3% scores +1\n"
        )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_levels, tab_trends, tab_sectors, tab_earnings = st.tabs([
        "🏛️ Key Levels", "📅 Multi-Timeframe Trends", "🗂️ Sector Analysis", "💰 Earnings & Estimates"
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Key Levels (existing view)
    # ════════════════════════════════════════════════════════════════════════
    with tab_levels:
        # ── Load data ────────────────────────────────────────────────────────
        rows, raw = _run(asset_type)

        # ── Summary metrics ──────────────────────────────────────────────────
        total_syms   = len({r["symbol"]                for r in rows})
        safety_count = len({(r["symbol"], r["level"])  for r in rows if r["is_safety_line"]})
        forming_count= len({(r["symbol"], r["level"])  for r in rows if r["forming_soon"]})
        compressed_c = len({r["symbol"]                for r in rows if r["is_compressed"]})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Symbols Analysed",  total_syms)
        m2.metric("Safety Lines Found", safety_count)
        m3.metric("⚡ Forming Soon",   forming_count)
        m4.metric("📉 ATR Compressed", compressed_c)

        st.divider()

        # ── Forming-soon spotlight ────────────────────────────────────────────
        forming_rows = sorted(
            [r for r in rows if r["forming_soon"]],
            key=lambda r: -r["score"],
        )

        if forming_rows:
            st.subheader(f"⚡ Forming Now — {len(forming_rows)} structure(s)  (top 12 shown)")
            cols = st.columns(4)
            for i, r in enumerate(forming_rows[:12]):
                cols[i % 4].markdown(_card_html(r), unsafe_allow_html=True)
        else:
            st.info("No structures forming soon. Adjust filters or refresh data.")

        st.divider()

        # ── Full key-levels table ─────────────────────────────────────────────
        st.subheader("📋 All Key Levels")
        table_df = _build_table(rows, min_touches, max_pct, min_score, safety_only)
        if not table_df.empty:
            st.dataframe(
                _style_table(table_df),
                use_container_width=True,
                height=420,
            )
            st.caption(f"{len(table_df)} rows matching current filters")
        else:
            st.info("No levels match current filters.")

        st.divider()

        # ── Chart ─────────────────────────────────────────────────────────────
        st.subheader("📈 Price Chart with Safety Lines  (last 90 trading days)")

        all_syms = sorted({r["symbol"] for r in rows})
        default   = forming_rows[0]["symbol"] if forming_rows else (all_syms[0] if all_syms else None)
        def_idx   = all_syms.index(default) if default in all_syms else 0

        if all_syms:
            col_sel, col_info = st.columns([2, 5])
            selected = col_sel.selectbox("Symbol", all_syms, index=def_idx)

            sym_rows = [r for r in rows if r["symbol"] == selected]
            if sym_rows:
                meta = sym_rows[0]
                col_info.markdown(
                    f"**{meta['name']}**  |  "
                    f"Price: `{meta['price']}`  |  "
                    f"ATR ratio: `{meta['atr_ratio']:.2f}`  |  "
                    f"{'📉 ATR compressed' if meta['atr_ratio'] < COMPRESSION_RATIO else 'ATR normal'}"
                )

            fig = _make_chart(selected, rows, raw)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander(f"Key level details for {selected}"):
                sym_df = _build_table(sym_rows, 1, 100.0, 0, False)
                if not sym_df.empty:
                    st.dataframe(_style_table(sym_df), use_container_width=True, height=300)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Multi-Timeframe Trend Scanner
    # ════════════════════════════════════════════════════════════════════════
    with tab_trends:
        st.subheader("📅 Multi-Timeframe Trend Scanner")
        st.caption(
            "**Trend direction** is derived from the % price change inside each window. "
            ">+2% = 🟢 Bull  |  <-2% = 🔴 Bear  |  otherwise ⚪ Neutral.  "
            "**Daily Signal** uses EMA cross + SMA break + Donchian breakout (same as main scanner).  "
            "**52W High / Low** are the max / min close over the last 252 trading bars."
        )

        scan_rows = _run_trend_scan(asset_type)

        # ── Quick summary cards ───────────────────────────────────────────────
        bull_1y = sum(1 for r in scan_rows if r["tf_1Y_dir"] == "BULLISH")
        bear_1y = sum(1 for r in scan_rows if r["tf_1Y_dir"] == "BEARISH")
        near_52h = sum(1 for r in scan_rows
                       if r["pct_from_52w_high"] is not None and r["pct_from_52w_high"] >= -3)
        near_52l = sum(1 for r in scan_rows
                       if r["pct_from_52w_low"]  is not None and r["pct_from_52w_low"]  <= 3)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Bullish 1Y", bull_1y)
        c2.metric("🔴 Bearish 1Y", bear_1y)
        c3.metric("📈 Near 52W High (≤3%)", near_52h)
        c4.metric("📉 Near 52W Low  (≤3%)", near_52l)

        st.divider()

        # ── Filters ───────────────────────────────────────────────────────────
        f1, f2, f3 = st.columns(3)
        tf_filter = f1.selectbox(
            "Filter by 1Y Trend", ["All", "BULLISH", "BEARISH", "NEUTRAL"], index=0,
            key="tf_tf_filter",
        )
        score_filter = f2.slider("Min |Daily Score|", 0, 3, 0, key="tf_score_filter")
        search_sym   = f3.text_input("Search symbol / name", key="tf_search")

        filtered = scan_rows
        if tf_filter != "All":
            filtered = [r for r in filtered if r["tf_1Y_dir"] == tf_filter]
        if score_filter:
            filtered = [r for r in filtered if abs(r["daily_score"]) >= score_filter]
        if search_sym:
            q = search_sym.upper()
            filtered = [r for r in filtered
                        if q in r["symbol"].upper() or q in r["name"].upper()]

        trend_df = _build_trend_table(filtered)
        if not trend_df.empty:
            st.dataframe(
                _style_trend_table(trend_df),
                use_container_width=True,
                height=520,
            )
            st.caption(f"{len(trend_df)} symbols  |  sorted by 1Y % Change")
        else:
            st.info("No symbols match the current filters.")

        # ── 52W extremes sub-tables ───────────────────────────────────────────
        st.divider()
        col_hi, col_lo = st.columns(2)

        with col_hi:
            st.markdown("#### 📈 Closest to 52W High")
            near_high_df = _build_trend_table(
                sorted(
                    [r for r in scan_rows if r["pct_from_52w_high"] is not None],
                    key=lambda r: r["pct_from_52w_high"],
                    reverse=True,
                )[:15]
            )
            if not near_high_df.empty:
                st.dataframe(
                    near_high_df[["Symbol", "Name", "Price", "52W High", "% vs 52W High", "1Y Trend"]].style.format({
                        "Price":         "{:.4f}",
                        "52W High":      "{:.4f}",
                        "% vs 52W High": "{:+.2f}%",
                    }),
                    use_container_width=True, height=360,
                )

        with col_lo:
            st.markdown("#### 📉 Closest to 52W Low")
            near_low_df = _build_trend_table(
                sorted(
                    [r for r in scan_rows if r["pct_from_52w_low"] is not None],
                    key=lambda r: r["pct_from_52w_low"],
                )[:15]
            )
            if not near_low_df.empty:
                st.dataframe(
                    near_low_df[["Symbol", "Name", "Price", "52W Low", "% vs 52W Low", "1Y Trend"]].style.format({
                        "Price":        "{:.4f}",
                        "52W Low":      "{:.4f}",
                        "% vs 52W Low": "{:+.2f}%",
                    }),
                    use_container_width=True, height=360,
                )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Sector Analysis
    # ════════════════════════════════════════════════════════════════════════
    with tab_sectors:
        st.subheader("🗂️ Sector Analysis — Aggregate Returns & Trends")
        st.caption(
            "Sector averages are computed across all symbols in each sector that have "
            "sufficient data. YTD = price change since Jan 1 of the current year."
        )

        scan_rows_s = _run_trend_scan(asset_type)

        # ── Sector aggregate table ────────────────────────────────────────────
        sector_df = _build_sector_table(scan_rows_s)

        if not sector_df.empty:
            # Top-level sector performance bar chart (1Y avg %)
            fig_bar = go.Figure()
            colors = [
                "#26a69a" if v >= 0 else "#ef5350"
                for v in sector_df["Avg 1Y%"].fillna(0)
            ]
            fig_bar.add_trace(go.Bar(
                x=sector_df["Sector"],
                y=sector_df["Avg 1Y%"],
                marker_color=colors,
                name="Avg 1Y Return %",
                text=[f"{v:+.1f}%" if v is not None else "—"
                      for v in sector_df["Avg 1Y%"]],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Sector Average 1Y Return %",
                template="plotly_dark",
                height=360,
                margin=dict(l=30, r=30, t=50, b=80),
                yaxis_title="Avg 1Y %",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Multi-period grouped bar chart
            periods = ["Avg YTD%", "Avg 2M%", "Avg 4M%", "Avg 6M%", "Avg 1Y%"]
            period_labels = ["YTD", "2M", "4M", "6M", "1Y"]
            palette = ["#90caf9", "#80cbc4", "#a5d6a7", "#fff176", "#ef9a9a"]

            fig_multi = go.Figure()
            for col, label, color in zip(periods, period_labels, palette):
                fig_multi.add_trace(go.Bar(
                    name=label,
                    x=sector_df["Sector"],
                    y=sector_df[col].fillna(0),
                    marker_color=color,
                ))
            fig_multi.update_layout(
                barmode="group",
                title="Sector Returns by Horizon",
                template="plotly_dark",
                height=420,
                margin=dict(l=30, r=30, t=50, b=80),
                yaxis_title="Avg %",
                xaxis_tickangle=-35,
                legend=dict(orientation="h", y=1.07),
            )
            st.plotly_chart(fig_multi, use_container_width=True)

            # Bull vs Bear % stacked bar
            fig_bull = go.Figure()
            fig_bull.add_trace(go.Bar(
                name="🟢 Bullish 1Y%",
                x=sector_df["Sector"],
                y=sector_df["1Y Bull%"],
                marker_color="#26a69a",
                text=[f"{v:.0f}%" for v in sector_df["1Y Bull%"]],
                textposition="inside",
            ))
            fig_bull.add_trace(go.Bar(
                name="🔴 Bearish 1Y%",
                x=sector_df["Sector"],
                y=sector_df["1Y Bear%"],
                marker_color="#ef5350",
                text=[f"{v:.0f}%" for v in sector_df["1Y Bear%"]],
                textposition="inside",
            ))
            fig_bull.update_layout(
                barmode="stack",
                title="% of Symbols Bullish vs Bearish (1Y)",
                template="plotly_dark",
                height=360,
                margin=dict(l=30, r=30, t=50, b=80),
                yaxis_title="% of Symbols",
                xaxis_tickangle=-35,
                legend=dict(orientation="h", y=1.07),
            )
            st.plotly_chart(fig_bull, use_container_width=True)

            st.divider()
            st.subheader("📋 Sector Summary Table")
            st.dataframe(
                _style_sector_table(sector_df),
                use_container_width=True,
                height=460,
            )
            st.caption(f"{len(sector_df)} sectors  |  sorted by Avg 1Y %")

            # ── Drill-down: symbols within a sector ───────────────────────────
            st.divider()
            st.subheader("🔍 Sector Drill-Down — Symbol Returns")
            sectors_available = sorted(sector_df["Sector"].tolist())
            chosen_sector = st.selectbox("Choose a sector", sectors_available, key="sector_drill")

            drill_rows = [r for r in scan_rows_s if r.get("sector") == chosen_sector]
            drill_df   = _build_trend_table(drill_rows)

            if not drill_df.empty:
                display_cols = [
                    "Symbol", "Name", "Price",
                    "YTD%", "2M Chg%", "4M Chg%", "6M Chg%", "1Y Chg%",
                    "Daily Signal", "Score", "RSI",
                ]
                st.dataframe(
                    _style_trend_table(drill_df[display_cols]),
                    use_container_width=True,
                    height=460,
                )
                st.caption(f"{len(drill_df)} symbols in {chosen_sector}  |  sorted by 1Y % Change")
        else:
            st.info("No sector data available. Run a scan first.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — Earnings & Analyst Estimates
    # ════════════════════════════════════════════════════════════════════════
    with tab_earnings:
        st.subheader("💰 Earnings & Analyst Estimates")
        st.caption(
            "Analyst price targets, recommendation consensus, forward EPS estimates, "
            "and upcoming earnings dates sourced from yfinance.  "
            "Data is cached for 1 hour.  Only symbols with at least one data point are shown."
        )

        er_rows = _run_earnings_scan(asset_type)

        if not er_rows:
            st.info("No analyst/earnings data found for the current symbol set.")
        else:
            today = datetime.date.today()

            # ── Summary metrics ───────────────────────────────────────────────
            with_target   = [r for r in er_rows if r.get("mean_target") is not None]
            upsides       = [r["upside_pct"] for r in with_target if r.get("upside_pct") is not None]
            avg_upside    = round(sum(upsides) / len(upsides), 1) if upsides else 0.0
            upcoming_7d   = sum(1 for r in er_rows
                                if r.get("next_earnings") and
                                0 <= (r["next_earnings"] - today).days <= 7)
            upcoming_30d  = sum(1 for r in er_rows
                                if r.get("next_earnings") and
                                0 <= (r["next_earnings"] - today).days <= 30)
            strong_buy_ct = sum(1 for r in er_rows if _consensus_label(r) in ("Strong Buy", "Buy"))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Symbols w/ Price Target", len(with_target))
            m2.metric("Avg Analyst Upside",      f"{avg_upside:+.1f}%")
            m3.metric("⏰ Earnings Next 7 Days",  upcoming_7d)
            m4.metric("📅 Earnings Next 30 Days", upcoming_30d)

            st.divider()

            # ── Charts ────────────────────────────────────────────────────────
            col_a, col_b = st.columns(2)

            # Chart 1: Average analyst upside by sector
            with col_a:
                sector_upside: dict[str, list[float]] = {}
                for r in with_target:
                    if r.get("upside_pct") is not None:
                        sector_upside.setdefault(r.get("sector", "Other"), []).append(r["upside_pct"])

                if sector_upside:
                    s_labels = sorted(sector_upside)
                    s_values = [round(sum(sector_upside[s]) / len(sector_upside[s]), 1) for s in s_labels]
                    s_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in s_values]
                    fig_upside = go.Figure(go.Bar(
                        x=s_labels, y=s_values,
                        marker_color=s_colors,
                        text=[f"{v:+.1f}%" for v in s_values],
                        textposition="outside",
                    ))
                    fig_upside.update_layout(
                        title="Avg Analyst Upside% by Sector",
                        template="plotly_dark",
                        height=360,
                        margin=dict(l=30, r=30, t=50, b=90),
                        yaxis_title="Avg Upside %",
                        xaxis_tickangle=-40,
                    )
                    st.plotly_chart(fig_upside, use_container_width=True)

            # Chart 2: Analyst consensus breakdown by sector
            with col_b:
                sector_consensus: dict[str, dict[str, int]] = {}
                for r in er_rows:
                    sec = r.get("sector", "Other")
                    d   = sector_consensus.setdefault(sec, {"Buy": 0, "Hold": 0, "Sell": 0})
                    lbl = _consensus_label(r)
                    if "Buy" in lbl:
                        d["Buy"] += 1
                    elif "Sell" in lbl:
                        d["Sell"] += 1
                    else:
                        d["Hold"] += 1

                if sector_consensus:
                    secs = sorted(sector_consensus)
                    fig_cons = go.Figure()
                    for sentiment, color in [("Buy", "#26a69a"), ("Hold", "#90a4ae"), ("Sell", "#ef5350")]:
                        fig_cons.add_trace(go.Bar(
                            name=sentiment,
                            x=secs,
                            y=[sector_consensus[s].get(sentiment, 0) for s in secs],
                            marker_color=color,
                        ))
                    fig_cons.update_layout(
                        barmode="stack",
                        title="Analyst Consensus by Sector",
                        template="plotly_dark",
                        height=360,
                        margin=dict(l=30, r=30, t=50, b=90),
                        yaxis_title="# Symbols",
                        xaxis_tickangle=-40,
                        legend=dict(orientation="h", y=1.08),
                    )
                    st.plotly_chart(fig_cons, use_container_width=True)

            # ── Top upside opportunities ──────────────────────────────────────
            st.divider()
            st.subheader("🎯 Highest Analyst Upside  (top 20)")
            top_upside = sorted(with_target,
                                key=lambda r: r.get("upside_pct") or -999,
                                reverse=True)[:20]
            top_df = _build_earnings_table(top_upside)
            if not top_df.empty:
                display_cols = [c for c in [
                    "Symbol", "Name", "Sector", "Price",
                    "Mean Target", "Low Target", "High Target",
                    "Upside%", "# Analysts", "Consensus",
                ] if c in top_df.columns]
                st.dataframe(
                    _style_earnings_table(top_df[display_cols]),
                    use_container_width=True,
                    height=460,
                )

            # ── Upcoming earnings calendar ────────────────────────────────────
            st.divider()
            st.subheader("📅 Upcoming Earnings  (next 30 days)")
            calendar_rows = sorted(
                [r for r in er_rows
                 if r.get("next_earnings") and
                 0 <= (r["next_earnings"] - today).days <= 30],
                key=lambda r: r["next_earnings"],
            )
            if calendar_rows:
                cal_df = _build_earnings_table(calendar_rows)
                cal_cols = [c for c in [
                    "Symbol", "Name", "Sector", "Price",
                    "Next Earnings", "Mean Target", "Upside%",
                    "Consensus", "# Analysts",
                    "EPS Curr Yr", "EPS Next Yr",
                ] if c in cal_df.columns]
                st.dataframe(
                    _style_earnings_table(cal_df[cal_cols]),
                    use_container_width=True,
                    height=420,
                )
                st.caption(f"{len(calendar_rows)} earnings events in the next 30 days")
            else:
                st.info("No earnings events found in the next 30 days.")

            # ── Full analyst table ────────────────────────────────────────────
            st.divider()
            st.subheader("📋 Full Earnings & Analyst Table")

            # Per-sector filter
            ea_sectors = sorted({r.get("sector", "Other") for r in er_rows})
            ef1, ef2 = st.columns([2, 4])
            ea_sector_filter = ef1.selectbox(
                "Filter by Sector", ["All"] + ea_sectors, key="ea_sector_filter"
            )
            ea_search = ef2.text_input("Search symbol / name", key="ea_search")

            filtered_er = er_rows
            if ea_sector_filter != "All":
                filtered_er = [r for r in filtered_er if r.get("sector") == ea_sector_filter]
            if ea_search:
                q = ea_search.upper()
                filtered_er = [r for r in filtered_er
                               if q in r["symbol"].upper() or q in r["name"].upper()]

            full_df = _build_earnings_table(filtered_er)
            if not full_df.empty:
                st.dataframe(
                    _style_earnings_table(full_df),
                    use_container_width=True,
                    height=540,
                )
                st.caption(f"{len(full_df)} symbols  |  sorted by Analyst Upside%")
            else:
                st.info("No symbols match the current filters.")


if __name__ == "__main__":
    main()
