"""
FlagTrends — Tori Trades Dashboard
====================================
Run:  streamlit run dashboard.py
"""

from __future__ import annotations

import csv
import datetime
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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


@st.cache_data(ttl=300, show_spinner="Fetching & analysing market data…")
def _run(asset_type: str) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """
    Returns:
        rows     – flat list of dicts (one per key level, all symbols)
        raw      – OHLCV DataFrames keyed by Libertex symbol
    """
    symbols = _load_symbols()
    if asset_type != "All":
        symbols = [s for s in symbols if s["asset_type"].lower() == asset_type.lower()]

    sym_ticker = {
        s["symbol"]: config.SYMBOL_MAP[s["symbol"]]
        for s in symbols
        if s["symbol"] in config.SYMBOL_MAP
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
    symbols = _load_symbols()
    if asset_type != "All":
        symbols = [s for s in symbols if s["asset_type"].lower() == asset_type.lower()]

    sym_ticker = {
        s["symbol"]: config.SYMBOL_MAP[s["symbol"]]
        for s in symbols
        if s["symbol"] in config.SYMBOL_MAP
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
            "price":             ts.price,
            "week52_high":       ts.week52_high,
            "week52_low":        ts.week52_low,
            "pct_from_52w_high": ts.pct_from_52w_high,
            "pct_from_52w_low":  ts.pct_from_52w_low,
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
    def row_color(row: pd.Series) -> list[str]:
        if row["Forming"] == "⚡":
            if "BULLISH" in str(row.get("Direction", "")):
                return ["background-color: #0c2216"] * len(row)
            if "BEARISH" in str(row.get("Direction", "")):
                return ["background-color: #220c0c"] * len(row)
        return [""] * len(row)

    return (
        df.style
        .apply(row_color, axis=1)
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
            "Type":          r["asset_type"],
            "Price":         r["price"],
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


def _style_trend_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def cell_color(val: str) -> str:
        if isinstance(val, str):
            if "Bull" in val:
                return "color: #26a69a"
            if "Bear" in val:
                return "color: #ef5350"
        return ""

    def chg_color(val: float) -> str:
        if isinstance(val, (int, float)):
            if val > 2:
                return "color: #26a69a"
            if val < -2:
                return "color: #ef5350"
        return "color: #9e9e9e"

    return (
        df.style
        .applymap(cell_color, subset=["2M Trend", "4M Trend", "6M Trend", "1Y Trend", "Daily Signal"])
        .applymap(chg_color,  subset=["2M Chg%", "4M Chg%", "6M Chg%", "1Y Chg%"])
        .format({
            "Price":         "{:.4f}",
            "2M Chg%":       "{:+.2f}%",
            "4M Chg%":       "{:+.2f}%",
            "6M Chg%":       "{:+.2f}%",
            "1Y Chg%":       "{:+.2f}%",
            "52W High":      "{:.4f}",
            "52W Low":       "{:.4f}",
            "% vs 52W High": "{:+.2f}%",
            "% vs 52W Low":  "{:+.2f}%",
        }, na_rep="—")
    )


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
    tab_levels, tab_trends = st.tabs(["🏛️ Key Levels", "📅 Multi-Timeframe Trends"])

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


if __name__ == "__main__":
    main()
