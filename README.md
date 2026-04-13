# FlagTrends

Daily scanner and interactive dashboard that identifies **safety lines**, **breakout structures**, and **forming patterns** across the Libertex universe (stocks + commodities) using the **Tori Trades strategy**.

---

## Strategy Overview — Tori Trades

### Safety Line
A **safety line** is a horizontal price level where price has tested and reacted at least **N times** (default: 2 touches). The more touches, the stronger the level.

- **Support safety line** — level sits *below* current price. Bulls defend it; a break downward through it is bearish.
- **Resistance safety line** — level sits *above* current price. Bears defend it; a break upward through it is bullish.

The safety line acts as the **structure anchor** — the reference price that defines both the trade trigger and the invalidation line.

```
                ▲
                │          ╔═ Resistance safety line (3 touches ●●●)
   Price        │ ─ ─ ─ ─ ╚══════╗─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                │                 ║← breakout structure forming here
   Current ──▶  │         ━━━━━━━╝
                │                       ╔═ Support safety line (4 touches ●●●●)
                │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ╚═══
                └──────────────────────────────────────▶ time
```

---

### Breakout Structure
A **breakout structure** is consolidation building *near* a safety line. Characteristics:

| Feature | What it means |
|---|---|
| **ATR compression** | Current ATR is below its 50-bar average (ratio < 0.85). Price range is shrinking — energy coiling. |
| **Proximity** | Price is within ≤ 3 % of the safety line — approaching the decision zone. |
| **Recent touch** | The level was tested within the last 5–20 bars, confirming it is still active. |

When all three align the structure scores highest and is flagged as **Forming Soon**.

---

### Touch Counting
A "touch" is a **discrete approach-and-reaction event**:

1. One or more consecutive bars whose high–low range overlaps within **0.4 %** of the level price.
2. Consecutive bars inside the zone are **deduplicated** — a week of sideways grinding at the level counts as **one** touch, not five.

```
Level:  100.00

Bar A:  H=100.3  L=99.6   ← enters zone         }
Bar B:  H=100.2  L=99.7   ← still in zone        }  = 1 touch
Bar C:  H=99.4   L=98.8   ← exits zone           }
Bar D:  H=100.5  L=100.1  ← new entry → touch 2
Bar E:  H=101.0  L=100.6  ← exits zone
Bar F:  H=100.1  L=99.65  ← new entry → touch 3
```

---

### Scoring System (0 – 10)

Each detected level receives a composite **score**:

| Component | Max pts | Rule |
|---|---|---|
| **Touches** | 4 | `touches − 1`, capped at 4  (2T→1, 3T→2, 4T→3, 5T+→4) |
| **Recency** | 2 | last touch ≤ 5 bars → 2 pts, ≤ 20 bars → 1 pt |
| **ATR compression** | 3 | ratio < 0.70 → 3, < 0.85 → 2, < 1.00 → 1 |
| **Proximity** | 1 | price within 3 % of level |

Higher score = higher-probability, more imminent breakout.

---

### Forming Soon
A structure is **Forming Soon** when:
- ≥ 2 confirmed touches (`is_safety_line = True`)
- Price is within **3 %** of the level

Direction is inferred from the level type:
- Near **support** → `BULLISH BREAKOUT` expected upward
- Near **resistance** → `BEARISH BREAKOUT` expected downward

---

## Installation

```bash
git clone git@github.com:tim7en/FlagTrends.git
cd FlagTrends
pip install -r requirements.txt
```

---

## Dashboard

```bash
streamlit run dashboard.py
```

The dashboard opens at `http://localhost:8501` and auto-caches data for 5 minutes.

### Layout

```
┌─ Sidebar ──────────────┐  ┌─ Main ────────────────────────────────────────┐
│ Asset Type  [All ▼]    │  │  📊 Summary: symbols | safety lines | forming  │
│ Min Touches [2 ▶]      │  ├───────────────────────────────────────────────┤
│ Max % Away  [5.0 ▶]    │  │  ⚡ Forming Now — N structures (top 12 cards)  │
│ Min Score   [3 ▶]      │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│ Safety only [✓]        │  │  │▲ AAPL    │ │▼ XOM     │ │▲ JPM     │ ...  │
│                        │  │  │Level 178 │ │Level 155 │ │Level 298 │      │
│ [🔄 Refresh Data]      │  │  │3T ●●● 8/10│ │4T ●●●● 9│ │2T ●● 7  │      │
│                        │  │  └──────────┘ └──────────┘ └──────────┘      │
│ Scoring guide          │  ├───────────────────────────────────────────────┤
│ ● Touches              │  │  📋 All Key Levels (sortable, colour-coded)    │
│ ● Recency              │  │  green rows = bullish forming                  │
│ ● ATR ratio            │  │  red   rows = bearish forming                  │
│ ● Proximity            │  ├───────────────────────────────────────────────┤
└────────────────────────┘  │  📈 Price Chart with Safety Lines (last 90d)   │
                             │  [select symbol] candlestick + S/R overlays    │
                             └───────────────────────────────────────────────┘
```

---

## CLI Scanner

In addition to the dashboard, a fast CLI scan is available:

```bash
# All signals (EMA cross, Price/SMA, Donchian) — min score 2
python main.py --min-score 2

# Tori safety-line scan: stocks only, 3 % proximity window
python main.py --type stock

# Save full results to CSV
python main.py --output results.csv

# Force fresh download
python main.py --no-cache

# Narrow the breakout detection window to 3 days
python main.py --signal-window 3 --min-score 2
```

---

## Project Structure

```
FlagTrends/
├── dashboard.py        Streamlit dashboard (Tori Trades view)
├── main.py             CLI trend-break scanner
├── tori.py             Safety line & breakout structure detection
├── analyzer.py         EMA-cross, Price/SMA, Donchian signals
├── fetcher.py          yfinance downloader (parallel, parquet cache)
├── config.py           Parameters + Libertex → yfinance symbol map
├── reporter.py         Console & CSV output for CLI
├── symbols.csv         Libertex symbol universe
└── requirements.txt
```

---

## Data Source

Daily OHLCV data fetched via **[yfinance](https://github.com/ranaroussi/yfinance)**.  
Data is cached locally in `.cache/` as Parquet files (refreshed daily).

---

## Parameters (config.py / tori.py)

| Parameter | Default | Description |
|---|---|---|
| `LOOKBACK_DAYS` | 365 | Days of history to fetch |
| `MIN_TOUCHES` | 2 | Touches needed to qualify as safety line |
| `TOUCH_PCT` | 0.4 % | Zone width for touch detection |
| `APPROACH_PCT` | 3.0 % | "Forming soon" proximity threshold |
| `COMPRESSION_RATIO` | 0.85 | ATR_current / ATR_hist threshold for compression |
| `PIVOT_WINDOW` | 5 | Bars each side for swing-high/low detection |
| `CLUSTER_PCT` | 1.0 % | Merge nearby levels into one zone |
| `MAX_LEVELS` | 8 | Max key levels returned per symbol |
