"""
Trend Break Scanner
===================
Flags Libertex symbols showing trend-breaking signals using yfinance daily data.

Usage
-----
    python main.py [options]

Options
-------
    --type {stock,commodity,all}    Filter asset type (default: all)
    --min-score N                   Minimum |score| to display (default: 1)
    --days N                        Historical days to fetch (default: 365)
    --signal-window N               Days to look back for break signal (default: 5)
    --donchian N                    Donchian channel period (default: 20)
    --output FILE                   Also save full results to CSV
    --no-cache                      Force fresh download (ignore cache)

Signals (each contributes ±1 to score)
---------------------------------------
    EMA Cross      EMA20 crossed EMA50 in the last --signal-window days
    Price/SMA50    Close crossed SMA50  in the last --signal-window days
    Donchian       Close is a new N-day high/low

Score thresholds
----------------
    |score| == 3   Strong break (all three signals aligned)
    |score| == 2   Break (two signals aligned)
    |score| == 1   Weak signal (shown when --min-score 1)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import config
from analyzer import TrendSignal, analyze
from fetcher import fetch_all
from reporter import print_report, save_csv


def load_symbols(csv_path: str = "symbols.csv") -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trend Break Scanner — flags trend-breaking symbols via yfinance daily data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--type", choices=["stock", "commodity", "all"], default="all",
                        help="Filter by asset type (default: all)")
    parser.add_argument("--min-score", type=int, default=1, metavar="N",
                        help="Minimum |score| to display (default: 1)")
    parser.add_argument("--days", type=int, default=config.LOOKBACK_DAYS, metavar="N",
                        help=f"Days of history to fetch (default: {config.LOOKBACK_DAYS})")
    parser.add_argument("--signal-window", type=int, default=config.SIGNAL_WINDOW, metavar="N",
                        help=f"Recent-break look-back window in days (default: {config.SIGNAL_WINDOW})")
    parser.add_argument("--donchian", type=int, default=config.DONCHIAN_PERIOD, metavar="N",
                        help=f"Donchian channel period (default: {config.DONCHIAN_PERIOD})")
    parser.add_argument("--output", type=str, default=None, metavar="FILE",
                        help="Save full results to this CSV file")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force fresh download, ignore cached data")
    args = parser.parse_args()

    # ── Load and filter symbols ──
    symbols = load_symbols("symbols.csv")
    # Derive asset_type from URL path if not already present
    for s in symbols:
        if "asset_type" not in s:
            url = s.get("url", "")
            parts = url.strip("/").split("/")
            s["asset_type"] = parts[1].capitalize() if len(parts) > 2 else "Stock"
        if "asset_name" not in s:
            s["asset_name"] = s.get("name", "") or s["symbol"]
    if args.type != "all":
        symbols = [s for s in symbols if s["asset_type"].lower() == args.type.lower()]

    # ── Build symbol → ticker mapping ──
    # Use SYMBOL_MAP when available; otherwise assume yfinance ticker == symbol
    sym_ticker: dict[str, str] = {}
    for s in symbols:
        sym = s["symbol"]
        sym_ticker[sym] = config.SYMBOL_MAP.get(sym, sym)

    print(f"[i] {len(sym_ticker)} symbols ({sum(1 for s in sym_ticker if s in config.SYMBOL_MAP)} mapped, "
          f"{sum(1 for s in sym_ticker if s not in config.SYMBOL_MAP)} auto-derived)")
    print(f"Fetching data for {len(sym_ticker)} symbols "
          f"({args.days} days, cache={'off' if args.no_cache else 'on'}) …")

    # ── Download data ──
    data = fetch_all(
        sym_ticker,
        period_days=args.days,
        use_cache=not args.no_cache,
        workers=config.FETCH_WORKERS,
    )
    print(f"  → received data for {len(data)} symbols")

    # ── Analyse ──
    name_map = {s["symbol"]: s["asset_name"] for s in symbols}
    type_map = {s["symbol"]: s["asset_type"] for s in symbols}

    print(f"Analysing (signal_window={args.signal_window}d, "
          f"donchian={args.donchian}d) …")

    signals: list[TrendSignal] = []
    failed: list[str] = []

    for sym, df in data.items():
        min_bars = config.LONG_MA_PERIOD + 10
        if df is None or len(df) < min_bars:
            failed.append(f"{sym} (too few bars: {0 if df is None else len(df)})")
            continue
        try:
            sig = analyze(
                symbol=sym,
                asset_name=name_map.get(sym, sym),
                asset_type=type_map.get(sym, "Unknown"),
                ticker=sym_ticker[sym],
                df=df,
                short_ma=config.SHORT_MA_PERIOD,
                medium_ma=config.MEDIUM_MA_PERIOD,
                long_ma=config.LONG_MA_PERIOD,
                signal_window=args.signal_window,
                donchian_period=args.donchian,
            )
            signals.append(sig)
        except Exception as exc:
            failed.append(f"{sym} ({exc})")

    if failed:
        print(f"[!] Skipped {len(failed)} symbol(s): {', '.join(failed)}", file=sys.stderr)

    # ── Report ──
    print_report(signals, min_score=args.min_score)

    if args.output:
        save_csv(signals, args.output)


if __name__ == "__main__":
    main()
