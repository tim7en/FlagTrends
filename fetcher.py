"""
Fetches daily OHLCV data from yfinance with optional disk caching.
Uses parallel threads to download multiple tickers concurrently.
"""

from __future__ import annotations

import datetime
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(".cache")


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns produced by some yfinance versions."""
    if isinstance(df.columns, pd.MultiIndex):
        # Keep only the first level (OHLCV field names) and drop ticker level
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    # Drop rows where all OHLCV values are NaN
    df.dropna(how="all", inplace=True)
    return df


def fetch_ticker(
    ticker: str,
    period_days: int = 365,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Return a daily OHLCV DataFrame for *ticker*, or None on failure.

    Uses ``yf.Ticker.history()`` (per-object, thread-safe) rather than the
    global ``yf.download()`` which has shared state and is not safe to call
    concurrently from multiple threads.
    """
    safe_name = ticker.replace("=", "_").replace(".", "_")
    cache_file = CACHE_DIR / f"{safe_name}.parquet"

    if use_cache and cache_file.exists():
        mtime = datetime.datetime.fromtimestamp(cache_file.stat().st_mtime)
        if mtime.date() == datetime.date.today():
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass  # stale or corrupt cache – re-download

    try:
        end = datetime.date.today()
        # Fetch extra days so rolling windows (e.g. SMA200) are fully warmed up
        start = end - datetime.timedelta(days=period_days + 60)

        t = yf.Ticker(ticker)
        df = t.history(
            start=str(start),
            end=str(end),
            auto_adjust=True,
            actions=False,
        )
        df = _normalise_df(df)
        if df.empty:
            return None

        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_file)

        return df

    except Exception as exc:
        print(f"  [warn] {ticker}: {exc}", file=sys.stderr)
        return None


def fetch_all(
    symbol_ticker_map: dict[str, str],
    period_days: int = 365,
    use_cache: bool = True,
    workers: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Download data for every symbol in *symbol_ticker_map* in parallel.

    Multiple Libertex symbols can map to the same yfinance ticker (e.g. CL and
    WT both map to CL=F). The download is deduplicated and the result is
    shared between those symbols.
    """
    # Deduplicate: ticker -> [list of Libertex symbols]
    ticker_to_syms: dict[str, list[str]] = {}
    for sym, ticker in symbol_ticker_map.items():
        ticker_to_syms.setdefault(ticker, []).append(sym)

    ticker_data: dict[str, pd.DataFrame | None] = {}

    def _fetch(ticker: str) -> tuple[str, pd.DataFrame | None]:
        return ticker, fetch_ticker(ticker, period_days, use_cache)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, t): t for t in ticker_to_syms}
        for future in as_completed(futures):
            ticker, df = future.result()
            ticker_data[ticker] = df

    # Map results back to Libertex symbols
    results: dict[str, pd.DataFrame] = {}
    for ticker, syms in ticker_to_syms.items():
        df = ticker_data.get(ticker)
        if df is not None:
            for sym in syms:
                results[sym] = df.copy()

    return results
