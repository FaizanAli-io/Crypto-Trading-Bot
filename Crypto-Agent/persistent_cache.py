import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class CandleCache:
    """SQLite-backed persistent cache for OHLCV candles.

    Schema:
      candles(symbol TEXT, interval TEXT, ts INTEGER, open REAL, high REAL,
              low REAL, close REAL, volume REAL,
              PRIMARY KEY(symbol, interval, ts))
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.execute(
            "PRAGMA journal_mode=WAL"
        )  # Write-Ahead Logging for better concurrency
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    ts INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY(symbol, interval, ts)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_candles_symbol_interval_ts
                ON candles(symbol, interval, ts)
                """
            )

    def upsert_klines(self, symbol: str, interval: str, klines: List[List]):
        """Upsert Binance klines (list of lists). Expects ts in ms at index 0.
        Columns order: [ts, open, high, low, close, volume, close_time, ...]
        """
        if not klines:
            return 0
        rows = [
            (
                symbol,
                interval,
                int(k[0]),
                float(k[1]),
                float(k[2]),
                float(k[3]),
                float(k[4]),
                float(k[5]),
            )
            for k in klines
        ]
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO candles
                (symbol, interval, ts, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def upsert_candles(self, symbol: str, interval: str, candles: List[Dict]):
        """Upsert candle dicts with keys: timestamp_ms, open, high, low, close, volume"""
        if not candles:
            return 0
        rows = [
            (
                symbol,
                interval,
                int(c["timestamp_ms"]),
                float(c["open"]),
                float(c["high"]),
                float(c["low"]),
                float(c["close"]),
                float(c["volume"]),
            )
            for c in candles
        ]
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO candles
                (symbol, interval, ts, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def get_last_candles(
        self, symbol: str, interval: str, n: int
    ) -> List[Tuple[int, float, float, float, float, float]]:
        """Return last n candles ordered ascending by ts.
        Returns list of tuples (ts_ms, open, high, low, close, volume).
        """
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM candles
                WHERE symbol=? AND interval=?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (symbol, interval, int(n)),
            )
            rows = cur.fetchall()
        # Return ascending by ts
        return list(reversed(rows))

    def get_range(
        self, symbol: str, interval: str, start_ts: int, end_ts: int
    ) -> List[Tuple[int, float, float, float, float, float]]:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM candles
                WHERE symbol=? AND interval=? AND ts BETWEEN ? AND ?
                ORDER BY ts ASC
                """,
                (symbol, interval, int(start_ts), int(end_ts)),
            )
            rows = cur.fetchall()
        return rows
