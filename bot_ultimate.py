# bot_ultimate.py — ULTIMATE TRADING BOT v4.0
# Technical Analysis + Sentiment Analysis + News Monitoring + Adaptive Learning
# 100% FREE - No paid subscriptions required
#
# ============================================================================
# v4.0 UPDATES (Feb 13-14, 2026) - Weekend 3:
# ============================================================================
# CRITICAL FIXES:
#   ✅ ATR-based dynamic Stop-Loss & Take-Profit (replaces fixed bps)
#   ✅ Trend Filter (EMA50) - blocks mean-reversion in downtrends
#   ✅ Bollinger Bands as additional entry confirmation
#   ✅ Risk-based Position Sizing (1-2% risk per trade)
#   ✅ Fixed: double CoinGecko API call (was 16x/cycle, now 1x)
#   ✅ Fixed: duplicate except block in fetch_bars()
#   ✅ Fixed: hardcoded 5min timeframe (now reads config)
#
# ARCHITECTURE:
#   Bot reads optimized parameters from optimizer_params.json
#   Separate optimizer.py runs alongside and tunes parameters
#
# New Railway Variables:
#   - ATR_SL_MULTIPLIER (default: 1.5)
#   - ATR_TP_MULTIPLIER (default: 3.0)
#   - RISK_PER_TRADE_PCT (default: 1.5)
#   - TREND_FILTER_ENABLED (default: true)
#   - BB_FILTER_ENABLED (default: true)
#   - BB_PERIOD (default: 20)
#   - BB_STD (default: 2.0)
#   - EMA_TREND_PERIOD (default: 50)
# ============================================================================

import os
import sys
import time
import json
import sqlite3
import traceback
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import feedparser
import requests
from urllib.parse import quote

# Google Sheets integration (optional)
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging():
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('bot_ultimate.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


# ============================================================================
# CONFIG
# ============================================================================
@dataclass
class Config:
    # Trading params
    symbols: List[str]
    max_pos_pct: float
    poll_seconds: int

    # Technical params
    lookback_bars: int
    vwap_window: int
    rsi_len: int
    rsi_buy: float
    rsi_sell: float
    entry_bps: float
    tp_bps: float       # fallback only — ATR-based TP preferred
    sl_bps: float       # fallback only — ATR-based SL preferred

    # NEW v4.0: ATR-based TP/SL
    atr_sl_multiplier: float
    atr_tp_multiplier: float
    risk_per_trade_pct: float

    # NEW v4.0: Trend filter
    trend_filter_enabled: bool
    ema_trend_period: int

    # NEW v4.0: Bollinger Bands filter
    bb_filter_enabled: bool
    bb_period: int
    bb_std: float

    # Risk management
    max_drawdown_pct: float
    max_trades_per_day: int
    min_volume: int
    cooldown_sec: int

    # Sentiment params
    sentiment_enabled: bool
    sentiment_min_threshold: float
    sentiment_weight: float

    # Data params
    db_path: str
    data_feed: str
    timeframe_str: str
    timeframe_minutes: int
    fractional_enabled: bool


def load_config() -> Config:
    """Load configuration from environment variables + optimizer overrides"""
    symbols = os.getenv("SYMBOLS", "NVDA,AMD,MSFT").replace(" ", "").split(",")

    # Try to load optimizer overrides
    overrides = {}
    try:
        if os.path.exists("optimizer_params.json"):
            with open("optimizer_params.json", "r") as f:
                overrides = json.load(f)
            logger.info(f"Loaded optimizer overrides: {list(overrides.keys())}")
    except Exception as e:
        logger.warning(f"Could not load optimizer params: {e}")

    def get(key, default, type_fn=str):
        """Get from overrides first, then env, then default"""
        if key.lower() in overrides:
            return type_fn(overrides[key.lower()])
        return type_fn(os.getenv(key, str(default)))

    # Parse timeframe
    tf_str = get("TIMEFRAME_STR", "5Min", str)
    tf_minutes = int(re.search(r'\d+', tf_str).group()) if re.search(r'\d+', tf_str) else 5

    return Config(
        symbols=symbols,
        max_pos_pct=get("MAX_POS_PCT", 0.10, float),
        poll_seconds=get("POLL_SECONDS", 30, int),

        lookback_bars=get("LOOKBACK_BARS", 300, int),
        vwap_window=get("VWAP_WINDOW", 30, int),
        rsi_len=get("RSI_LEN", 5, int),
        rsi_buy=get("RSI_BUY", 30, float),
        rsi_sell=get("RSI_SELL", 55, float),
        entry_bps=get("ENTRY_BPS", 20, float),
        tp_bps=get("TP_BPS", 80, float),
        sl_bps=get("SL_BPS", 50, float),

        # v4.0
        atr_sl_multiplier=get("ATR_SL_MULTIPLIER", 1.5, float),
        atr_tp_multiplier=get("ATR_TP_MULTIPLIER", 3.0, float),
        risk_per_trade_pct=get("RISK_PER_TRADE_PCT", 1.5, float),
        trend_filter_enabled=get("TREND_FILTER_ENABLED", "true", str).lower() == "true",
        ema_trend_period=get("EMA_TREND_PERIOD", 50, int),
        bb_filter_enabled=get("BB_FILTER_ENABLED", "true", str).lower() == "true",
        bb_period=get("BB_PERIOD", 20, int),
        bb_std=get("BB_STD", 2.0, float),

        max_drawdown_pct=get("MAX_DRAWDOWN_PCT", 10.0, float),
        max_trades_per_day=get("MAX_TRADES_PER_DAY", 3, int),
        min_volume=get("MIN_VOLUME", 10000, int),
        cooldown_sec=get("COOLDOWN_SEC", 60, int),

        sentiment_enabled=get("SENTIMENT_ENABLED", "true", str).lower() == "true",
        sentiment_min_threshold=get("SENTIMENT_MIN_THRESHOLD", -0.5, float),
        sentiment_weight=get("SENTIMENT_WEIGHT", 0.3, float),

        db_path=get("DB_PATH", "bot_ultimate.db", str),
        data_feed=get("DATA_FEED", "iex", str).lower(),
        timeframe_str=tf_str,
        timeframe_minutes=tf_minutes,
        fractional_enabled=get("FRACTIONAL_ENABLED", "true", str).lower() == "true",
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def bps_to_mult(bps: float) -> float:
    return 1.0 + (bps / 10000.0)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    rs = up.rolling(n).mean() / (down.rolling(n).mean() + 1e-9)
    return 100 - (100 / (1 + rs))


def rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"].astype(float)
    v = df["volume"].astype(float)
    return pv.rolling(window).sum() / (v.rolling(window).sum() + 1e-12)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 0.0
    try:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_value = tr.rolling(period).mean().iloc[-1]
        return float(atr_value) if not pd.isna(atr_value) else 0.0
    except Exception as e:
        logger.warning(f"ATR calculation failed: {e}")
        return 0.0


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> Tuple[float, float, float]:
    """Calculate Bollinger Bands. Returns (upper, middle, lower)."""
    if len(df) < period:
        return (0.0, 0.0, 0.0)
    try:
        close = df['close'].astype(float)
        middle = close.rolling(period).mean().iloc[-1]
        std = close.rolling(period).std().iloc[-1]
        upper = middle + (std * std_mult)
        lower = middle - (std * std_mult)
        return (float(upper), float(middle), float(lower))
    except Exception:
        return (0.0, 0.0, 0.0)


def calculate_ema(df: pd.DataFrame, period: int) -> float:
    """Calculate EMA for trend detection."""
    if len(df) < period:
        return 0.0
    try:
        return float(df['close'].astype(float).ewm(span=period, adjust=False).mean().iloc[-1])
    except Exception:
        return 0.0


def calculate_market_volatility(df: pd.DataFrame) -> str:
    if len(df) < 20:
        return "UNKNOWN"
    try:
        returns = df['close'].pct_change().tail(20)
        std_dev = returns.std() * 100
        if std_dev < 1.0:
            return "LOW"
        elif std_dev < 2.0:
            return "NORMAL"
        elif std_dev < 3.5:
            return "HIGH"
        else:
            return "EXTREME"
    except Exception:
        return "UNKNOWN"


def is_market_hours() -> bool:
    import pytz
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    if now_ny.weekday() >= 5:
        return False
    start_time = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return start_time <= now_ny <= end_time


def can_trade_now() -> bool:
    if not is_market_hours():
        return False
    import pytz
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    trade_start = now_ny.replace(hour=11, minute=30, second=0, microsecond=0)
    return now_ny >= trade_start


# ============================================================================
# CRYPTO RISK MONITOR (with cache to avoid API spam)
# ============================================================================
class CryptoMonitor:
    """Cached CoinGecko monitor — calls API max once per 5 minutes."""

    def __init__(self):
        self._cache = None
        self._cache_time = 0
        self._cache_ttl = 300  # 5 minutes

    def get_data(self) -> dict:
        now = time.time()
        if self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            btc = data['bitcoin']['usd_24h_change']
            eth = data['ethereum']['usd_24h_change']

            if btc < -10.0:
                risk = "CRASH"
            elif btc < -5.0:
                risk = "HIGH"
            elif btc < -2.0:
                risk = "MEDIUM"
            elif btc > 3.0:
                risk = "BULLISH"
            else:
                risk = "NORMAL"

            self._cache = {
                'btc_change_24h': btc,
                'eth_change_24h': eth,
                'risk_level': risk,
                'should_pause': btc < -10.0
            }
            self._cache_time = now
            return self._cache

        except Exception as e:
            logger.warning(f"Crypto API failed: {e}")
            return {
                'btc_change_24h': 0.0, 'eth_change_24h': 0.0,
                'risk_level': 'UNKNOWN', 'should_pause': False
            }


# ============================================================================
# SENTIMENT ANALYZER (100% FREE)
# ============================================================================
class SentimentAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.rss_feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        ]
        self.positive_keywords = [
            'surge', 'soar', 'gain', 'rally', 'jump', 'rise', 'profit', 'beat',
            'upgrade', 'bullish', 'growth', 'innovation', 'breakthrough', 'record',
            'strong', 'positive', 'optimistic', 'boom', 'success', 'expansion'
        ]
        self.negative_keywords = [
            'drop', 'plunge', 'fall', 'crash', 'loss', 'miss', 'cut', 'downgrade',
            'bearish', 'decline', 'warning', 'weak', 'concern', 'risk', 'trouble',
            'lawsuit', 'investigation', 'scandal', 'delay', 'failure', 'slump'
        ]
        self.news_cache: Dict[str, List[dict]] = {}
        self.cache_timestamp: Dict[str, float] = {}
        self.cache_duration = 300
        logger.info("SentimentAnalyzer initialized (100% FREE sources)")

    def fetch_rss_feeds(self) -> List[dict]:
        all_news = []
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:
                    all_news.append({
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'published': entry.get('published', ''),
                        'link': entry.get('link', ''),
                        'source': 'RSS'
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch RSS {feed_url}: {e}")
        return all_news

    def fetch_google_news(self, symbol: str) -> List[dict]:
        news = []
        try:
            query = f"{symbol} stock"
            url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                news.append({
                    'title': entry.get('title', ''),
                    'published': entry.get('published', ''),
                    'link': entry.get('link', ''),
                    'source': 'Google News'
                })
        except Exception as e:
            logger.warning(f"Failed to fetch Google News for {symbol}: {e}")
        return news

    def get_recent_news(self, symbol: str, max_age_hours: int = 24) -> List[dict]:
        cache_key = f"{symbol}_{max_age_hours}"
        if cache_key in self.news_cache:
            age = time.time() - self.cache_timestamp[cache_key]
            if age < self.cache_duration:
                return self.news_cache[cache_key]

        all_news = []
        rss_news = self.fetch_rss_feeds()
        company = self._get_company_name(symbol).lower()
        for news in rss_news:
            text = (news['title'] + ' ' + news.get('summary', '')).lower()
            if symbol.lower() in text or company in text:
                all_news.append(news)

        google_news = self.fetch_google_news(symbol)
        all_news.extend(google_news)

        self.news_cache[cache_key] = all_news
        self.cache_timestamp[cache_key] = time.time()
        return all_news

    def _get_company_name(self, symbol: str) -> str:
        mapping = {
            'NVDA': 'NVIDIA', 'MSFT': 'Microsoft', 'AMD': 'AMD',
            'AAPL': 'Apple', 'GOOGL': 'Google', 'TSLA': 'Tesla',
            'META': 'Meta', 'AMZN': 'Amazon', 'PLTR': 'Palantir',
        }
        return mapping.get(symbol, symbol)

    def analyze_sentiment(self, news_list: List[dict]) -> dict:
        if not news_list:
            return {'sentiment': 0.0, 'confidence': 0.0, 'positive_count': 0,
                    'negative_count': 0, 'neutral_count': 0, 'total_articles': 0,
                    'sample_headlines': []}

        pos_count = neg_count = neu_count = 0
        scores = []
        headlines = []

        for news in news_list[:15]:
            text = (news['title'] + ' ' + news.get('summary', '')).lower()
            pos = sum(1 for kw in self.positive_keywords if kw in text)
            neg = sum(1 for kw in self.negative_keywords if kw in text)

            if pos > neg:
                scores.append(min(1.0, pos * 0.2))
                pos_count += 1
                headlines.append(('POSITIVE', news['title']))
            elif neg > pos:
                scores.append(-min(1.0, neg * 0.2))
                neg_count += 1
                headlines.append(('NEGATIVE', news['title']))
            else:
                scores.append(0.0)
                neu_count += 1

        total = pos_count + neg_count + neu_count
        sentiment = float(np.mean(scores)) if scores else 0.0
        confidence = (max(pos_count, neg_count) / total * min(1.0, total / 10.0)) if total > 0 else 0.0

        return {
            'sentiment': sentiment, 'confidence': confidence,
            'positive_count': pos_count, 'negative_count': neg_count,
            'neutral_count': neu_count, 'total_articles': total,
            'sample_headlines': headlines[:5]
        }


# ============================================================================
# DATABASE
# ============================================================================
class TradingDB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.init_schema()

    def _table_exists(self, name: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,))
        return cur.fetchone() is not None

    def _cols(self, table: str) -> List[str]:
        return [r[1] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()]

    def _ensure_cols(self, table: str, cols_sql: Dict[str, str]):
        if not self._table_exists(table):
            return
        existing = set(self._cols(table))
        for col, coldef in cols_sql.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coldef};")
        self.conn.commit()

    def init_schema(self):
        self.conn.execute("""CREATE TABLE IF NOT EXISTS bars (
            symbol TEXT NOT NULL, ts_utc TEXT NOT NULL, timeframe TEXT NOT NULL,
            open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
            close REAL NOT NULL, volume REAL NOT NULL,
            PRIMARY KEY(symbol, ts_utc, timeframe));""")

        self.conn.execute("""CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL, symbol TEXT NOT NULL,
            action TEXT NOT NULL, reason TEXT NOT NULL);""")

        self.conn.execute("""CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY, ts_utc TEXT, symbol TEXT,
            side TEXT, qty REAL, status TEXT, raw_json TEXT);""")

        self.conn.execute("""CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL, symbol TEXT NOT NULL, qty REAL NOT NULL);""")

        self.conn.execute("""CREATE TABLE IF NOT EXISTS eod_reports (
            date TEXT PRIMARY KEY, start_equity REAL, end_equity REAL,
            pnl_realized REAL, pnl_unrealized REAL, total_trades INTEGER,
            winning_trades INTEGER, fees_estimated REAL, report_json TEXT);""")

        self.conn.execute("""CREATE TABLE IF NOT EXISTS news_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL, symbol TEXT NOT NULL, title TEXT NOT NULL,
            source TEXT, sentiment REAL, link TEXT);""")

        self.conn.commit()

        self._ensure_cols("signals", {
            "price": "REAL", "vwap": "REAL", "rsi": "REAL",
            "equity": "REAL", "cash": "REAL",
            "position_qty": "REAL", "position_value": "REAL", "max_pos_value": "REAL",
            "entry_price": "REAL", "tp_price": "REAL", "sl_price": "REAL",
            "sentiment": "REAL", "sentiment_confidence": "REAL", "news_count": "INTEGER",
            "atr": "REAL", "ema50": "REAL", "bb_lower": "REAL",
            "trend_ok": "INTEGER", "bb_ok": "INTEGER",
            "raw_json": "TEXT",
        })

        self._ensure_cols("positions", {
            "market_value": "REAL", "avg_entry_price": "REAL",
            "unrealized_pl": "REAL", "raw_json": "TEXT",
        })

        self.conn.commit()
        logger.info("Database schema initialized")

    def upsert_bar(self, symbol, ts, timeframe, o, h, l, c, v):
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)",
                (symbol, iso(ts), timeframe, float(o), float(h), float(l), float(c), float(v)))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert bar for {symbol}: {e}")

    def insert_signal(self, ts, symbol, action, reason, **kw):
        self.conn.execute(
            """INSERT INTO signals(ts_utc,symbol,action,reason,price,vwap,rsi,
               equity,cash,position_qty,position_value,max_pos_value,
               entry_price,tp_price,sl_price,sentiment,sentiment_confidence,news_count,
               atr,ema50,bb_lower,trend_ok,bb_ok,raw_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (iso(ts), symbol, action, reason,
             kw.get("price"), kw.get("vwap"), kw.get("rsi"),
             kw.get("equity"), kw.get("cash"),
             kw.get("position_qty"), kw.get("position_value"), kw.get("max_pos_value"),
             kw.get("entry_price"), kw.get("tp_price"), kw.get("sl_price"),
             kw.get("sentiment"), kw.get("sentiment_confidence"), kw.get("news_count"),
             kw.get("atr"), kw.get("ema50"), kw.get("bb_lower"),
             1 if kw.get("trend_ok") else 0, 1 if kw.get("bb_ok") else 0,
             json.dumps(kw.get("raw", {}), ensure_ascii=False)))
        self.conn.commit()

    def upsert_order(self, order_obj):
        d = order_obj.model_dump(mode='json') if hasattr(order_obj, "model_dump") else dict(order_obj)
        self.conn.execute(
            "INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?)",
            (d.get("id"), iso(utc_now()), d.get("symbol"), str(d.get("side")),
             float(d.get("qty")) if d.get("qty") is not None else None,
             str(d.get("status")), json.dumps(d, ensure_ascii=False)))
        self.conn.commit()

    def insert_position(self, ts, pos_obj):
        d = pos_obj.model_dump() if hasattr(pos_obj, "model_dump") else dict(pos_obj)
        self.conn.execute(
            """INSERT INTO positions(ts_utc,symbol,qty,market_value,avg_entry_price,unrealized_pl,raw_json)
               VALUES (?,?,?,?,?,?,?)""",
            (iso(ts), d.get("symbol"), float(d.get("qty")),
             float(d.get("market_value")) if d.get("market_value") is not None else None,
             float(d.get("avg_entry_price")) if d.get("avg_entry_price") is not None else None,
             float(d.get("unrealized_pl")) if d.get("unrealized_pl") is not None else None,
             json.dumps(d, ensure_ascii=False, default=str)))
        self.conn.commit()

    def save_news(self, symbol, title, source, sentiment, link=""):
        self.conn.execute(
            "INSERT INTO news_cache(ts_utc,symbol,title,source,sentiment,link) VALUES (?,?,?,?,?,?)",
            (iso(utc_now()), symbol, title, source, sentiment, link))
        self.conn.commit()

    def save_eod_report(self, date, report):
        self.conn.execute(
            """INSERT OR REPLACE INTO eod_reports VALUES (?,?,?,?,?,?,?,?,?)""",
            (date, report.get("start_equity"), report.get("end_equity"),
             report.get("pnl_realized"), report.get("pnl_unrealized"),
             report.get("total_trades"), report.get("winning_trades"),
             report.get("fees_estimated"), json.dumps(report, ensure_ascii=False)))
        self.conn.commit()

    def get_entry_price(self, symbol):
        cur = self.conn.execute(
            "SELECT entry_price FROM signals WHERE symbol=? AND action='BUY' AND entry_price IS NOT NULL ORDER BY ts_utc DESC LIMIT 1",
            (symbol,))
        row = cur.fetchone()
        return float(row[0]) if row else None

    def get_start_equity(self, date):
        cur = self.conn.execute(
            "SELECT end_equity FROM eod_reports WHERE date < ? ORDER BY date DESC LIMIT 1", (date,))
        row = cur.fetchone()
        return float(row[0]) if row else None

    def get_trades_today(self, symbol, date):
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE symbol=? AND action='BUY' AND DATE(ts_utc)=?",
            (symbol, date))
        row = cur.fetchone()
        return int(row[0]) if row else 0


# ============================================================================
# GOOGLE SHEETS REPORTER
# ============================================================================
class GoogleSheetsReporter:
    def __init__(self, sheet_name="Trading Bot Dashboard"):
        self.enabled = GSHEETS_AVAILABLE
        self.sheet = None
        if not self.enabled:
            logger.info("Google Sheets: DISABLED (missing libraries)")
            return
        try:
            creds_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
            if not creds_json:
                logger.warning("Google Sheets: No credentials found")
                self.enabled = False
                return
            creds_dict = json.loads(creds_json)
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
            try:
                self.sheet = client.open(sheet_name).sheet1
            except gspread.exceptions.SpreadsheetNotFound:
                sp = client.create(sheet_name)
                sp.share('', perm_type='anyone', role='reader')
                self.sheet = sp.sheet1
                self.sheet.append_row(['Date','Time','Equity','P/L','P/L%','Positions','Cash','WinRate','Trades','DD%','BTC%','Risk'])
            logger.info(f"Google Sheets: Connected to '{sheet_name}'")
        except Exception as e:
            logger.error(f"Google Sheets: Auth failed | {e}")
            self.enabled = False

    def update_daily_stats(self, data):
        if not self.enabled or not self.sheet:
            return
        try:
            now = utc_now()
            pnl = data['equity'] - data['daily_start']
            pnl_pct = (pnl / data['daily_start'] * 100) if data['daily_start'] > 0 else 0
            wr = (data['winning_trades'] / data['total_trades'] * 100) if data['total_trades'] > 0 else 0
            self.sheet.append_row([
                now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'),
                f"{data['equity']:.2f}", f"{pnl:+.2f}", f"{pnl_pct:+.2f}",
                data['positions'], f"{data['cash']:.2f}", f"{wr:.1f}",
                data['total_trades'], f"{data['drawdown_pct']:.2f}",
                f"{data['crypto_data'].get('btc_change_24h',0):+.2f}",
                data['crypto_data'].get('risk_level', 'UNKNOWN')])
        except Exception as e:
            logger.error(f"Google Sheets: Update failed | {e}")


# ============================================================================
# BOT v4.0
# ============================================================================
class UltimateBot:
    def __init__(self, cfg: Config, db: TradingDB):
        self.cfg = cfg
        self.db = db

        key = os.environ["APCA_API_KEY_ID"]
        secret = os.environ["APCA_API_SECRET_KEY"]
        self.trading = TradingClient(key, secret, paper=True)
        self.data = StockHistoricalDataClient(key, secret)

        self.sentiment = SentimentAnalyzer() if cfg.sentiment_enabled else None
        self.sheets = GoogleSheetsReporter()
        self.crypto = CryptoMonitor()  # v4.0: single cached instance

        self.last_trade: Dict[str, float] = {s: 0.0 for s in cfg.symbols}
        self.entry_price: Dict[str, float] = {}
        self.start_equity: Optional[float] = None
        self.peak_equity: Optional[float] = None
        self.daily_peak_equity: Optional[float] = None
        self.daily_start_equity: Optional[float] = None
        self.symbol_cooldowns: Dict[str, datetime] = {}
        self.paused_until: Optional[datetime] = None

        self._load_entry_prices()

        logger.info(f"UltimateBot v4.0 | sentiment={'ON' if cfg.sentiment_enabled else 'OFF'} "
                     f"| trend_filter={'ON' if cfg.trend_filter_enabled else 'OFF'} "
                     f"| bb_filter={'ON' if cfg.bb_filter_enabled else 'OFF'} "
                     f"| ATR SL={cfg.atr_sl_multiplier}x TP={cfg.atr_tp_multiplier}x")
        logger.info(f"Symbols: {cfg.symbols} | max_pos: {cfg.max_pos_pct*100}% | risk/trade: {cfg.risk_per_trade_pct}%")

    def _load_entry_prices(self):
        for sym in self.cfg.symbols:
            entry = self.db.get_entry_price(sym)
            if entry:
                self.entry_price[sym] = entry
                logger.info(f"Loaded entry price for {sym}: ${entry:.2f}")

    # --- Risk checks ---
    def count_consecutive_losses(self, symbol):
        try:
            cur = self.db.conn.execute(
                "SELECT entry_price, price FROM signals WHERE symbol=? AND action='SELL' AND entry_price IS NOT NULL ORDER BY ts_utc DESC LIMIT 10",
                (symbol,))
            consecutive = 0
            for row in cur:
                if row[1] and row[0] and row[1] < row[0]:
                    consecutive += 1
                else:
                    break
            return consecutive
        except Exception:
            return 0

    def calculate_daily_drawdown(self, eq):
        if self.daily_peak_equity is None:
            self.daily_peak_equity = eq
        self.daily_peak_equity = max(self.daily_peak_equity, eq)
        return ((self.daily_peak_equity - eq) / self.daily_peak_equity * 100) if self.daily_peak_equity > 0 else 0.0

    def check_symbol_cooldown(self, symbol):
        if symbol in self.symbol_cooldowns:
            if utc_now() < self.symbol_cooldowns[symbol]:
                remaining = (self.symbol_cooldowns[symbol] - utc_now()).total_seconds() / 3600
                logger.info(f"{symbol} in cooldown for {remaining:.1f}h more")
                return False
            else:
                del self.symbol_cooldowns[symbol]
        losses = self.count_consecutive_losses(symbol)
        cooldown_days = int(os.getenv("CONSECUTIVE_LOSS_COOLDOWN_DAYS", "2"))
        if losses >= 3:
            self.symbol_cooldowns[symbol] = utc_now() + timedelta(days=cooldown_days)
            logger.warning(f"{symbol}: {losses} consecutive losses - COOLDOWN {cooldown_days} days")
            return False
        return True

    def check_daily_loss_limit(self, eq):
        if self.paused_until:
            if utc_now() < self.paused_until:
                return False
            self.paused_until = None
        if self.daily_start_equity is None:
            self.daily_start_equity = eq
        pnl_pct = ((eq - self.daily_start_equity) / self.daily_start_equity) * 100
        max_loss = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))
        if pnl_pct < -max_loss:
            import pytz
            tz_ny = pytz.timezone('America/New_York')
            midnight = (datetime.now(tz_ny) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            self.paused_until = midnight.astimezone(timezone.utc)
            logger.critical(f"DAILY LOSS LIMIT: {pnl_pct:.2f}% - PAUSED until midnight")
            return False
        return True

    def check_drawdown(self, eq):
        if self.peak_equity is None:
            self.peak_equity = eq
            return False
        self.peak_equity = max(self.peak_equity, eq)
        dd = ((self.peak_equity - eq) / self.peak_equity) * 100
        if dd >= self.cfg.max_drawdown_pct:
            logger.critical(f"MAX DRAWDOWN | Peak: ${self.peak_equity:.2f} | Current: ${eq:.2f} | DD: {dd:.2f}%")
            return True
        return False

    def can_trade_today(self, symbol):
        today = utc_now().strftime("%Y-%m-%d")
        return self.db.get_trades_today(symbol, today) < self.cfg.max_trades_per_day

    # --- v4.0: Risk-based position sizing ---
    def calculate_position_size(self, equity: float, price: float, atr: float, cash: float) -> float:
        """
        Calculate position size based on risk percentage and ATR stop.
        Risk per trade = cfg.risk_per_trade_pct of equity.
        """
        if price <= 0 or atr <= 0:
            return self._fallback_qty(equity, price, cash)

        risk_amount = equity * (self.cfg.risk_per_trade_pct / 100.0)
        sl_distance = atr * self.cfg.atr_sl_multiplier

        if sl_distance <= 0:
            return self._fallback_qty(equity, price, cash)

        qty = risk_amount / sl_distance

        # Cap by max position size and available cash
        max_qty_by_pct = (equity * self.cfg.max_pos_pct) / price
        max_qty_by_cash = (cash * 0.95) / price
        qty = min(qty, max_qty_by_pct, max_qty_by_cash)

        if not self.cfg.fractional_enabled:
            qty = int(qty)

        return max(0.0, qty)

    def _fallback_qty(self, equity, price, cash):
        """Fallback to fixed percentage sizing if ATR unavailable."""
        if price <= 0:
            return 0.0
        max_investment = min(equity * self.cfg.max_pos_pct, cash * 0.95)
        qty = max_investment / price
        if not self.cfg.fractional_enabled:
            qty = int(qty)
        return max(0.0, qty)

    # --- Data ---
    def fetch_bars(self, sym):
        try:
            tf_min = self.cfg.timeframe_minutes
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame(tf_min, TimeFrameUnit.Minute),
                start=utc_now() - timedelta(minutes=self.cfg.lookback_bars * tf_min),
                end=utc_now(),
                feed=self.cfg.data_feed,
            )
            df = self.data.get_stock_bars(req).df
            if df is None or len(df) == 0:
                return pd.DataFrame()
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                df = df[df["symbol"] == sym].set_index("timestamp")
            return df.sort_index()
        except Exception as e:
            logger.error(f"fetch_bars({sym}): {type(e).__name__}: {e}")
            return pd.DataFrame()

    def validate_bars(self, df, sym):
        if df.empty:
            return False
        required = max(self.cfg.vwap_window + 5, self.cfg.rsi_len + 5, self.cfg.ema_trend_period + 5, self.cfg.bb_period + 5)
        if len(df) < required:
            logger.debug(f"{sym}: Not enough bars ({len(df)}/{required})")
            return False
        if pd.isna(df["close"].iloc[-1]) or df["close"].iloc[-1] <= 0:
            return False
        if df["volume"].tail(10).sum() < self.cfg.min_volume:
            return False
        return True

    def submit_mkt(self, sym, side, qty):
        if qty <= 0 or (self.cfg.fractional_enabled and qty < 0.01):
            return None
        try:
            qty = round(qty, 4) if self.cfg.fractional_enabled else int(qty)
            order = self.trading.submit_order(
                MarketOrderRequest(symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY))
            self.db.upsert_order(order)
            self.last_trade[sym] = time.time()
            logger.info(f"{sym}: {side} order submitted | qty={qty}")
            return order
        except Exception as e:
            logger.error(f"{sym}: Order failed | {e}")
            return None

    # --- Main processing ---
    def run_once(self):
        try:
            acct = self.trading.get_account()
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return

        equity = float(acct.equity)
        cash = float(acct.cash)

        if self.start_equity is None:
            self.start_equity = equity
        if self.check_drawdown(equity):
            raise KeyboardInterrupt("Max drawdown exceeded")

        try:
            positions = {p.symbol: p for p in self.trading.get_all_positions()}
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            positions = {}

        # v4.0: Fetch crypto data ONCE per cycle
        crypto_data = self.crypto.get_data()

        logger.info(f"Account: equity=${equity:.2f} cash=${cash:.2f} | CRYPTO BTC:{crypto_data['btc_change_24h']:+.1f}% Risk:{crypto_data['risk_level']}")

        for sym in self.cfg.symbols:
            try:
                self._process_symbol(sym, equity, cash, positions, crypto_data)
            except Exception as e:
                logger.error(f"{sym}: Processing error | {e}")
                traceback.print_exc()

        # Snapshot positions
        ts = utc_now()
        for p in self.trading.get_all_positions():
            self.db.insert_position(ts, p)

        # Google Sheets update
        try:
            if self.sheets.enabled and self.daily_start_equity:
                daily_dd = self.calculate_daily_drawdown(equity)
                cur = self.db.conn.execute(
                    "SELECT COUNT(*), SUM(CASE WHEN price > entry_price THEN 1 ELSE 0 END) FROM signals WHERE action='SELL' AND DATE(ts_utc)=?",
                    (utc_now().strftime('%Y-%m-%d'),))
                row = cur.fetchone()
                self.sheets.update_daily_stats({
                    'equity': equity, 'daily_start': self.daily_start_equity,
                    'positions': len(positions), 'cash': cash,
                    'total_trades': int(row[0]) if row and row[0] else 0,
                    'winning_trades': int(row[1]) if row and row[1] else 0,
                    'drawdown_pct': daily_dd, 'crypto_data': crypto_data})
        except Exception as e:
            logger.warning(f"Google Sheets update failed: {e}")

    def _process_symbol(self, sym, equity, cash, positions, crypto_data):
        df = self.fetch_bars(sym)
        if not self.validate_bars(df, sym):
            return

        # Store bars
        for ts, row in df.tail(3).iterrows():
            ts_dt = ts.to_pydatetime()
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            self.db.upsert_bar(sym, ts_dt, self.cfg.timeframe_str,
                               row["open"], row["high"], row["low"], row["close"], row["volume"])

        # === INDICATORS ===
        price = float(df["close"].iloc[-1])
        vwap_val = float(rolling_vwap(df, self.cfg.vwap_window).iloc[-1])
        rsi_val = float(rsi(df["close"].astype(float), self.cfg.rsi_len).iloc[-1])
        atr_val = calculate_atr(df, 14)
        volatility = calculate_market_volatility(df)

        # v4.0: Trend filter (EMA)
        ema50 = calculate_ema(df, self.cfg.ema_trend_period)
        trend_ok = (not self.cfg.trend_filter_enabled) or (price > ema50 and ema50 > 0)

        # v4.0: Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df, self.cfg.bb_period, self.cfg.bb_std)
        bb_ok = (not self.cfg.bb_filter_enabled) or (price < bb_lower and bb_lower > 0)

        logger.info(f"{sym} | ${price:.2f} | VWAP=${vwap_val:.2f} RSI={rsi_val:.1f} ATR=${atr_val:.2f} | "
                     f"EMA{self.cfg.ema_trend_period}=${ema50:.2f} trend={'OK' if trend_ok else 'BLOCKED'} | "
                     f"BB=[{bb_lower:.2f}-{bb_upper:.2f}] bb={'OK' if bb_ok else 'NO'} | Vol={volatility}")

        # Consecutive losses
        consecutive_losses = self.count_consecutive_losses(sym)
        if consecutive_losses > 0:
            logger.info(f"{sym} | Consecutive losses: {consecutive_losses}")

        # Daily drawdown
        daily_dd = self.calculate_daily_drawdown(equity)
        if daily_dd > 0.5:
            logger.info(f"Daily DD: {daily_dd:.2f}%")

        # Position info
        pos = positions.get(sym)
        pos_qty = float(pos.qty) if pos else 0.0

        # Entry price from Alpaca (survives restarts)
        entry = None
        if pos and hasattr(pos, "avg_entry_price") and pos.avg_entry_price:
            try:
                entry = float(pos.avg_entry_price)
                self.entry_price[sym] = entry
            except Exception:
                pass
        if entry is None and pos_qty == 0:
            entry = self.entry_price.get(sym)

        # v4.0: ATR-based TP/SL (with fallback to bps)
        if entry and atr_val > 0:
            tp_price = entry + (atr_val * self.cfg.atr_tp_multiplier)
            sl_price = entry - (atr_val * self.cfg.atr_sl_multiplier)
        elif entry:
            tp_price = entry * bps_to_mult(self.cfg.tp_bps)
            sl_price = entry * bps_to_mult(-self.cfg.sl_bps)
        else:
            tp_price = sl_price = None

        # === SENTIMENT ===
        sentiment_score = 0.0
        sentiment_confidence = 0.0
        news_count = 0
        if self.cfg.sentiment_enabled and self.sentiment:
            try:
                news_list = self.sentiment.get_recent_news(sym, max_age_hours=4)
                if news_list:
                    sd = self.sentiment.analyze_sentiment(news_list)
                    sentiment_score = sd['sentiment']
                    sentiment_confidence = sd['confidence']
                    news_count = sd['total_articles']
                    if sd['sample_headlines']:
                        logger.info(f"{sym} | Sentiment: {sentiment_score:+.2f} conf={sentiment_confidence:.2f}")
                    for h in sd['sample_headlines']:
                        self.db.save_news(sym, h[1], 'RSS', sentiment_score, "")
            except Exception as e:
                logger.warning(f"{sym}: Sentiment failed | {e}")

        # === SIGNALS ===
        # v4.0: Enhanced entry with trend + BB filters
        technical_entry = (
            (price <= vwap_val * bps_to_mult(-self.cfg.entry_bps)) and
            (rsi_val <= self.cfg.rsi_buy) and
            trend_ok
        )

        sentiment_ok = True
        if self.cfg.sentiment_enabled and news_count > 0:
            if sentiment_score < self.cfg.sentiment_min_threshold:
                sentiment_ok = False
                logger.info(f"{sym} | BLOCKED by negative sentiment ({sentiment_score:.2f})")

        entry_trigger = technical_entry and sentiment_ok

        exit_trigger = (price >= vwap_val) or (rsi_val >= self.cfg.rsi_sell)
        tp_hit = (tp_price is not None and price >= tp_price)
        sl_hit = (sl_price is not None and price <= sl_price)

        cooled = (time.time() - self.last_trade[sym]) >= self.cfg.cooldown_sec
        can_trade = self.can_trade_today(sym)

        # Safe guards
        safe = True
        if not self.check_daily_loss_limit(equity):
            safe = False
        if not self.check_symbol_cooldown(sym):
            safe = False
        if crypto_data['should_pause']:
            safe = False
            logger.warning(f"SAFE GUARD: Crypto crash (BTC {crypto_data['btc_change_24h']:.1f}%)")

        action = "HOLD"
        reason = "none"

        if pos_qty == 0:
            if entry_trigger and cooled and can_trade and safe:
                # v4.0: Risk-based position sizing
                # BB bonus: if Bollinger confirms, use higher risk per trade
                if bb_ok and self.cfg.bb_filter_enabled:
                    risk_mult = 1.5  # 50% bigger position
                else:
                    risk_mult = 1.0
                qty = self.calculate_position_size(equity, price, atr_val, cash) * risk_mult
                if qty > 0:
                    self.last_trade[sym] = time.time()
                    action = "BUY"
                    reason = "entry"
                    if not trend_ok:
                        reason += "+trend_override"
                    if bb_ok and self.cfg.bb_filter_enabled:
                        reason += "+bb_confirmed"
                    if sentiment_score > 0.3:
                        reason += "+sentiment"

                    logger.info(
                        f"{sym} | BUY {qty:.4f} @${price:.2f} | VWAP=${vwap_val:.2f} RSI={rsi_val:.1f} "
                        f"ATR=${atr_val:.2f} | TP=${price + atr_val * self.cfg.atr_tp_multiplier:.2f} "
                        f"SL=${price - atr_val * self.cfg.atr_sl_multiplier:.2f}")

                    order = self.submit_mkt(sym, OrderSide.BUY, qty)
                    if order:
                        self.entry_price[sym] = price
                        entry = price
                        if atr_val > 0:
                            tp_price = entry + (atr_val * self.cfg.atr_tp_multiplier)
                            sl_price = entry - (atr_val * self.cfg.atr_sl_multiplier)
                        else:
                            tp_price = entry * bps_to_mult(self.cfg.tp_bps)
                            sl_price = entry * bps_to_mult(-self.cfg.sl_bps)
                        cash -= qty * price
            elif entry_trigger and not can_trade:
                logger.info(f"{sym} | Max trades/day reached")
        else:
            if tp_price and sl_price:
                logger.info(f"{sym} | pos={pos_qty:.4f} entry=${entry:.2f} TP=${tp_price:.2f} SL=${sl_price:.2f} | price=${price:.2f}")

            if (exit_trigger or tp_hit or sl_hit) and cooled:
                action = "SELL"
                if tp_hit:
                    reason = "take_profit"
                elif sl_hit:
                    reason = "stop_loss"
                else:
                    reason = "technical_exit"

                qty = pos_qty if self.cfg.fractional_enabled else int(pos_qty)
                pnl_pct = ((price - entry) / entry * 100) if entry else 0

                logger.info(f"{sym} | SELL {qty:.4f} @${price:.2f} | {reason} | P/L: {pnl_pct:+.2f}%")
                order = self.submit_mkt(sym, OrderSide.SELL, qty)
                if order:
                    self.entry_price.pop(sym, None)

        if action != "HOLD":
            self.db.insert_signal(
                ts=utc_now(), symbol=sym, action=action, reason=reason,
                price=price, vwap=vwap_val, rsi=rsi_val,
                equity=equity, cash=cash,
                position_qty=pos_qty,
                position_value=float(pos.market_value) if pos and hasattr(pos, "market_value") else 0,
                max_pos_value=equity * self.cfg.max_pos_pct,
                entry_price=entry, tp_price=tp_price, sl_price=sl_price,
                sentiment=sentiment_score, sentiment_confidence=sentiment_confidence,
                news_count=news_count, atr=atr_val, ema50=ema50, bb_lower=bb_lower,
                trend_ok=trend_ok, bb_ok=bb_ok,
                raw={"technical_entry": technical_entry, "sentiment_ok": sentiment_ok,
                     "exit_trigger": exit_trigger, "tp_hit": tp_hit, "sl_hit": sl_hit})

    def generate_eod_report(self):
        today = utc_now().strftime("%Y-%m-%d")
        try:
            end_equity = float(self.trading.get_account().equity)
        except Exception:
            end_equity = 0.0

        start_equity = self.start_equity or self.db.get_start_equity(today) or end_equity
        cur = self.db.conn.execute(
            "SELECT action, entry_price, price FROM signals WHERE DATE(ts_utc)=? AND action IN ('BUY','SELL')", (today,))
        trades = cur.fetchall()
        total = len([t for t in trades if t[0] == 'SELL'])
        wins = sum(1 for a, e, p in trades if a == 'SELL' and e and p and p > e)
        pnl = end_equity - start_equity

        report = {"date": today, "start_equity": start_equity, "end_equity": end_equity,
                  "pnl_realized": pnl, "pnl_pct": (pnl / start_equity * 100) if start_equity > 0 else 0,
                  "total_trades": total, "winning_trades": wins,
                  "win_rate": (wins / total * 100) if total > 0 else 0,
                  "fees_estimated": total * 2 * 0.005}
        self.db.save_eod_report(today, report)

        logger.info(f"\n{'='*60}\nEOD REPORT {today}\n{'='*60}\n"
                     f"Start: ${start_equity:,.2f} | End: ${end_equity:,.2f} | P/L: ${pnl:+,.2f} ({report['pnl_pct']:+.2f}%)\n"
                     f"Trades: {total} | Wins: {wins} | Rate: {report['win_rate']:.1f}%\n{'='*60}")
        return report

    def run(self):
        logger.info("=" * 60)
        logger.info("ULTIMATE BOT START - v4.0")
        logger.info("=" * 60)
        logger.info(f"Symbols: {self.cfg.symbols}")
        logger.info(f"Strategy: Mean-Reversion + VWAP + RSI + EMA{self.cfg.ema_trend_period} + BB({self.cfg.bb_period},{self.cfg.bb_std})")
        logger.info(f"Risk: {self.cfg.risk_per_trade_pct}%/trade | ATR SL={self.cfg.atr_sl_multiplier}x TP={self.cfg.atr_tp_multiplier}x")
        logger.info(f"Max Trades/Day: {self.cfg.max_trades_per_day} | Position Cap: {self.cfg.max_pos_pct*100}%")
        logger.info("=" * 60)

        try:
            acct = self.trading.get_account()
            logger.info(f"Account: equity=${acct.equity} cash=${acct.cash}")
        except Exception as e:
            logger.error(f"Failed to get account: {e}")

        last_eod_date = None
        daily_equity_set = False

        while True:
            try:
                if not can_trade_now():
                    if not is_market_hours():
                        current_date = utc_now().strftime("%Y-%m-%d")
                        if current_date != last_eod_date:
                            self.generate_eod_report()
                            last_eod_date = current_date
                            daily_equity_set = False
                        logger.info("Market closed, sleeping...")
                    else:
                        logger.info("Pre-market hours (9:30-11:30 AM) - waiting...")
                    time.sleep(300)
                    continue

                if not daily_equity_set:
                    try:
                        acct = self.trading.get_account()
                        self.daily_start_equity = float(acct.equity)
                        self.daily_peak_equity = self.daily_start_equity
                        self.peak_equity = max(self.peak_equity or 0, self.daily_start_equity)
                        daily_equity_set = True
                        logger.info(f"Daily start equity: ${self.daily_start_equity:.2f}")
                    except Exception as e:
                        logger.error(f"Failed to set daily equity: {e}")

                self.run_once()

            except KeyboardInterrupt:
                logger.info("Stopping...")
                self.generate_eod_report()
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                traceback.print_exc()

            time.sleep(self.cfg.poll_seconds)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("ULTIMATE TRADING BOT v4.0")
    print("Technical + Sentiment + Adaptive Learning")
    print("=" * 60)

    required = ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"ERROR: Missing: {', '.join(missing)}")
        sys.exit(1)

    cfg = load_config()
    db = TradingDB(cfg.db_path)
    bot = UltimateBot(cfg, db)
    bot.run()


if __name__ == "__main__":
    main()
