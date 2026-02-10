# bot_ultimate.py — ULTIMATE TRADING BOT v3.0
# Technical Analysis + Sentiment Analysis + News Monitoring
# 100% FREE - No paid subscriptions required
#
# ============================================================================
# WEEKEND 1+2 UPDATES (Feb 8-9, 2026):
# ============================================================================
# MONITORING (logs only - no trading changes):
#   ✅ Crypto Risk Monitor (BTC/ETH via CoinGecko API)
#   ✅ Consecutive Losses Tracking
#   ✅ Daily Drawdown Calculation  
#   ✅ ATR (Average True Range) Volatility
#   ✅ Market Volatility Classification
#
# SAFE GUARDS (auto-pause protection):
#   ✅ Daily Loss Limit (-5% auto-pause until midnight)
#   ✅ Max Drawdown Circuit Breaker (-10% emergency stop)
#   ✅ Crypto Crash Protection (BTC < -10% = no buys)
#   ✅ Consecutive Loss Cooldown (3 losses = 2-day pause)
#
# REPORTING:
#   ✅ Google Sheets Dashboard (auto-sync every trade)
#   ✅ Enhanced logging with risk metrics
#
# New Railway Variables:
#   - MAX_DAILY_LOSS_PCT (default: 5.0)
#   - MAX_DRAWDOWN_PCT (default: 10.0)
#   - CRYPTO_CRASH_THRESHOLD (default: -10.0)
#   - CONSECUTIVE_LOSS_COOLDOWN_DAYS (default: 2)
#   - GOOGLE_SHEETS_CREDENTIALS (optional: service account JSON)
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

# NEW: Google Sheets integration (Weekend 1+2)
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False
    logger.warning("gspread not installed - Google Sheets disabled")

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
    """Setup proper logging"""
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    
    # Remove old handlers
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
    tp_bps: float
    sl_bps: float
    
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
    fractional_enabled: bool


def load_config() -> Config:
    """Load configuration from environment variables"""
    symbols = os.getenv("SYMBOLS", "NVDA,AMD,MSFT").replace(" ", "").split(",")
    return Config(
        # Trading
        symbols=symbols,
        max_pos_pct=float(os.getenv("MAX_POS_PCT", "0.10")),
        poll_seconds=int(os.getenv("POLL_SECONDS", "30")),
        
        # Technical
        lookback_bars=int(os.getenv("LOOKBACK_BARS", "300")),
        vwap_window=int(os.getenv("VWAP_WINDOW", "30")),
        rsi_len=int(os.getenv("RSI_LEN", "5")),
        rsi_buy=float(os.getenv("RSI_BUY", "30")),
        rsi_sell=float(os.getenv("RSI_SELL", "55")),
        entry_bps=float(os.getenv("ENTRY_BPS", "20")),
        tp_bps=float(os.getenv("TP_BPS", "40")),
        sl_bps=float(os.getenv("SL_BPS", "15")),
        
        # Risk management
        max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", "10.0")),
        max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", "2")),
        min_volume=int(os.getenv("MIN_VOLUME", "10000")),
        cooldown_sec=int(os.getenv("COOLDOWN_SEC", "60")),
        
        # Sentiment
        sentiment_enabled=os.getenv("SENTIMENT_ENABLED", "true").lower() == "true",
        sentiment_min_threshold=float(os.getenv("SENTIMENT_MIN_THRESHOLD", "-0.5")),
        sentiment_weight=float(os.getenv("SENTIMENT_WEIGHT", "0.3")),
        
        # Data
        db_path=os.getenv("DB_PATH", "bot_ultimate.db"),
        data_feed=os.getenv("DATA_FEED", "iex").lower(),
        timeframe_str=os.getenv("TIMEFRAME_STR", "1Min"),
        fractional_enabled=os.getenv("FRACTIONAL_ENABLED", "true").lower() == "true",
    )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def bps_to_mult(bps: float) -> float:
    """Convert basis points to multiplier - FIXED"""
    return 1.0 + (bps / 10000.0)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    d = close.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    rs = up.rolling(n).mean() / (down.rolling(n).mean() + 1e-9)
    return 100 - (100 / (1 + rs))


def rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate rolling VWAP"""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"].astype(float)
    v = df["volume"].astype(float)
    return pv.rolling(window).sum() / (v.rolling(window).sum() + 1e-12)


def is_market_hours() -> bool:
    """Check if US market is open - FIXED version from working bot"""
    import pytz
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    
    # Weekend check
    if now_ny.weekday() >= 5:
        return False
    
    # Market hours: 9:30 AM - 4:00 PM EST
    start_time = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return start_time <= now_ny <= end_time


def can_trade_now() -> bool:
    """
    Check if we can trade NOW (avoid PDT by skipping first 2 hours)
    This gives us time to exit same-day without PDT violation
    """
    if not is_market_hours():
        return False
    
    import pytz
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    
    # Allow trading only after 11:30 AM EST (2h after open)
    # This leaves 4.5h to exit same day (11:30 AM - 4:00 PM)
    trade_start = now_ny.replace(hour=11, minute=30, second=0, microsecond=0)
    
    return now_ny >= trade_start


# ============================================================================
# NEW: MONITORING & RISK MANAGEMENT FUNCTIONS (Weekend 1+2)
# ============================================================================

def get_crypto_sentiment() -> dict:
    """
    FREE Crypto Risk Monitor using CoinGecko API
    Returns BTC/ETH 24h changes and risk classification
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        btc_change = data['bitcoin']['usd_24h_change']
        eth_change = data['ethereum']['usd_24h_change']
        
        # Risk classification (from Warsaw bot)
        if btc_change < -10.0:
            risk_level = "CRASH"
        elif btc_change < -5.0:
            risk_level = "HIGH"
        elif btc_change < -2.0:
            risk_level = "MEDIUM"
        elif btc_change > 3.0:
            risk_level = "BULLISH"
        else:
            risk_level = "NORMAL"
        
        return {
            'btc_change_24h': btc_change,
            'eth_change_24h': eth_change,
            'risk_level': risk_level,
            'should_pause': btc_change < -10.0  # Crash protection
        }
    
    except Exception as e:
        logger.warning(f"Crypto API failed: {e}")
        return {
            'btc_change_24h': 0.0,
            'eth_change_24h': 0.0,
            'risk_level': 'UNKNOWN',
            'should_pause': False
        }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range (volatility measure)
    Used for adaptive stop-loss in future phases
    """
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


def calculate_market_volatility(df: pd.DataFrame) -> str:
    """
    Classify market volatility based on price standard deviation
    Returns: LOW, NORMAL, HIGH, EXTREME
    """
    if len(df) < 20:
        return "UNKNOWN"
    
    try:
        returns = df['close'].pct_change().tail(20)
        std_dev = returns.std() * 100  # Convert to percentage
        
        if std_dev < 1.0:
            return "LOW"
        elif std_dev < 2.0:
            return "NORMAL"
        elif std_dev < 3.5:
            return "HIGH"
        else:
            return "EXTREME"
    
    except Exception as e:
        logger.warning(f"Volatility calculation failed: {e}")
        return "UNKNOWN"


# ============================================================================
# SENTIMENT ANALYZER (100% FREE)
# ============================================================================
class SentimentAnalyzer:
    """
    Free sentiment analysis from RSS feeds and web scraping
    No paid subscriptions required!
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Free RSS feeds
        self.rss_feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # Tech news
        ]
        
        # Positive and negative keywords for sentiment
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
        
        # Cache for news (avoid re-fetching)
        self.news_cache: Dict[str, List[dict]] = {}
        self.cache_timestamp: Dict[str, float] = {}
        self.cache_duration = 300  # 5 minutes
        
        logger.info("SentimentAnalyzer initialized (100% FREE sources)")
    
    def fetch_rss_feeds(self) -> List[dict]:
        """Fetch news from free RSS feeds"""
        all_news = []
        
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:  # Last 20 articles
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
        """Scrape Google News (free, no API needed)"""
        news = []
        
        try:
            # Google News RSS for specific query
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
        """
        Get recent news for symbol from multiple free sources
        Uses cache to avoid excessive requests
        """
        cache_key = f"{symbol}_{max_age_hours}"
        
        # Check cache
        if cache_key in self.news_cache:
            age = time.time() - self.cache_timestamp[cache_key]
            if age < self.cache_duration:
                logger.debug(f"Using cached news for {symbol}")
                return self.news_cache[cache_key]
        
        logger.info(f"Fetching fresh news for {symbol}")
        
        all_news = []
        
        # 1. RSS feeds (general tech/finance news)
        rss_news = self.fetch_rss_feeds()
        
        # Filter for symbol mentions
        for news in rss_news:
            text = (news['title'] + ' ' + news.get('summary', '')).lower()
            if symbol.lower() in text or self._get_company_name(symbol).lower() in text:
                all_news.append(news)
        
        # 2. Google News RSS (symbol-specific)
        google_news = self.fetch_google_news(symbol)
        all_news.extend(google_news)
        
        # Cache results
        self.news_cache[cache_key] = all_news
        self.cache_timestamp[cache_key] = time.time()
        
        logger.info(f"Found {len(all_news)} news articles for {symbol}")
        return all_news
    
    def _get_company_name(self, symbol: str) -> str:
        """Map symbol to company name for better news matching"""
        mapping = {
            'NVDA': 'NVIDIA',
            'MSFT': 'Microsoft',
            'AMD': 'AMD',
            'AAPL': 'Apple',
            'GOOGL': 'Google',
            'TSLA': 'Tesla',
            'META': 'Meta',
            'AMZN': 'Amazon',
            'PLTR': 'Palantir',
        }
        return mapping.get(symbol, symbol)
    
    def analyze_sentiment(self, news_list: List[dict]) -> dict:
        """
        Analyze sentiment from news headlines
        Returns: {
            'sentiment': float (-1 to 1),
            'confidence': float (0 to 1),
            'positive_count': int,
            'negative_count': int,
            'neutral_count': int
        }
        """
        if not news_list:
            return {
                'sentiment': 0.0,
                'confidence': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'sample_headlines': []
            }
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        sentiment_scores = []
        sample_headlines = []
        
        for news in news_list[:15]:  # Analyze max 15 most recent
            title = news['title'].lower()
            summary = news.get('summary', '').lower()
            text = title + ' ' + summary
            
            # Count keyword matches
            pos_matches = sum(1 for kw in self.positive_keywords if kw in text)
            neg_matches = sum(1 for kw in self.negative_keywords if kw in text)
            
            # Calculate article sentiment
            if pos_matches > neg_matches:
                article_sentiment = min(1.0, pos_matches * 0.2)
                positive_count += 1
                sentiment_scores.append(article_sentiment)
                sample_headlines.append(('POSITIVE', news['title']))
            
            elif neg_matches > pos_matches:
                article_sentiment = -min(1.0, neg_matches * 0.2)
                negative_count += 1
                sentiment_scores.append(article_sentiment)
                sample_headlines.append(('NEGATIVE', news['title']))
            
            else:
                neutral_count += 1
                sentiment_scores.append(0.0)
        
        # Overall sentiment (average)
        overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        
        # Confidence based on volume and agreement
        total = positive_count + negative_count + neutral_count
        if total > 0:
            agreement = max(positive_count, negative_count) / total
            volume_factor = min(1.0, total / 10.0)  # More news = higher confidence
            confidence = agreement * volume_factor
        else:
            confidence = 0.0
        
        return {
            'sentiment': float(overall_sentiment),
            'confidence': float(confidence),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_articles': total,
            'sample_headlines': sample_headlines[:5]  # Top 5
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
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", 
            (name,)
        )
        return cur.fetchone() is not None

    def _cols(self, table: str) -> List[str]:
        cur = self.conn.execute(f"PRAGMA table_info({table})")
        return [r[1] for r in cur.fetchall()]

    def _ensure_cols(self, table: str, cols_sql: Dict[str, str]):
        if not self._table_exists(table):
            return
        existing = set(self._cols(table))
        for col, coldef in cols_sql.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coldef};")
        self.conn.commit()

    def init_schema(self):
        """Initialize database schema"""
        # Bars table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS bars (
            symbol TEXT NOT NULL,
            ts_utc TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY(symbol, ts_utc, timeframe)
        );
        """)
        
        # Signals table (with sentiment columns)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT NOT NULL
        );
        """)
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            ts_utc TEXT,
            symbol TEXT,
            side TEXT,
            qty REAL,
            status TEXT,
            raw_json TEXT
        );
        """)
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            symbol TEXT NOT NULL,
            qty REAL NOT NULL
        );
        """)
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS eod_reports (
            date TEXT PRIMARY KEY,
            start_equity REAL,
            end_equity REAL,
            pnl_realized REAL,
            pnl_unrealized REAL,
            total_trades INTEGER,
            winning_trades INTEGER,
            fees_estimated REAL,
            report_json TEXT
        );
        """)
        
        # News cache table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS news_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            symbol TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT,
            sentiment REAL,
            link TEXT
        );
        """)
        
        self.conn.commit()
        
        # Ensure all columns exist (auto-migration)
        self._ensure_cols("signals", {
            "price": "REAL",
            "vwap": "REAL",
            "rsi": "REAL",
            "equity": "REAL",
            "cash": "REAL",
            "position_qty": "REAL",
            "position_value": "REAL",
            "max_pos_value": "REAL",
            "entry_price": "REAL",
            "tp_price": "REAL",
            "sl_price": "REAL",
            "sentiment": "REAL",
            "sentiment_confidence": "REAL",
            "news_count": "INTEGER",
            "raw_json": "TEXT",
        })
        
        self._ensure_cols("positions", {
            "market_value": "REAL",
            "avg_entry_price": "REAL",
            "unrealized_pl": "REAL",
            "raw_json": "TEXT",
        })
        
        self.conn.commit()
        logger.info("Database schema initialized")

    def upsert_bar(self, symbol: str, ts: datetime, timeframe: str, o, h, l, c, v):
        try:
            self.conn.execute(
                """INSERT OR REPLACE INTO bars
                   (symbol, ts_utc, timeframe, open, high, low, close, volume) 
                   VALUES (?,?,?,?,?,?,?,?)""",
                (symbol, iso(ts), timeframe, float(o), float(h), float(l), float(c), float(v)),
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert bar for {symbol}: {e}")

    def insert_signal(self, ts: datetime, symbol: str, action: str, reason: str, **kw):
        self.conn.execute(
            """INSERT INTO signals(
              ts_utc, symbol, action, reason, price, vwap, rsi, equity, cash,
              position_qty, position_value, max_pos_value, entry_price, tp_price, sl_price,
              sentiment, sentiment_confidence, news_count, raw_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                iso(ts), symbol, action, reason,
                kw.get("price"), kw.get("vwap"), kw.get("rsi"),
                kw.get("equity"), kw.get("cash"),
                kw.get("position_qty"), kw.get("position_value"), kw.get("max_pos_value"),
                kw.get("entry_price"), kw.get("tp_price"), kw.get("sl_price"),
                kw.get("sentiment"), kw.get("sentiment_confidence"), kw.get("news_count"),
                json.dumps(kw.get("raw", {}), ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def upsert_order(self, order_obj):
        d = order_obj.model_dump(mode='json') if hasattr(order_obj, "model_dump") else dict(order_obj)
        self.conn.execute(
            """INSERT OR REPLACE INTO orders
               (id, ts_utc, symbol, side, qty, status, raw_json) 
               VALUES (?,?,?,?,?,?,?)""",
            (d.get("id"), iso(utc_now()), d.get("symbol"), str(d.get("side")),
             float(d.get("qty")) if d.get("qty") is not None else None,
             str(d.get("status")), json.dumps(d, ensure_ascii=False)),
        )
        self.conn.commit()

    def insert_position(self, ts: datetime, pos_obj):
        d = pos_obj.model_dump() if hasattr(pos_obj, "model_dump") else dict(pos_obj)
        self.conn.execute(
            """INSERT INTO positions
               (ts_utc, symbol, qty, market_value, avg_entry_price, unrealized_pl, raw_json)
               VALUES (?,?,?,?,?,?,?)""",
            (iso(ts), d.get("symbol"), float(d.get("qty")),
             float(d.get("market_value")) if d.get("market_value") is not None else None,
             float(d.get("avg_entry_price")) if d.get("avg_entry_price") is not None else None,
             float(d.get("unrealized_pl")) if d.get("unrealized_pl") is not None else None,
             json.dumps(d, ensure_ascii=False, default=str))
        )
        self.conn.commit()
    
    def save_news(self, symbol: str, title: str, source: str, sentiment: float, link: str = ""):
        """Save news article to database"""
        self.conn.execute(
            """INSERT INTO news_cache(ts_utc, symbol, title, source, sentiment, link)
               VALUES (?,?,?,?,?,?)""",
            (iso(utc_now()), symbol, title, source, sentiment, link)
        )
        self.conn.commit()

    def save_eod_report(self, date: str, report: dict):
        self.conn.execute(
            """INSERT OR REPLACE INTO eod_reports
               (date, start_equity, end_equity, pnl_realized, pnl_unrealized, 
                total_trades, winning_trades, fees_estimated, report_json)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (date, report.get("start_equity"), report.get("end_equity"),
             report.get("pnl_realized"), report.get("pnl_unrealized"),
             report.get("total_trades"), report.get("winning_trades"),
             report.get("fees_estimated"), json.dumps(report, ensure_ascii=False))
        )
        self.conn.commit()

    def get_entry_price(self, symbol: str) -> Optional[float]:
        cur = self.conn.execute(
            """SELECT entry_price FROM signals 
               WHERE symbol=? AND action='BUY' AND entry_price IS NOT NULL
               ORDER BY ts_utc DESC LIMIT 1""",
            (symbol,)
        )
        row = cur.fetchone()
        return float(row[0]) if row else None

    def get_start_equity(self, date: str) -> Optional[float]:
        cur = self.conn.execute(
            """SELECT end_equity FROM eod_reports 
               WHERE date < ? ORDER BY date DESC LIMIT 1""",
            (date,)
        )
        row = cur.fetchone()
        return float(row[0]) if row else None
    
    def get_trades_today(self, symbol: str, date: str) -> int:
        """Count trades for symbol today"""
        cur = self.conn.execute(
            """SELECT COUNT(*) FROM signals 
               WHERE symbol=? AND action='BUY' AND DATE(ts_utc)=?""",
            (symbol, date)
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


# ============================================================================
# GOOGLE SHEETS REPORTER (Weekend 1+2)
# ============================================================================
class GoogleSheetsReporter:
    """
    Syncs trading data to Google Sheets dashboard
    Requires: gspread, oauth2client
    Setup: Create service account JSON in Railway secrets
    """
    
    def __init__(self, sheet_name: str = "Trading Bot Dashboard"):
        self.enabled = GSHEETS_AVAILABLE
        self.sheet_name = sheet_name
        self.client = None
        self.sheet = None
        
        if not self.enabled:
            logger.info("Google Sheets: DISABLED (missing libraries)")
            return
        
        # Try to authenticate
        try:
            # Check for service account JSON in environment
            creds_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
            
            if not creds_json:
                logger.warning("Google Sheets: No credentials found (set GOOGLE_SHEETS_CREDENTIALS)")
                self.enabled = False
                return
            
            # Parse JSON credentials
            import json
            creds_dict = json.loads(creds_json)
            
            # Authenticate
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            self.client = gspread.authorize(creds)
            
            # Open or create sheet
            try:
                self.sheet = self.client.open(sheet_name).sheet1
                logger.info(f"Google Sheets: Connected to '{sheet_name}'")
            except gspread.exceptions.SpreadsheetNotFound:
                # Create new sheet
                spreadsheet = self.client.create(sheet_name)
                spreadsheet.share('', perm_type='anyone', role='reader')  # Public read
                self.sheet = spreadsheet.sheet1
                self._init_headers()
                logger.info(f"Google Sheets: Created new '{sheet_name}'")
        
        except Exception as e:
            logger.error(f"Google Sheets: Auth failed | {e}")
            self.enabled = False
    
    def _init_headers(self):
        """Initialize spreadsheet headers"""
        if not self.enabled or not self.sheet:
            return
        
        try:
            headers = [
                'Date', 'Time (UTC)', 'Equity', 'Daily P/L', 'Daily P/L %',
                'Positions', 'Cash', 'Win Rate %', 'Total Trades',
                'Drawdown %', 'BTC 24h %', 'Risk Level'
            ]
            self.sheet.append_row(headers)
            logger.info("Google Sheets: Headers initialized")
        
        except Exception as e:
            logger.error(f"Google Sheets: Header init failed | {e}")
    
    def update_daily_stats(self, data: dict):
        """
        Update daily stats row
        data = {
            'equity': float,
            'daily_start': float,
            'positions': int,
            'cash': float,
            'total_trades': int,
            'winning_trades': int,
            'drawdown_pct': float,
            'crypto_data': dict
        }
        """
        if not self.enabled or not self.sheet:
            return
        
        try:
            now = utc_now()
            daily_pnl = data['equity'] - data['daily_start']
            daily_pnl_pct = (daily_pnl / data['daily_start']) * 100 if data['daily_start'] > 0 else 0.0
            win_rate = (data['winning_trades'] / data['total_trades'] * 100) if data['total_trades'] > 0 else 0.0
            
            row = [
                now.strftime('%Y-%m-%d'),
                now.strftime('%H:%M:%S'),
                f"{data['equity']:.2f}",
                f"{daily_pnl:+.2f}",
                f"{daily_pnl_pct:+.2f}",
                data['positions'],
                f"{data['cash']:.2f}",
                f"{win_rate:.1f}",
                data['total_trades'],
                f"{data['drawdown_pct']:.2f}",
                f"{data['crypto_data'].get('btc_change_24h', 0):+.2f}",
                data['crypto_data'].get('risk_level', 'UNKNOWN')
            ]
            
            self.sheet.append_row(row)
            logger.info(f"Google Sheets: Updated | Equity: ${data['equity']:.2f} | P/L: {daily_pnl_pct:+.2f}%")
        
        except Exception as e:
            logger.error(f"Google Sheets: Update failed | {e}")


# ============================================================================
# BOT
# ============================================================================
class UltimateBot:
    def __init__(self, cfg: Config, db: TradingDB):
        self.cfg = cfg
        self.db = db
        
        # API clients
        key = os.environ["APCA_API_KEY_ID"]
        secret = os.environ["APCA_API_SECRET_KEY"]
        self.trading = TradingClient(key, secret, paper=True)
        self.data = StockHistoricalDataClient(key, secret)
        
        # Sentiment analyzer (if enabled)
        self.sentiment = SentimentAnalyzer() if cfg.sentiment_enabled else None
        
        # NEW: Google Sheets reporter (Weekend 1+2)
        self.sheets = GoogleSheetsReporter()
        
        # State tracking
        self.last_trade: Dict[str, float] = {s: 0.0 for s in cfg.symbols}
        self.entry_price: Dict[str, float] = {}
        self.start_equity: Optional[float] = None
        self.peak_equity: Optional[float] = None
        
        # NEW: Weekend 1+2 - Risk tracking
        self.daily_peak_equity: Optional[float] = None
        self.daily_start_equity: Optional[float] = None
        self.symbol_cooldowns: Dict[str, datetime] = {}  # Track cooldown end times
        self.paused_until: Optional[datetime] = None  # Daily loss limit pause
        
        # Load entry prices from DB
        self._load_entry_prices()
        
        logger.info(f"UltimateBot initialized | sentiment={'ENABLED' if cfg.sentiment_enabled else 'DISABLED'}")
        logger.info(f"Symbols: {cfg.symbols} | max_pos: {cfg.max_pos_pct*100}%")

    def _load_entry_prices(self):
        for sym in self.cfg.symbols:
            entry = self.db.get_entry_price(sym)
            if entry:
                self.entry_price[sym] = entry
                logger.info(f"Loaded entry price for {sym}: ${entry:.2f}")
    
    # NEW: Weekend 1+2 Methods
    def count_consecutive_losses(self, symbol: str) -> int:
        """
        Count consecutive losing trades for a symbol
        Used for cooldown protection
        """
        try:
            cur = self.db.conn.execute(
                """SELECT entry_price, price 
                   FROM signals 
                   WHERE symbol=? AND action='SELL' AND entry_price IS NOT NULL
                   ORDER BY ts_utc DESC LIMIT 10""",
                (symbol,)
            )
            
            consecutive = 0
            for row in cur:
                entry, exit_price = row
                if exit_price and entry and exit_price < entry:
                    consecutive += 1
                else:
                    break
            
            return consecutive
        
        except Exception as e:
            logger.warning(f"Count consecutive losses failed: {e}")
            return 0
    
    def calculate_daily_drawdown(self, current_equity: float) -> float:
        """
        Calculate drawdown from today's peak
        Returns percentage drawdown
        """
        if self.daily_peak_equity is None:
            self.daily_peak_equity = current_equity
        
        self.daily_peak_equity = max(self.daily_peak_equity, current_equity)
        
        if self.daily_peak_equity > 0:
            dd = ((self.daily_peak_equity - current_equity) / self.daily_peak_equity) * 100
            return dd
        
        return 0.0
    
    def check_symbol_cooldown(self, symbol: str) -> bool:
        """
        Check if symbol is in cooldown period after consecutive losses
        Returns True if can trade, False if in cooldown
        """
        # Check if symbol has cooldown
        if symbol in self.symbol_cooldowns:
            cooldown_end = self.symbol_cooldowns[symbol]
            if utc_now() < cooldown_end:
                remaining = (cooldown_end - utc_now()).total_seconds() / 3600
                logger.info(f"{symbol} in cooldown for {remaining:.1f}h more")
                return False
            else:
                # Cooldown expired, remove it
                del self.symbol_cooldowns[symbol]
        
        # Check consecutive losses
        consecutive_losses = self.count_consecutive_losses(symbol)
        
        # Get cooldown days from config (default 2)
        cooldown_days = int(os.getenv("CONSECUTIVE_LOSS_COOLDOWN_DAYS", "2"))
        
        if consecutive_losses >= 3:
            # Set cooldown
            cooldown_end = utc_now() + timedelta(days=cooldown_days)
            self.symbol_cooldowns[symbol] = cooldown_end
            logger.warning(f"{symbol}: {consecutive_losses} consecutive losses - COOLDOWN {cooldown_days} days")
            return False
        
        return True
    
    def check_daily_loss_limit(self, current_equity: float) -> bool:
        """
        Check if daily loss limit exceeded
        Returns True if can trade, False if paused
        """
        # Check if already paused
        if self.paused_until:
            if utc_now() < self.paused_until:
                remaining = (self.paused_until - utc_now()).total_seconds() / 3600
                logger.warning(f"PAUSED due to daily loss limit for {remaining:.1f}h more")
                return False
            else:
                # Pause expired
                self.paused_until = None
        
        # Calculate daily P/L
        if self.daily_start_equity is None:
            self.daily_start_equity = current_equity
        
        daily_pnl_pct = ((current_equity - self.daily_start_equity) / self.daily_start_equity) * 100
        
        # Get limit from config (default -5%)
        max_daily_loss = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))
        
        if daily_pnl_pct < -max_daily_loss:
            # Set pause until midnight EST
            import pytz
            tz_ny = pytz.timezone('America/New_York')
            now_ny = datetime.now(tz_ny)
            midnight = (now_ny + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            self.paused_until = midnight.astimezone(timezone.utc)
            
            logger.critical(f"DAILY LOSS LIMIT HIT: {daily_pnl_pct:.2f}% - PAUSED until midnight")
            return False
        
        return True

    def fetch_bars(self, sym: str) -> pd.DataFrame:
        try:
            logger.info(f">>> FETCHING BARS for {sym} with 5Min timeframe")
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=utc_now() - timedelta(minutes=self.cfg.lookback_bars * 5),
                end=utc_now(),
                feed=self.cfg.data_feed,
            )
            logger.info(f">>> Request: start={req.start}, end={req.end}, feed={req.feed}")
            
            df = self.data.get_stock_bars(req).df
            
            logger.info(f">>> Response: df is None={df is None}, len={len(df) if df is not None else 0}")
            
            if df is None or len(df) == 0:
                logger.error(f">>> NO BARS RETURNED for {sym}!")
                return pd.DataFrame()
            
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                df = df[df["symbol"] == sym].set_index("timestamp")
            
            logger.info(f">>> SUCCESS: Retrieved {len(df)} bars for {sym}")
            return df.sort_index()
        
        except Exception as e:
            logger.error(f">>> FETCH_BARS ERROR for {sym}: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f">>> TRACEBACK: {traceback.format_exc()}")
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Failed to fetch bars for {sym}: {e}")
            return pd.DataFrame()

    def validate_bars(self, df: pd.DataFrame, sym: str) -> bool:
        if df.empty:
            return False
        
        required_bars = max(self.cfg.vwap_window + 5, self.cfg.rsi_len + 5)
        if len(df) < required_bars:
            return False
        
        last_price = df["close"].iloc[-1]
        if pd.isna(last_price) or last_price <= 0:
            return False
        
        total_volume = df["volume"].tail(10).sum()
        if total_volume < self.cfg.min_volume:
            logger.debug(f"{sym}: Low volume ({total_volume})")
            return False
        
        return True

    def max_qty(self, equity: float, price: float, cash: float) -> float:
        if price <= 0:
            return 0.0
        
        max_investment = min(
            equity * self.cfg.max_pos_pct,
            cash * 0.95
        )
        
        qty = max_investment / price
        
        if not self.cfg.fractional_enabled:
            qty = int(qty)
        
        return max(0.0, qty)

    def submit_mkt(self, sym: str, side: OrderSide, qty: float) -> Optional[object]:
        if qty <= 0:
            return None
        
        if self.cfg.fractional_enabled and qty < 0.01:
            return None
        
        try:
            if self.cfg.fractional_enabled:
                qty = round(qty, 4)
            else:
                qty = int(qty)
            
            order = self.trading.submit_order(
                MarketOrderRequest(
                    symbol=sym,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            )
            
            self.db.upsert_order(order)
            self.last_trade[sym] = time.time()
            
            logger.info(f"{sym}: {side} order submitted | qty={qty}")
            return order
        
        except Exception as e:
            logger.error(f"{sym}: Order failed | {e}")
            return None

    def check_drawdown(self, current_equity: float) -> bool:
        if self.peak_equity is None:
            self.peak_equity = current_equity
            return False
        
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        drawdown_pct = ((self.peak_equity - current_equity) / self.peak_equity) * 100
        
        if drawdown_pct >= self.cfg.max_drawdown_pct:
            logger.critical(
                f"MAX DRAWDOWN | Peak: ${self.peak_equity:.2f} | "
                f"Current: ${current_equity:.2f} | DD: {drawdown_pct:.2f}%"
            )
            return True
        
        return False
    
    def can_trade_today(self, symbol: str) -> bool:
        """Check if can still trade this symbol today"""
        today = utc_now().strftime("%Y-%m-%d")
        trades_count = self.db.get_trades_today(symbol, today)
        return trades_count < self.cfg.max_trades_per_day

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
        
        for sym in self.cfg.symbols:
            try:
                self._process_symbol(sym, equity, cash, positions)
            except Exception as e:
                logger.error(f"{sym}: Processing error | {e}")
                traceback.print_exc()
        
        # Snapshot positions
        ts = utc_now()
        for p in self.trading.get_all_positions():
            self.db.insert_position(ts, p)
        
        # NEW: Update Google Sheets (Weekend 1+2) - every run
        try:
            if self.sheets.enabled and self.daily_start_equity:
                crypto_data = get_crypto_sentiment()
                daily_dd = self.calculate_daily_drawdown(equity)
                
                # Count trades and wins from DB
                cur = self.db.conn.execute(
                    "SELECT COUNT(*), SUM(CASE WHEN price > entry_price THEN 1 ELSE 0 END) FROM signals WHERE action='SELL' AND DATE(ts_utc)=?",
                    (utc_now().strftime('%Y-%m-%d'),)
                )
                row = cur.fetchone()
                total_trades = int(row[0]) if row and row[0] else 0
                winning_trades = int(row[1]) if row and row[1] else 0
                
                self.sheets.update_daily_stats({
                    'equity': equity,
                    'daily_start': self.daily_start_equity,
                    'positions': len(positions),
                    'cash': cash,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'drawdown_pct': daily_dd,
                    'crypto_data': crypto_data
                })
        except Exception as e:
            logger.warning(f"Google Sheets update failed: {e}")

    def _process_symbol(self, sym: str, equity: float, cash: float, positions: Dict):
        # Fetch and validate bars
        df = self.fetch_bars(sym)
        if not self.validate_bars(df, sym):
            return
        
        # Store latest bars
        for ts, row in df.tail(3).iterrows():
            ts_dt = ts.to_pydatetime()
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            self.db.upsert_bar(
                sym, ts_dt, self.cfg.timeframe_str,
                row["open"], row["high"], row["low"], row["close"], row["volume"]
            )
        
        # Technical indicators
        logger.info(f">>> Processing {sym}: df has {len(df)} bars, vwap_window={self.cfg.vwap_window}, rsi_len={self.cfg.rsi_len}")
        price = float(df["close"].iloc[-1])
        vwap_val = float(rolling_vwap(df, self.cfg.vwap_window).iloc[-1])
        rsi_val = float(rsi(df["close"].astype(float), self.cfg.rsi_len).iloc[-1])
        logger.info(f"{sym} | Price: ${price:.2f} | VWAP: ${vwap_val:.2f} | RSI: {rsi_val:.1f}")
        
        # === NEW: MONITORING METRICS (Weekend 1) ===
        try:
            # 1. Crypto risk monitor
            crypto_data = get_crypto_sentiment()
            logger.info(f"CRYPTO | BTC: {crypto_data['btc_change_24h']:+.2f}% | ETH: {crypto_data['eth_change_24h']:+.2f}% | Risk: {crypto_data['risk_level']}")
            
            # 2. ATR (volatility)
            atr_value = calculate_atr(df, 14)
            logger.info(f"{sym} | ATR(14): ${atr_value:.2f}")
            
            # 3. Market volatility classification
            volatility = calculate_market_volatility(df)
            logger.info(f"{sym} | Market Volatility: {volatility}")
            
            # 4. Consecutive losses tracking
            consecutive_losses = self.count_consecutive_losses(sym)
            if consecutive_losses > 0:
                logger.info(f"{sym} | Consecutive losses: {consecutive_losses}")
            
            # 5. Daily drawdown
            daily_dd = self.calculate_daily_drawdown(equity)
            if daily_dd > 0.5:
                logger.info(f"Daily Drawdown: {daily_dd:.2f}% | Peak: ${self.daily_peak_equity:.2f} | Current: ${equity:.2f}")
        
        except Exception as e:
            logger.warning(f"Monitoring metrics failed: {e}")
        
        # Position info
        pos = positions.get(sym)
        pos_qty = float(pos.qty) if pos else 0.0
        pos_value = float(pos.market_value) if (pos and hasattr(pos, "market_value")) else 0.0
        
        # Entry price - ALWAYS use Alpaca position data (survives restarts!)
        entry = None
        if pos and hasattr(pos, "avg_entry_price") and pos.avg_entry_price:
            try:
                entry = float(pos.avg_entry_price)
                # Update local cache
                self.entry_price[sym] = entry
            except Exception as e:
                logger.warning(f"{sym}: Failed to get entry from position: {e}")
        
        # Fallback to cache only if no position
        if entry is None and pos_qty == 0:
            entry = self.entry_price.get(sym)
        
        
        # TP/SL levels
        tp_price = entry * bps_to_mult(self.cfg.tp_bps) if entry else None
        sl_price = entry * bps_to_mult(-self.cfg.sl_bps) if entry else None
        
        # === SENTIMENT ANALYSIS ===
        sentiment_score = 0.0
        sentiment_confidence = 0.0
        news_count = 0
        
        if self.cfg.sentiment_enabled and self.sentiment:
            try:
                news_list = self.sentiment.get_recent_news(sym, max_age_hours=4)
                if news_list:
                    sentiment_data = self.sentiment.analyze_sentiment(news_list)
                    sentiment_score = sentiment_data['sentiment']
                    sentiment_confidence = sentiment_data['confidence']
                    news_count = sentiment_data['total_articles']
                    
                    # Log sentiment details
                    if sentiment_data['sample_headlines']:
                        logger.info(f"{sym} | Sentiment: {sentiment_score:+.2f} | Confidence: {sentiment_confidence:.2f}")
                        for sent_type, headline in sentiment_data['sample_headlines'][:2]:
                            logger.info(f"  [{sent_type}] {headline[:80]}")
                    
                    # Save news to DB
                    for headline_data in sentiment_data['sample_headlines']:
                        self.db.save_news(sym, headline_data[1], 'RSS', sentiment_score, "")
            
            except Exception as e:
                logger.warning(f"{sym}: Sentiment analysis failed | {e}")
        
        # === TRADING SIGNALS ===
        
        # Technical entry trigger
        technical_entry = (
            (price <= vwap_val * bps_to_mult(-self.cfg.entry_bps)) and 
            (rsi_val <= self.cfg.rsi_buy)
        )
        
        # Sentiment filter (if enabled)
        sentiment_ok = True
        if self.cfg.sentiment_enabled and news_count > 0:
            # Don't buy if very negative sentiment
            if sentiment_score < self.cfg.sentiment_min_threshold:
                sentiment_ok = False
                logger.info(f"{sym} | BLOCKED by negative sentiment ({sentiment_score:.2f})")
        
        # Combined entry trigger
        entry_trigger = technical_entry and sentiment_ok
        
        # Exit triggers
        exit_trigger = (
            (price >= vwap_val) or 
            (rsi_val >= self.cfg.rsi_sell)
        )
        
        tp_hit = (tp_price is not None and price >= tp_price)
        sl_hit = (sl_price is not None and price <= sl_price)

        # Check cooldown and trade limits
        cooled = (time.time() - self.last_trade[sym]) >= self.cfg.cooldown_sec
        can_trade = self.can_trade_today(sym)
        
        # === NEW: SAFE GUARDS (Weekend 2) ===
        safe_guards_passed = True
        
        # 1. Daily loss limit check
        if not self.check_daily_loss_limit(equity):
            safe_guards_passed = False
            logger.warning(f"SAFE GUARD: Daily loss limit - trading paused")
        
        # 2. Symbol cooldown check (consecutive losses)
        if not self.check_symbol_cooldown(sym):
            safe_guards_passed = False
            logger.warning(f"SAFE GUARD: {sym} in cooldown after consecutive losses")
        
        # 3. Crypto crash protection
        crypto_data = get_crypto_sentiment()
        if crypto_data['should_pause']:
            safe_guards_passed = False
            logger.warning(f"SAFE GUARD: Crypto crash detected (BTC {crypto_data['btc_change_24h']:.2f}%) - no new buys")
        
        # 4. Max drawdown circuit breaker
        max_dd_threshold = float(os.getenv("MAX_DRAWDOWN_PCT", "10.0"))
        if self.peak_equity and equity < self.peak_equity:
            total_dd = ((self.peak_equity - equity) / self.peak_equity) * 100
            if total_dd > max_dd_threshold:
                logger.critical(f"CIRCUIT BREAKER: Max drawdown {total_dd:.2f}% exceeded threshold {max_dd_threshold}%")
                raise KeyboardInterrupt("Circuit breaker triggered - max drawdown exceeded")
        
        action = "HOLD"
        reason = "none"
        
        # === EXECUTE TRADES ===
        
        if pos_qty == 0:
            # No position - look for entry
            if entry_trigger and cooled and can_trade and safe_guards_passed:
                qty = self.max_qty(equity, price, cash)
                
                if qty > 0:
                    # Lock symbol immediately (prevent double-buy)
                    self.last_trade[sym] = time.time()
                    action = "BUY"
                    reason = "technical_entry"
                    if self.cfg.sentiment_enabled and sentiment_score > 0.3:
                        reason = "technical_entry+positive_sentiment"
                    
                    logger.info(
                        f"{sym} | BUY {qty:.4f} @${price:.2f} | "
                        f"VWAP=${vwap_val:.2f} RSI={rsi_val:.1f} | "
                        f"Sentiment={sentiment_score:+.2f} ({news_count} news)"
                    )
                    
                    order = self.submit_mkt(sym, OrderSide.BUY, qty)
                    
                    if order:
                        self.entry_price[sym] = price
                        entry = price
                        tp_price = entry * bps_to_mult(self.cfg.tp_bps)
                        sl_price = entry * bps_to_mult(-self.cfg.sl_bps)
                        cash -= qty * price
            
            elif entry_trigger and not can_trade:
                logger.info(f"{sym} | Max trades/day reached ({self.cfg.max_trades_per_day})")
        
        else:
            logger.info(f"{sym} | Position check: pos_qty={pos_qty:.4f}, entry=${entry if entry else 0:.2f}, tp=${tp_price if tp_price else 0:.2f}, sl=${sl_price if sl_price else 0:.2f}")
            logger.info(f"{sym} | Exit conditions: exit_trigger={exit_trigger}, tp_hit={tp_hit}, sl_hit={sl_hit}, cooled={cooled}, price=${price:.2f}")
            # Have position - look for exit
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
                
                logger.info(
                    f"{sym} | SELL {qty:.4f} @${price:.2f} | "
                    f"{reason} | P/L: {pnl_pct:+.2f}% | "
                    f"Entry: ${entry:.2f} TP: ${tp_price:.2f} SL: ${sl_price:.2f}"
                )
                
                order = self.submit_mkt(sym, OrderSide.SELL, qty)
                
                if order:
                    self.entry_price.pop(sym, None)
        
        # Log signal ONLY if action happened (not just HOLD)
        if action != "HOLD":
            max_pos_value = equity * self.cfg.max_pos_pct
            self.db.insert_signal(
                ts=utc_now(), symbol=sym, action=action, reason=reason,
                price=price, vwap=vwap_val, rsi=rsi_val,
                equity=equity, cash=cash,
                position_qty=pos_qty, position_value=pos_value, max_pos_value=max_pos_value,
                entry_price=entry, tp_price=tp_price, sl_price=sl_price,
                sentiment=sentiment_score, sentiment_confidence=sentiment_confidence, 
                news_count=news_count,
                raw={
                    "technical_entry": technical_entry,
                    "sentiment_ok": sentiment_ok,
                    "exit_trigger": exit_trigger,
                    "tp_hit": tp_hit,
                    "sl_hit": sl_hit,
                },
            )

    def generate_eod_report(self) -> dict:
        today = utc_now().strftime("%Y-%m-%d")
        
        try:
            acct = self.trading.get_account()
            end_equity = float(acct.equity)
        except:
            end_equity = 0.0
        
        start_equity = self.start_equity or self.db.get_start_equity(today) or end_equity
        
        conn = self.db.conn
        cur = conn.execute(
            """SELECT action, entry_price, price 
               FROM signals 
               WHERE DATE(ts_utc) = ? AND action IN ('BUY', 'SELL')""",
            (today,)
        )
        trades = cur.fetchall()
        
        total_trades = len([t for t in trades if t[0] == 'SELL'])
        winning_trades = 0
        
        for action, entry, exit_price in trades:
            if action == 'SELL' and entry and exit_price:
                if exit_price > entry:
                    winning_trades += 1
        
        pnl_realized = end_equity - start_equity
        fees_estimated = total_trades * 2 * 0.005
        
        report = {
            "date": today,
            "start_equity": start_equity,
            "end_equity": end_equity,
            "pnl_realized": pnl_realized,
            "pnl_pct": (pnl_realized / start_equity * 100) if start_equity > 0 else 0,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            "fees_estimated": fees_estimated,
        }
        
        self.db.save_eod_report(today, report)
        
        logger.info(f"""
{'='*60}
EOD REPORT {today}
{'='*60}
Start Equity: ${start_equity:,.2f}
End Equity:   ${end_equity:,.2f}
P/L:          ${pnl_realized:+,.2f} ({report['pnl_pct']:+.2f}%)
Trades:       {total_trades} (Win: {winning_trades}, Rate: {report['win_rate']:.1f}%)
Est. Fees:    ${fees_estimated:.2f}
{'='*60}
        """)
        
        return report

    def run(self):
        logger.info("="*60)
        logger.info("ULTIMATE BOT START")
        logger.info("="*60)
        logger.info(f"Symbols: {self.cfg.symbols}")
        logger.info(f"Sentiment Analysis: {'ENABLED' if self.cfg.sentiment_enabled else 'DISABLED'}")
        logger.info(f"Max Trades/Day: {self.cfg.max_trades_per_day}")
        logger.info(f"Position Size: {self.cfg.max_pos_pct*100}%")
        logger.info("="*60)
        
        try:
            acct = self.trading.get_account()
            logger.info(f"Account: equity=${acct.equity} cash=${acct.cash}")
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
        
        last_eod_date = None
        daily_equity_set = False  # NEW: Track if we set today's start
        
        while True:
            try:
                if not can_trade_now():
                    if not is_market_hours():
                        current_date = utc_now().strftime("%Y-%m-%d")
                        
                        if current_date != last_eod_date:
                            self.generate_eod_report()
                            last_eod_date = current_date
                            # NEW: Reset daily trackers at market close
                            daily_equity_set = False
                        
                        logger.info("Market closed, sleeping...")
                    else:
                        logger.info("Pre-market hours (9:30-11:30 AM) - waiting to trade...")
                    
                    time.sleep(300)
                    continue
                
                # NEW: Set daily start equity once per day at first run
                if not daily_equity_set:
                    try:
                        acct = self.trading.get_account()
                        self.daily_start_equity = float(acct.equity)
                        self.daily_peak_equity = self.daily_start_equity
                        self.peak_equity = max(self.peak_equity or 0, self.daily_start_equity)
                        daily_equity_set = True
                        logger.info(f"NEW: Daily start equity set: ${self.daily_start_equity:.2f}")
                    except Exception as e:
                        logger.error(f"Failed to set daily equity: {e}")
                
                self.run_once()
                
            except KeyboardInterrupt:
                logger.info("CTRL+C received, stopping...")
                self.generate_eod_report()
                break
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                traceback.print_exc()
            
            time.sleep(self.cfg.poll_seconds)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*60)
    print("ULTIMATE TRADING BOT v2.0")
    print("Technical + Sentiment Analysis")
    print("="*60)
    print()
    
    # Check for required environment variables
    required_vars = ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print()
        print("Please set them in .env file or as environment variables:")
        print("  APCA_API_KEY_ID=your_key")
        print("  APCA_API_SECRET_KEY=your_secret")
        sys.exit(1)
    
    cfg = load_config()
    db = TradingDB(cfg.db_path)
    bot = UltimateBot(cfg, db)
    
    bot.run()


if __name__ == "__main__":
    main()
