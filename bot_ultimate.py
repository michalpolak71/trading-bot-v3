# bot_ultimate.py — ULTIMATE TRADING BOT v5.0 AGGRESSIVE
# Strategy: Yesterday's Low + Rebound (like SAPER bot) + Sentiment Filter
# 100% AGGRESSIVE - No safe guards, maximum profit potential
#
# ============================================================================
# v5.0 CHANGES (Feb 25, 2026):
# ============================================================================
# STRATEGY OVERHAUL:
#   ✅ Replaced mean-reversion with "dip buying" strategy
#   ✅ Entry: Yesterday's low + 0.2% tolerance + rebound confirmation
#   ✅ All-in position sizing (95% capital per trade)
#   ✅ Sentiment filter (only addition vs SAPER bot)
#   ✅ ATR-based dynamic TP/SL (better than fixed %)
#   ❌ REMOVED: All safe guards (daily loss, drawdown, cooldowns)
#   ❌ REMOVED: VWAP, RSI, EMA, Bollinger Bands entry filters
#
# New Railway Variables:
#   - DIP_TOLERANCE_PCT (default: 0.2) - % above yesterday's low to buy
#   - REBOUND_THRESHOLD_PCT (default: 0.1) - % rebound needed
#   - LOOKBACK_DAYS (default: 1) - days to look back for low
#   - REBOUND_BARS (default: 5) - bars to check for rebound
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

import numpy as np
import pandas as pd
import feedparser
import requests
from urllib.parse import quote

# Google Sheets (optional)
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
# LOGGING
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
    symbols: List[str]
    max_pos_pct: float
    poll_seconds: int
    
    # New v5.0: Dip buying params
    dip_tolerance_pct: float
    rebound_threshold_pct: float
    lookback_days: int
    rebound_bars: int
    
    # ATR-based exits
    atr_sl_multiplier: float
    atr_tp_multiplier: float
    
    # Kept from v4
    max_trades_per_day: int
    min_volume: int
    cooldown_sec: int
    sentiment_enabled: bool
    sentiment_min_threshold: float
    
    # Data
    db_path: str
    data_feed: str
    timeframe_str: str
    timeframe_minutes: int
    fractional_enabled: bool


def load_config() -> Config:
    symbols = os.getenv("SYMBOLS", "NVDA,AMD,MSFT,GOOGL,AAPL,TSLA,META,AMZN").replace(" ", "").split(",")
    
    tf_str = os.getenv("TIMEFRAME_STR", "1Min")
    tf_minutes = int(re.search(r'\d+', tf_str).group()) if re.search(r'\d+', tf_str) else 1
    
    return Config(
        symbols=symbols,
        max_pos_pct=float(os.getenv("MAX_POS_PCT", "0.95")),
        poll_seconds=int(os.getenv("POLL_SECONDS", "30")),
        
        # v5.0: Dip buying strategy
        dip_tolerance_pct=float(os.getenv("DIP_TOLERANCE_PCT", "0.2")),
        rebound_threshold_pct=float(os.getenv("REBOUND_THRESHOLD_PCT", "0.1")),
        lookback_days=int(os.getenv("LOOKBACK_DAYS", "1")),
        rebound_bars=int(os.getenv("REBOUND_BARS", "5")),
        
        atr_sl_multiplier=float(os.getenv("ATR_SL_MULTIPLIER", "1.5")),
        atr_tp_multiplier=float(os.getenv("ATR_TP_MULTIPLIER", "3.0")),
        
        max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", "999")),
        min_volume=int(os.getenv("MIN_VOLUME", "1000")),
        cooldown_sec=int(os.getenv("COOLDOWN_SEC", "60")),
        
        sentiment_enabled=os.getenv("SENTIMENT_ENABLED", "true").lower() == "true",
        sentiment_min_threshold=float(os.getenv("SENTIMENT_MIN_THRESHOLD", "-0.5")),
        
        db_path=os.getenv("DB_PATH", "bot_ultimate.db"),
        data_feed=os.getenv("DATA_FEED", "iex").lower(),
        timeframe_str=tf_str,
        timeframe_minutes=tf_minutes,
        fractional_enabled=os.getenv("FRACTIONAL_ENABLED", "true").lower() == "true",
    )


# ============================================================================
# UTILITIES
# ============================================================================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()

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
    except Exception:
        return 0.0

def is_market_hours() -> bool:
    import pytz
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    if now_ny.weekday() >= 5:
        return False
    start = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= now_ny <= end

def can_trade_now() -> bool:
    if not is_market_hours():
        return False
    import pytz
    tz_ny = pytz.timezone('America/New_York')
    now_ny = datetime.now(tz_ny)
    trade_start = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    return now_ny >= trade_start


# ============================================================================
# SENTIMENT ANALYZER
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
                logger.warning(f"RSS fetch failed {feed_url}: {e}")
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
            logger.warning(f"Google News failed for {symbol}: {e}")
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
                    'negative_count': 0, 'neutral_count': 0, 'total_articles': 0}
        
        pos_count = neg_count = neu_count = 0
        scores = []
        
        for news in news_list[:15]:
            text = (news['title'] + ' ' + news.get('summary', '')).lower()
            pos = sum(1 for kw in self.positive_keywords if kw in text)
            neg = sum(1 for kw in self.negative_keywords if kw in text)
            
            if pos > neg:
                scores.append(min(1.0, pos * 0.2))
                pos_count += 1
            elif neg > pos:
                scores.append(-min(1.0, neg * 0.2))
                neg_count += 1
            else:
                scores.append(0.0)
                neu_count += 1
        
        total = pos_count + neg_count + neu_count
        sentiment = float(np.mean(scores)) if scores else 0.0
        confidence = (max(pos_count, neg_count) / total * min(1.0, total / 10.0)) if total > 0 else 0.0
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_count': pos_count,
            'negative_count': neg_count,
            'neutral_count': neu_count,
            'total_articles': total
        }


# ============================================================================
# DATABASE
# ============================================================================
class TradingDB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.init_schema()

    def init_schema(self):
        self.conn.execute("""CREATE TABLE IF NOT EXISTS bars (
            symbol TEXT, ts_utc TEXT, timeframe TEXT,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY(symbol, ts_utc, timeframe));""")
        
        self.conn.execute("""CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT, symbol TEXT, action TEXT, reason TEXT,
            price REAL, yday_low REAL, rebound_pct REAL,
            equity REAL, cash REAL, position_qty REAL,
            entry_price REAL, tp_price REAL, sl_price REAL,
            atr REAL, sentiment REAL, sentiment_confidence REAL,
            news_count INTEGER, raw_json TEXT);""")
        
        self.conn.execute("""CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY, ts_utc TEXT, symbol TEXT,
            side TEXT, qty REAL, status TEXT, raw_json TEXT);""")
        
        self.conn.execute("""CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT, symbol TEXT, qty REAL,
            market_value REAL, avg_entry_price REAL,
            unrealized_pl REAL, raw_json TEXT);""")
        
        self.conn.execute("""CREATE TABLE IF NOT EXISTS eod_reports (
            date TEXT PRIMARY KEY, start_equity REAL, end_equity REAL,
            pnl_realized REAL, total_trades INTEGER,
            winning_trades INTEGER, report_json TEXT);""")
        
        self.conn.commit()
        logger.info("Database schema initialized")

    def upsert_bar(self, symbol, ts, timeframe, o, h, l, c, v):
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)",
                (symbol, iso(ts), timeframe, float(o), float(h), float(l), float(c), float(v)))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Bar insert failed for {symbol}: {e}")

    def insert_signal(self, ts, symbol, action, reason, **kw):
        self.conn.execute(
            """INSERT INTO signals(ts_utc,symbol,action,reason,price,yday_low,rebound_pct,
               equity,cash,position_qty,entry_price,tp_price,sl_price,atr,
               sentiment,sentiment_confidence,news_count,raw_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (iso(ts), symbol, action, reason,
             kw.get("price"), kw.get("yday_low"), kw.get("rebound_pct"),
             kw.get("equity"), kw.get("cash"), kw.get("position_qty"),
             kw.get("entry_price"), kw.get("tp_price"), kw.get("sl_price"),
             kw.get("atr"), kw.get("sentiment"), kw.get("sentiment_confidence"),
             kw.get("news_count"), json.dumps(kw.get("raw", {}), ensure_ascii=False)))
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

    def get_entry_price(self, symbol):
        cur = self.conn.execute(
            "SELECT entry_price FROM signals WHERE symbol=? AND action='BUY' AND entry_price IS NOT NULL ORDER BY ts_utc DESC LIMIT 1",
            (symbol,))
        row = cur.fetchone()
        return float(row[0]) if row else None

    def get_trades_today(self, symbol, date):
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE symbol=? AND action='BUY' AND DATE(ts_utc)=?",
            (symbol, date))
        row = cur.fetchone()
        return int(row[0]) if row else 0


# ============================================================================
# BOT v5.0 - AGGRESSIVE DIP BUYER
# ============================================================================
class AggressiveBot:
    def __init__(self, cfg: Config, db: TradingDB):
        self.cfg = cfg
        self.db = db
        
        key = os.environ["APCA_API_KEY_ID"]
        secret = os.environ["APCA_API_SECRET_KEY"]
        self.trading = TradingClient(key, secret, paper=True)
        self.data = StockHistoricalDataClient(key, secret)
        
        self.sentiment = SentimentAnalyzer() if cfg.sentiment_enabled else None
        
        self.last_trade: Dict[str, float] = {s: 0.0 for s in cfg.symbols}
        self.entry_price: Dict[str, float] = {}
        self.start_equity: Optional[float] = None
        
        self._load_entry_prices()
        
        logger.info(f"AggressiveBot v5.0 | Strategy: DIP BUYING (like SAPER)")
        logger.info(f"Entry: Yesterday's low + {cfg.dip_tolerance_pct}% | Rebound: {cfg.rebound_threshold_pct}%")
        logger.info(f"Position sizing: ALL-IN ({cfg.max_pos_pct*100}%) | ATR SL={cfg.atr_sl_multiplier}x TP={cfg.atr_tp_multiplier}x")
        logger.info(f"Sentiment filter: {'ON' if cfg.sentiment_enabled else 'OFF'}")
        logger.info(f"Symbols: {cfg.symbols}")

    def _load_entry_prices(self):
        for sym in self.cfg.symbols:
            entry = self.db.get_entry_price(sym)
            if entry:
                self.entry_price[sym] = entry

    def fetch_historical_low(self, sym: str) -> Optional[float]:
        """Fetch yesterday's low (or N days back)"""
        try:
            end_dt = utc_now()
            start_dt = end_dt - timedelta(days=self.cfg.lookback_days + 1)
            
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start_dt,
                end=end_dt,
                feed=self.cfg.data_feed
            )
            
            df = self.data.get_stock_bars(req).df
            
            if df is None or len(df) == 0:
                return None
            
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                df = df[df["symbol"] == sym].set_index("timestamp")
            
            if len(df) < self.cfg.lookback_days:
                return None
            
            # Get yesterday's low (last complete day)
            yesterday_low = float(df['low'].iloc[-2])
            return yesterday_low
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {sym}: {e}")
            return None

    def fetch_bars(self, sym: str) -> pd.DataFrame:
        try:
            tf_min = self.cfg.timeframe_minutes
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame(tf_min, TimeFrameUnit.Minute),
                start=utc_now() - timedelta(minutes=300 * tf_min),  # 300 bars
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
            logger.error(f"fetch_bars({sym}): {e}")
            return pd.DataFrame()

    def validate_bars(self, df: pd.DataFrame) -> bool:
        if df.empty or len(df) < max(self.cfg.rebound_bars + 5, 20):
            return False
        if pd.isna(df["close"].iloc[-1]) or df["close"].iloc[-1] <= 0:
            return False
        if df["volume"].tail(10).sum() < self.cfg.min_volume:
            return False
        return True

    def submit_order(self, sym: str, side: OrderSide, qty: float):
        if qty <= 0 or (self.cfg.fractional_enabled and qty < 0.01):
            return None
        try:
            qty = round(qty, 4) if self.cfg.fractional_enabled else int(qty)
            order = self.trading.submit_order(
                MarketOrderRequest(symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY))
            self.db.upsert_order(order)
            self.last_trade[sym] = time.time()
            logger.info(f"{sym}: {side} {qty} shares")
            return order
        except Exception as e:
            logger.error(f"{sym}: Order failed | {e}")
            return None

    def can_trade_today(self, symbol):
        today = utc_now().strftime("%Y-%m-%d")
        return self.db.get_trades_today(symbol, today) < self.cfg.max_trades_per_day

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
        
        try:
            positions = {p.symbol: p for p in self.trading.get_all_positions()}
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            positions = {}
        
        logger.info(f"Account: equity=${equity:.2f} cash=${cash:.2f}")
        
        for sym in self.cfg.symbols:
            try:
                self._process_symbol(sym, equity, cash, positions)
            except Exception as e:
                logger.error(f"{sym}: Error | {e}")
                traceback.print_exc()
        
        # Snapshot positions
        ts = utc_now()
        for p in self.trading.get_all_positions():
            self.db.insert_position(ts, p)

    def _process_symbol(self, sym, equity, cash, positions):
        # Fetch bars
        df = self.fetch_bars(sym)
        if not self.validate_bars(df):
            return
        
        # Store bars
        for ts, row in df.tail(3).iterrows():
            ts_dt = ts.to_pydatetime()
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            self.db.upsert_bar(sym, ts_dt, self.cfg.timeframe_str,
                               row["open"], row["high"], row["low"], row["close"], row["volume"])
        
        price = float(df["close"].iloc[-1])
        atr_val = calculate_atr(df, 14)
        
        # Position info
        pos = positions.get(sym)
        pos_qty = float(pos.qty) if pos else 0.0
        
        # Entry price from Alpaca
        entry = None
        if pos and hasattr(pos, "avg_entry_price") and pos.avg_entry_price:
            try:
                entry = float(pos.avg_entry_price)
                self.entry_price[sym] = entry
            except Exception:
                pass
        if entry is None and pos_qty == 0:
            entry = self.entry_price.get(sym)
        
        # ATR-based TP/SL
        if entry and atr_val > 0:
            tp_price = entry + (atr_val * self.cfg.atr_tp_multiplier)
            sl_price = entry - (atr_val * self.cfg.atr_sl_multiplier)
        else:
            tp_price = sl_price = None
        
        # === SENTIMENT ===
        sentiment_score = 0.0
        sentiment_confidence = 0.0
        news_count = 0
        sentiment_ok = True
        
        if self.cfg.sentiment_enabled and self.sentiment:
            try:
                news_list = self.sentiment.get_recent_news(sym, max_age_hours=4)
                if news_list:
                    sd = self.sentiment.analyze_sentiment(news_list)
                    sentiment_score = sd['sentiment']
                    sentiment_confidence = sd['confidence']
                    news_count = sd['total_articles']
                    
                    if sentiment_score < self.cfg.sentiment_min_threshold:
                        sentiment_ok = False
                        logger.info(f"{sym} | BLOCKED by negative sentiment ({sentiment_score:.2f})")
            except Exception as e:
                logger.warning(f"{sym}: Sentiment failed | {e}")
        
        # === v5.0: DIP BUYING STRATEGY ===
        
        action = "HOLD"
        reason = "none"
        yday_low = None
        rebound_pct = None
        
        if pos_qty == 0:
            # NO POSITION - Look for entry
            
            # 1. Get yesterday's low
            yday_low = self.fetch_historical_low(sym)
            
            if yday_low is None:
                logger.debug(f"{sym}: No historical data")
                return
            
            # 2. Check if price is near yesterday's low (within tolerance)
            dip_threshold = yday_low * (1 + self.cfg.dip_tolerance_pct / 100.0)
            is_at_dip = price <= dip_threshold
            
            # 3. Check for rebound (price rising vs recent avg)
            recent_bars = df.tail(self.cfg.rebound_bars)
            avg_recent = recent_bars['close'].mean()
            rebound_pct = ((price - avg_recent) / avg_recent) * 100
            is_rebounding = rebound_pct >= self.cfg.rebound_threshold_pct
            
            # 4. Combine conditions
            entry_signal = is_at_dip and is_rebounding and sentiment_ok
            
            cooled = (time.time() - self.last_trade[sym]) >= self.cfg.cooldown_sec
            can_trade = self.can_trade_today(sym)
            
            logger.info(f"{sym} | ${price:.2f} | YdayLow=${yday_low:.2f} Thresh=${dip_threshold:.2f} | "
                         f"Dip={'YES' if is_at_dip else 'NO'} Rebound={rebound_pct:+.2f}% | "
                         f"Sentiment={sentiment_score:+.2f} | ATR=${atr_val:.2f}")
            
            if entry_signal and cooled and can_trade:
                # ALL-IN position sizing
                qty = (cash * self.cfg.max_pos_pct) / price
                
                if not self.cfg.fractional_enabled:
                    qty = int(qty)
                
                if qty > 0:
                    self.last_trade[sym] = time.time()
                    action = "BUY"
                    reason = f"dip_entry_yday_low+{self.cfg.dip_tolerance_pct}%"
                    
                    logger.info(
                        f"{sym} | 🎯 BUY {qty:.4f} @${price:.2f} | ALL-IN ({self.cfg.max_pos_pct*100}%) | "
                        f"YdayLow=${yday_low:.2f} Rebound={rebound_pct:+.2f}% | "
                        f"TP=${price + atr_val * self.cfg.atr_tp_multiplier:.2f} "
                        f"SL=${price - atr_val * self.cfg.atr_sl_multiplier:.2f}")
                    
                    order = self.submit_order(sym, OrderSide.BUY, qty)
                    
                    if order:
                        self.entry_price[sym] = price
                        entry = price
                        if atr_val > 0:
                            tp_price = entry + (atr_val * self.cfg.atr_tp_multiplier)
                            sl_price = entry - (atr_val * self.cfg.atr_sl_multiplier)
                        cash -= qty * price
            
            elif entry_signal and not can_trade:
                logger.info(f"{sym} | Max trades/day reached ({self.cfg.max_trades_per_day})")
        
        else:
            # HAVE POSITION - Look for exit
            
            if tp_price and sl_price:
                logger.info(f"{sym} | pos={pos_qty:.4f} entry=${entry:.2f} TP=${tp_price:.2f} SL=${sl_price:.2f} | price=${price:.2f}")
            
            tp_hit = (tp_price is not None and price >= tp_price)
            sl_hit = (sl_price is not None and price <= sl_price)
            cooled = (time.time() - self.last_trade[sym]) >= self.cfg.cooldown_sec
            
            if (tp_hit or sl_hit) and cooled:
                action = "SELL"
                reason = "take_profit" if tp_hit else "stop_loss"
                
                qty = pos_qty if self.cfg.fractional_enabled else int(pos_qty)
                pnl_pct = ((price - entry) / entry * 100) if entry else 0
                
                logger.info(f"{sym} | 💥 SELL {qty:.4f} @${price:.2f} | {reason} | P/L: {pnl_pct:+.2f}%")
                
                order = self.submit_order(sym, OrderSide.SELL, qty)
                
                if order:
                    self.entry_price.pop(sym, None)
        
        # Log signal if action taken
        if action != "HOLD":
            self.db.insert_signal(
                ts=utc_now(), symbol=sym, action=action, reason=reason,
                price=price, yday_low=yday_low, rebound_pct=rebound_pct,
                equity=equity, cash=cash, position_qty=pos_qty,
                entry_price=entry, tp_price=tp_price, sl_price=sl_price,
                atr=atr_val, sentiment=sentiment_score,
                sentiment_confidence=sentiment_confidence, news_count=news_count,
                raw={"is_at_dip": yday_low and price <= yday_low * (1 + self.cfg.dip_tolerance_pct / 100.0),
                     "tp_hit": tp_hit if action == "SELL" else None,
                     "sl_hit": sl_hit if action == "SELL" else None})

    def generate_eod_report(self):
        today = utc_now().strftime("%Y-%m-%d")
        try:
            end_equity = float(self.trading.get_account().equity)
        except Exception:
            end_equity = 0.0
        
        start_equity = self.start_equity or end_equity
        
        cur = self.db.conn.execute(
            "SELECT action, entry_price, price FROM signals WHERE DATE(ts_utc)=? AND action IN ('BUY','SELL')",
            (today,))
        trades = cur.fetchall()
        
        total = len([t for t in trades if t[0] == 'SELL'])
        wins = sum(1 for a, e, p in trades if a == 'SELL' and e and p and p > e)
        pnl = end_equity - start_equity
        
        report = {
            "date": today, "start_equity": start_equity, "end_equity": end_equity,
            "pnl_realized": pnl, "pnl_pct": (pnl / start_equity * 100) if start_equity > 0 else 0,
            "total_trades": total, "winning_trades": wins,
            "win_rate": (wins / total * 100) if total > 0 else 0
        }
        
        self.db.conn.execute(
            "INSERT OR REPLACE INTO eod_reports VALUES (?,?,?,?,?,?,?)",
            (today, start_equity, end_equity, pnl, total, wins, json.dumps(report)))
        self.db.conn.commit()
        
        logger.info(f"\n{'='*60}\nEOD REPORT {today}\n{'='*60}\n"
                     f"Start: ${start_equity:,.2f} | End: ${end_equity:,.2f} | P/L: ${pnl:+,.2f} ({report['pnl_pct']:+.2f}%)\n"
                     f"Trades: {total} | Wins: {wins} | Rate: {report['win_rate']:.1f}%\n{'='*60}")
        return report

    def run(self):
        logger.info("="*60)
        logger.info("AGGRESSIVE BOT v5.0 - DIP BUYING STRATEGY")
        logger.info("="*60)
        logger.info(f"Strategy: Buy at yesterday's low + {self.cfg.dip_tolerance_pct}% when rebounding")
        logger.info(f"Position sizing: ALL-IN ({self.cfg.max_pos_pct*100}% per trade)")
        logger.info(f"Exit: ATR-based TP={self.cfg.atr_tp_multiplier}x SL={self.cfg.atr_sl_multiplier}x")
        logger.info(f"Sentiment filter: {'ENABLED' if self.cfg.sentiment_enabled else 'DISABLED'}")
        logger.info(f"Max trades/day: {self.cfg.max_trades_per_day}")
        logger.info("="*60)
        
        try:
            acct = self.trading.get_account()
            logger.info(f"Account: equity=${acct.equity} cash=${acct.cash}")
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
        
        last_eod_date = None
        
        while True:
            try:
                if not can_trade_now():
                    if not is_market_hours():
                        current_date = utc_now().strftime("%Y-%m-%d")
                        if current_date != last_eod_date:
                            self.generate_eod_report()
                            last_eod_date = current_date
                        logger.info("Market closed")
                    else:
                        logger.info("Pre-market - waiting...")
                    time.sleep(300)
                    continue
                
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
    print("="*60)
    print("AGGRESSIVE TRADING BOT v5.0")
    print("Strategy: Dip Buying (like SAPER) + Sentiment Filter")
    print("="*60)
    
    required = ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"ERROR: Missing: {', '.join(missing)}")
        sys.exit(1)
    
    cfg = load_config()
    db = TradingDB(cfg.db_path)
    bot = AggressiveBot(cfg, db)
    bot.run()


if __name__ == "__main__":
    main()
