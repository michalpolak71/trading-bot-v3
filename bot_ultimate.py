# bot_ultimate.py — ULTIMATE TRADING BOT v7.0 ADAPTIVE
# Strategy: Yesterday's Low + Rebound + Sentiment + Regime Detection + Claude AI Market Analysis
#
# ============================================================================
# v7.0 CHANGES (Mar 2026):
# ============================================================================
#   ✅ MARKET ANALYZER — Claude AI ocenia DIP vs CRASH co 30 min
#   ✅ CRASH BUYING — mała pozycja (30%) gdy RSI<32 + wolumen maleje
#   ✅ TRAILING STOP — SL przesuwa się w górę za ceną (nigdy w dół)
#   ✅ CRASH EXIT — sprzedaj stratne pozycje gdy Claude wykryje krach
#   ✅ PDT PROTECTION — nie sprzedawaj akcji kupionych tego samego dnia
#   ✅ HISTORICAL FIX — pobiera 7 dni wstecz (fix weekendy/święta)
#   ✅ REGIME GATE — blokuje BUY w BEAR, pozwala SELL (fix z v6.0)
# ============================================================================
# v6.0 CHANGES (Mar 2026):
# ============================================================================
#   ✅ REGIME DETECTION - bot wyłącza się w BEAR market (fix straty 27.02!)
#   ✅ PDT FIX - zmiana TimeInForce.DAY dla fractional shares
#   ✅ ML ANALYZER - zapisuje i analizuje wyniki trades
#   ✅ Wszystkie zmiany WEWNĄTRZ tego pliku (zero nowych plików)
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
    
    # v5.0: Dip buying params
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
    
    # v7.0: Regime Detection
    regime_enabled: bool
    regime_check_interval: int   # sekundy między sprawdzeniami (domyślnie 3600 = 1h)
    regime_spy_trend_threshold: float  # % spadku SPY w 5 dni żeby wyłączyć bota

    # v7.0: Market Analyzer (Claude AI)
    market_analysis_enabled: bool
    market_analysis_interval: int    # sekundy między analizami (domyślnie 1800 = 30 min)
    trailing_stop_pct: float         # % trailing stop gdy DIP (domyślnie 3.0)
    crash_buy_pct: float             # % kapitału przy crash-buying (domyślnie 0.30)
    bot_sync_enabled: bool           # czy synchronizować z GPW botem
    bot_id: str                      # NYSE lub GPW
    opp_tp_pct: float                # % TP dla OPPORTUNISTIC/CRASH BUY (domyślnie 1.5)
    opp_sl_pct: float                # % SL dla OPPORTUNISTIC/CRASH BUY (domyślnie 1.0)
    opp_max_position: float          # max $ na jedną pozycję OPPORTUNISTIC (domyślnie 300)
    opp_daily_limit: float           # max $ łącznego ryzyka OPPORTUNISTIC dziennie (domyślnie 500)
    opp_min_rebound: float           # min Rebound% żeby wejść (domyślnie -0.5)
    
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
        
        # v5.0
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
        
        # v7.0: Regime Detection
        regime_enabled=os.getenv("REGIME_ENABLED", "true").lower() == "true",
        regime_check_interval=int(os.getenv("REGIME_CHECK_INTERVAL", "3600")),
        regime_spy_trend_threshold=float(os.getenv("REGIME_SPY_THRESHOLD", "-2.0")),

        # v7.0: Market Analyzer
        market_analysis_enabled=os.getenv("MARKET_ANALYSIS_ENABLED", "true").lower() == "true",
        market_analysis_interval=int(os.getenv("MARKET_ANALYSIS_INTERVAL", "1800")),
        trailing_stop_pct=float(os.getenv("TRAILING_STOP_PCT", "3.0")),
        crash_buy_pct=float(os.getenv("CRASH_BUY_PCT", "0.30")),
        opp_tp_pct=float(os.getenv("OPP_TP_PCT", "1.5")),
        opp_sl_pct=float(os.getenv("OPP_SL_PCT", "1.0")),
        opp_max_position=float(os.getenv("OPP_MAX_POSITION", "300")),
        opp_daily_limit=float(os.getenv("OPP_DAILY_LIMIT", "500")),
        opp_min_rebound=float(os.getenv("OPP_MIN_REBOUND", "-0.5")),
        bot_sync_enabled=os.getenv("BOT_SYNC_ENABLED", "false").lower() == "true",
        bot_id=os.getenv("BOT_ID", "NYSE"),
        
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
# v7.0: MARKET ANALYZER — Claude AI ocenia czy spadek to dip czy krach
# ============================================================================
class MarketAnalyzer:
    """
    Łączy dane techniczne SPY z analizą Claude AI.
    
    Tryby pozycji (position_mode):
      DIP       → chwilowy spadek, trzymaj pozycję z trailing stopem
      CRASH     → silny trend spadkowy, sprzedaj pozycje na minusie
      UNKNOWN   → brak danych, zachowaj się jak przy DIP (ostrożnie)
    
    Analiza co MARKET_ANALYSIS_INTERVAL sekund (domyślnie 1800 = 30 min).
    """

    MODE_DIP   = "DIP"
    MODE_CRASH = "CRASH"
    MODE_UNKNOWN = "UNKNOWN"

    def __init__(self, data_client, data_feed: str = "iex",
                 check_interval: int = 1800, anthropic_key: str = ""):
        self.data_client   = data_client
        self.data_feed     = data_feed
        self.check_interval = check_interval
        self.anthropic_key = anthropic_key
        self.enabled       = bool(anthropic_key)

        self._mode         = self.MODE_UNKNOWN
        self._reasoning    = "Brak analizy"
        self._buy_signal   = False   # Claude mówi: teraz dobry moment na zakup?
        self._last_check   = 0
        self._last_spy_data = {}     # cache danych SPY dla BotSync
        self._trailing_sl: Dict[str, float] = {}   # sym → aktualny trailing SL

    # ------------------------------------------------------------------
    # Dane techniczne SPY
    # ------------------------------------------------------------------
    def _get_spy_data(self) -> dict:
        """Zbiera wskaźniki SPY: ceny, EMA20/50, RSI14, zmiana 1d/5d/20d."""
        try:
            req = StockBarsRequest(
                symbol_or_symbols="SPY",
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=utc_now() - timedelta(days=70),
                end=utc_now(),
                feed=self.data_feed
            )
            df = self.data_client.get_stock_bars(req).df
            if df is None or df.empty:
                return {}

            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                df = df[df["symbol"] == "SPY"].set_index("timestamp")
            df = df.sort_index()

            closes = df["close"].tolist()
            volumes = df["volume"].tolist()
            if len(closes) < 20:
                return {}

            def ema(prices, span):
                k = 2 / (span + 1)
                e = prices[0]
                for p in prices[1:]:
                    e = p * k + e * (1 - k)
                return e

            # RSI 14
            deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            gains  = [max(d, 0) for d in deltas[-14:]]
            losses = [abs(min(d, 0)) for d in deltas[-14:]]
            avg_g  = sum(gains) / 14 if gains else 0
            avg_l  = sum(losses) / 14 if losses else 1
            rsi    = 100 - (100 / (1 + avg_g / avg_l)) if avg_l > 0 else 50

            current = closes[-1]
            vol_avg5  = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else 0
            vol_avg20 = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 1

            # Wolumen ostatnich 3 vs poprzednich 3 dni (trend wolumenu)
            vol_trend = round(sum(volumes[-3:]) / sum(volumes[-6:-3]), 2) if len(volumes) >= 6 and sum(volumes[-6:-3]) > 0 else 1.0

            return {
                "current_price": round(current, 2),
                "ema20":  round(ema(closes[-20:], 20), 2),
                "ema50":  round(ema(closes[-50:] if len(closes) >= 50 else closes, 50), 2),
                "rsi14":  round(rsi, 1),
                "change_1d":  round((closes[-1] - closes[-2]) / closes[-2] * 100, 2) if len(closes) >= 2 else 0,
                "change_5d":  round((closes[-1] - closes[-6]) / closes[-6] * 100, 2) if len(closes) >= 6 else 0,
                "change_10d": round((closes[-1] - closes[-11]) / closes[-11] * 100, 2) if len(closes) >= 11 else 0,
                "change_20d": round((closes[-1] - closes[-21]) / closes[-21] * 100, 2) if len(closes) >= 21 else 0,
                "volume_ratio": round(vol_avg5 / vol_avg20, 2) if vol_avg20 > 0 else 1.0,
                "volume_trend": vol_trend,  # >1 = wolumen rośnie (panika), <1 = maleje (wyczerpanie)
                "bars_count": len(closes),
            }
        except Exception as e:
            logger.error(f"MarketAnalyzer: błąd danych SPY: {e}")
            return {}

    # ------------------------------------------------------------------
    # Zapytaj Claude AI
    # ------------------------------------------------------------------
    def _ask_claude(self, spy: dict) -> tuple:
        """
        Pyta Claude AI o ocenę sytuacji rynkowej.
        Zwraca (mode: str, reasoning: str).
        """
        if not self.anthropic_key:
            return self.MODE_UNKNOWN, "Brak klucza Anthropic API"

        prompt = f"""Jesteś ekspertem analizy technicznej giełdy USA.
Przeanalizuj dane SPY (S&P 500 ETF) i oceń sytuację rynkową.

Dane SPY:
- Cena: ${spy.get('current_price')} | EMA20: {spy.get('ema20')} | EMA50: {spy.get('ema50')}
- RSI14: {spy.get('rsi14')}
- Zmiana 1d: {spy.get('change_1d')}% | 5d: {spy.get('change_5d')}% | 10d: {spy.get('change_10d')}% | 20d: {spy.get('change_20d')}%
- Wolumen ratio (5d/20d): {spy.get('volume_ratio')}x | Trend wolumenu (3d/3d-prev): {spy.get('volume_trend')}x

Zasady oceny:
NIE KUPUJ (CRASH) gdy:
- Trend spadkowy > 2 tygodnie (change_10d < -5%)
- Wolumen sprzedaży rośnie (volume_trend > 1.2 = panika)
- RSI > 35 (rynek jeszcze nie wyprzedany)

KUP OSTROŻNIE (DIP) gdy:
- RSI < 32 (rynek mocno wyprzedany)
- Wolumen sprzedaży maleje (volume_trend < 0.85 = wyczerpanie paniki)
- Lub pojawił się sygnał odwrócenia (RSI < 35 + volume_trend < 1.0)

Zwróć jeden z trybów:
- DIP = chwilowa korekta, można kupować (mała pozycja)
- CRASH = silny trend spadkowy, nie kupuj, zamknij straty

Odpowiedz DOKŁADNIE w tym formacie JSON (nic poza JSON):
{{"mode": "DIP" lub "CRASH", "confidence": 0-100, "buy_signal": true/false, "reasoning": "max 2 zdania po polsku"}}"""

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"].strip()

            # Usuń ewentualne markdown backticki
            text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)

            mode       = result.get("mode", self.MODE_UNKNOWN).upper()
            confidence = result.get("confidence", 50)
            reasoning  = result.get("reasoning", "brak uzasadnienia")
            buy_signal = result.get("buy_signal", False)

            if mode not in (self.MODE_DIP, self.MODE_CRASH):
                mode = self.MODE_UNKNOWN

            self._buy_signal = bool(buy_signal)
            return mode, f"[{confidence}% pewności] {reasoning}"

        except Exception as e:
            logger.error(f"MarketAnalyzer: błąd Claude API: {e}")
            return self.MODE_UNKNOWN, f"Błąd API: {e}"

    # ------------------------------------------------------------------
    # Główna metoda — wywołaj co 30 min
    # ------------------------------------------------------------------
    def analyze(self) -> str:
        """Zwraca aktualny tryb: DIP / CRASH / UNKNOWN. Używa cache."""
        now = time.time()
        if (now - self._last_check) < self.check_interval and self._mode != self.MODE_UNKNOWN:
            return self._mode

        spy = self._get_spy_data()
        if not spy:
            logger.warning("MarketAnalyzer: brak danych SPY — tryb UNKNOWN")
            self._mode = self.MODE_UNKNOWN
            self._reasoning = "Brak danych SPY"
            self._last_check = now
            return self._mode

        self._last_spy_data = spy  # cache dla BotSync
        mode, reasoning = self._ask_claude(spy)
        self._mode      = mode
        self._reasoning = reasoning
        self._last_check = now

        emoji = "📉" if mode == self.MODE_CRASH else "🔄" if mode == self.MODE_DIP else "❓"
        logger.info(
            f"MarketAnalyzer {emoji} {mode} | "
            f"SPY={spy['current_price']} RSI={spy['rsi14']} "
            f"5d={spy['change_5d']}% EMA20/50={spy['ema20']}/{spy['ema50']} | "
            f"{reasoning}"
        )
        return self._mode

    # ------------------------------------------------------------------
    # Trailing Stop
    # ------------------------------------------------------------------
    def update_trailing_sl(self, sym: str, price: float, entry: float,
                           trail_pct: float = 3.0) -> float:
        """
        Przesuwa SL w górę gdy cena rośnie (trailing stop).
        SL nigdy nie idzie w dół.
        Zwraca aktualny SL.
        """
        min_sl  = entry * (1 - trail_pct / 100)   # nigdy nie przekrocz straty od entry
        new_sl  = price * (1 - trail_pct / 100)

        current = self._trailing_sl.get(sym, min_sl)
        updated = max(current, new_sl, min_sl)
        self._trailing_sl[sym] = updated
        return updated

    def get_trailing_sl(self, sym: str, entry: float, trail_pct: float = 3.0) -> float:
        """Zwraca aktualny trailing SL dla symbolu (lub domyślny od entry)."""
        return self._trailing_sl.get(sym, entry * (1 - trail_pct / 100))

    def reset_trailing_sl(self, sym: str):
        """Usuń trailing SL po sprzedaży pozycji."""
        self._trailing_sl.pop(sym, None)

    def get_buy_signal(self) -> bool:
        """Czy Claude AI dał sygnał kupna? (RSI oversold + wyczerpanie paniki)"""
        return self._buy_signal

    def get_status(self) -> str:
        return f"{self._mode}: {self._reasoning}"


# ============================================================================
# v7.0: REGIME DETECTOR (wbudowany - bez zewnętrznych plików)
# ============================================================================
class RegimeDetector:
    """
    Wykrywa typ rynku na podstawie SPY (proxy S&P500).
    Wyłącza bota gdy rynek spada - to był problem 27.02.2026!
    
    Regime'y:
      BULL_TRENDING   → Bot aktywny, kup dipy
      BEAR_TRENDING   → Bot WYŁĄCZONY (SPY w dół + EMA death cross)
      HIGH_VOLATILITY → Bot WYŁĄCZONY (za ryzykowne)
      SIDEWAYS        → Bot aktywny, ostrożnie
    """
    
    def __init__(self, data_client, data_feed: str = "iex", check_interval: int = 3600):
        self.data_client = data_client
        self.data_feed = data_feed
        self.check_interval = check_interval
        
        # Cache
        self._last_regime = None
        self._last_check = 0
        self._last_reason = "Nie sprawdzono"
    
    def _get_spy_closes(self, days: int = 60) -> List[float]:
        """Pobiera zamknięcia SPY z ostatnich N dni"""
        try:
            req = StockBarsRequest(
                symbol_or_symbols="SPY",
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=utc_now() - timedelta(days=days + 10),
                end=utc_now(),
                feed=self.data_feed
            )
            df = self.data_client.get_stock_bars(req).df
            
            if df is None or df.empty:
                return []
            
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                df = df[df["symbol"] == "SPY"].set_index("timestamp")
            
            return df['close'].tolist()
        except Exception as e:
            logger.error(f"RegimeDetector: błąd pobierania SPY: {e}")
            return []
    
    def _calc_ema(self, prices: List[float], span: int) -> float:
        """Liczy EMA dla listy cen"""
        if len(prices) < span:
            return prices[-1] if prices else 0.0
        k = 2 / (span + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = p * k + ema * (1 - k)
        return ema
    
    def should_trade(self, spy_threshold_pct: float = -2.0) -> bool:
        """
        Główna metoda - zwraca True jeśli bot powinien tradować.
        Używa cache żeby nie odpytywać API co 30 sekund.
        """
        now = time.time()
        
        # Użyj cache jeśli świeży
        if self._last_regime is not None and (now - self._last_check) < self.check_interval:
            return self._last_regime
        
        logger.info("🔍 RegimeDetector: sprawdzam rynek (SPY)...")
        
        closes = self._get_spy_closes(days=60)
        
        if len(closes) < 20:
            logger.warning("RegimeDetector: za mało danych SPY, domyślnie STOP")
            self._last_regime = False
            self._last_check = now
            return False
        
        # EMA 20 i 50
        ema_20 = self._calc_ema(closes[-20:], 20)
        ema_50 = self._calc_ema(closes[-50:] if len(closes) >= 50 else closes, 50)
        
        # SPY trend ostatnie 5 dni
        spy_5d = ((closes[-1] - closes[-6]) / closes[-6] * 100) if len(closes) >= 6 else 0
        
        # Volatility
        changes = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        volatility = (sum((x - sum(changes)/len(changes))**2 for x in changes) / len(changes)) ** 0.5 if changes else 0
        vix_proxy = volatility * 100 * (252 ** 0.5)
        
        # === LOGIKA DECYZYJNA ===
        
        # BEAR - najważniejszy przypadek
        if ema_20 < ema_50 and spy_5d < spy_threshold_pct:
            self._last_reason = f"🛑 BEAR: EMA20({ema_20:.1f}) < EMA50({ema_50:.1f}), SPY 5d: {spy_5d:.1f}%"
            logger.warning(f"RegimeDetector: BEAR MARKET | {self._last_reason}")
            self._last_regime = False
            self._last_check = now
            return False
        
        # HIGH VOLATILITY
        if vix_proxy > 30:
            self._last_reason = f"🛑 HIGH VOL: VIX proxy={vix_proxy:.1f}"
            logger.warning(f"RegimeDetector: HIGH VOLATILITY | {self._last_reason}")
            self._last_regime = False
            self._last_check = now
            return False
        
        # BULL lub SIDEWAYS - traduj
        if ema_20 >= ema_50:
            self._last_reason = f"✅ BULL: EMA20({ema_20:.1f}) >= EMA50({ema_50:.1f}), SPY 5d: {spy_5d:.1f}%"
        else:
            self._last_reason = f"✅ SIDEWAYS: SPY 5d: {spy_5d:.1f}% (powyżej progu {spy_threshold_pct}%)"
        
        logger.info(f"RegimeDetector: OK | {self._last_reason}")
        self._last_regime = True
        self._last_check = now
        return True
    
    def get_status(self) -> str:
        return self._last_reason


# ============================================================================
# v7.0: ML ANALYZER (wbudowany)
# ============================================================================
class MLAnalyzer:
    """
    Analizuje historię trades z SQLite i wyciąga wzorce.
    Co 6h drukuje raport do logów Railway.
    """
    
    def __init__(self, db_conn):
        self.db = db_conn
        self._last_analysis = 0
        self.ANALYSIS_INTERVAL = 6 * 3600  # co 6h
    
    def maybe_analyze(self):
        """Uruchamia analizę co 6h jeśli jest wystarczająco danych"""
        now = time.time()
        if now - self._last_analysis < self.ANALYSIS_INTERVAL:
            return
        self._last_analysis = now
        self._run_analysis()
    
    def _run_analysis(self):
        try:
            cur = self.db.execute(
                "SELECT symbol, action, price, entry_price, atr, sentiment, rebound_pct FROM signals "
                "WHERE action='SELL' AND entry_price IS NOT NULL ORDER BY ts_utc DESC LIMIT 200"
            )
            trades = cur.fetchall()
            
            if len(trades) < 5:
                logger.info(f"📚 ML: Za mało trades ({len(trades)}) do analizy. Minimum: 5")
                return
            
            logger.info(f"\n{'='*50}\n📊 ML ANALYSIS ({len(trades)} trades)\n{'='*50}")
            
            # Oblicz P/L dla każdego trade
            results = []
            for sym, action, price, entry, atr, sentiment, rebound in trades:
                if entry and entry > 0:
                    pl_pct = (price - entry) / entry * 100
                    results.append({
                        'symbol': sym, 'pl_pct': pl_pct,
                        'atr': atr or 0, 'sentiment': sentiment or 0,
                        'rebound': rebound or 0, 'win': pl_pct > 0
                    })
            
            if not results:
                return
            
            wins = [r for r in results if r['win']]
            losses = [r for r in results if not r['win']]
            win_rate = len(wins) / len(results) * 100
            avg_pl = sum(r['pl_pct'] for r in results) / len(results)
            
            logger.info(f"Win rate: {win_rate:.1f}% | Avg P/L: {avg_pl:.2f}%")
            logger.info(f"Wins: {len(wins)} | Losses: {len(losses)}")
            
            # Symbol performance
            symbols = set(r['symbol'] for r in results)
            sym_stats = []
            for sym in symbols:
                sym_trades = [r for r in results if r['symbol'] == sym]
                sym_wins = [r for r in sym_trades if r['win']]
                sym_avg = sum(r['pl_pct'] for r in sym_trades) / len(sym_trades)
                sym_stats.append((sym, len(sym_trades), len(sym_wins)/len(sym_trades)*100, sym_avg))
            
            sym_stats.sort(key=lambda x: x[3], reverse=True)
            logger.info("\n📈 SYMBOL RANKING:")
            for sym, count, wr, avg in sym_stats:
                icon = "✅" if avg > 0 else "❌"
                logger.info(f"  {icon} {sym}: {count} trades | WR: {wr:.0f}% | Avg: {avg:.2f}%")
            
            # Sentiment analysis
            pos_sentiment = [r for r in results if r['sentiment'] > 0.1]
            neg_sentiment = [r for r in results if r['sentiment'] < -0.05]
            
            if pos_sentiment:
                pos_wr = sum(1 for r in pos_sentiment if r['win']) / len(pos_sentiment) * 100
                logger.info(f"\n💡 Sentiment > 0.1: {pos_wr:.0f}% win rate ({len(pos_sentiment)} trades)")
            if neg_sentiment:
                neg_wr = sum(1 for r in neg_sentiment if r['win']) / len(neg_sentiment) * 100
                logger.info(f"💡 Sentiment < -0.05: {neg_wr:.0f}% win rate ({len(neg_sentiment)} trades) → UNIKAJ!")
            
            logger.info('='*50)
            
        except Exception as e:
            logger.error(f"ML Analyzer błąd: {e}")


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
        
        # v7.0: Tabela dla regime history
        self.conn.execute("""CREATE TABLE IF NOT EXISTS regime_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT, regime TEXT, should_trade INTEGER, reason TEXT);""")
        
        self.conn.commit()
        logger.info("Database schema initialized (v7.0)")

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

    def log_regime(self, regime: str, should_trade: bool, reason: str):
        self.conn.execute(
            "INSERT INTO regime_log(ts_utc, regime, should_trade, reason) VALUES (?,?,?,?)",
            (iso(utc_now()), regime, int(should_trade), reason))
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

    def bought_today(self, symbol: str) -> bool:
        """Czy kupiłem ten symbol DZIŚ? Jeśli tak, SELL = PDT violation."""
        today = utc_now().strftime("%Y-%m-%d")
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE symbol=? AND action='BUY' AND DATE(ts_utc)=?",
            (symbol, today))
        row = cur.fetchone()
        return int(row[0]) > 0 if row else False

    def sold_today(self, symbol: str) -> bool:
        """Czy sprzedałem ten symbol DZIŚ? Jeśli tak, BUY = day-trade violation."""
        today = utc_now().strftime("%Y-%m-%d")
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM signals WHERE symbol=? AND action='SELL' AND DATE(ts_utc)=?",
            (symbol, today))
        row = cur.fetchone()
        return int(row[0]) > 0 if row else False


# ============================================================================
# v7.0: GOOGLE SHEETS WEBHOOK (przez Apps Script - bez service account)
# ============================================================================
class SheetsWebhook:
    """
    Wysyła dane do Google Sheets przez Apps Script webhook.
    Nie wymaga service account - tylko URL z deploymentu Apps Script.
    
    Ustaw zmienną: SHEETS_WEBHOOK_URL = "https://script.google.com/..."
    """
    
    def __init__(self, webhook_url: str):
        self.url = webhook_url
        self.enabled = bool(webhook_url)
        self.session = requests.Session()
        self._last_regime = "UNKNOWN"
    
    def _send(self, data: dict) -> bool:
        if not self.enabled:
            return False
        try:
            r = self.session.post(self.url, json=data, timeout=10)
            return r.status_code == 200
        except Exception as e:
            logger.warning(f"Sheets webhook błąd: {e}")
            return False
    
    def send_trade(self, symbol: str, action: str, price: float,
                   entry_price: float = None, pl_pct: float = None,
                   reason: str = "", atr: float = 0, sentiment: float = 0,
                   rebound_pct: float = 0, yday_low: float = 0, equity: float = 0):
        return self._send({
            "type": "trade",
            "timestamp": utc_now().isoformat(),
            "symbol": symbol,
            "action": action,
            "price": price,
            "entry_price": entry_price,
            "pl_pct": pl_pct,
            "reason": reason,
            "atr": atr,
            "sentiment": sentiment,
            "rebound_pct": rebound_pct,
            "yday_low": yday_low,
            "equity": equity,
            "regime": self._last_regime
        })
    
    def send_regime(self, regime: str, should_trade: bool, reason: str, spy_trend: float = 0):
        self._last_regime = regime
        return self._send({
            "type": "regime",
            "timestamp": utc_now().isoformat(),
            "regime": regime,
            "should_trade": should_trade,
            "reason": reason,
            "spy_trend": spy_trend
        })
    
    def send_summary(self, date: str, start_equity: float, end_equity: float,
                     total_trades: int, winning_trades: int):
        pnl = end_equity - start_equity
        pnl_pct = round(pnl / start_equity * 100, 2) if start_equity > 0 else 0
        win_rate = round(winning_trades / total_trades * 100, 1) if total_trades > 0 else 0
        return self._send({
            "type": "summary",
            "date": date,
            "start_equity": round(start_equity, 2),
            "end_equity": round(end_equity, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": pnl_pct,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "regime": self._last_regime
        })




# ============================================================================
# v7.0: BOT SYNC — konsultacja z GPW botem przez Google Sheets
# ============================================================================
class BotSync:
    """
    Synchronizuje sygnały między NYSE botem (Railway) a GPW botem (OVH).
    Oba boty piszą swoje sygnały do Google Sheets i czytają sygnały partnera.
    
    Ustaw: BOT_SYNC_ENABLED=true, BOT_ID=NYSE (lub GPW)
    """
    BOT_GPW  = "GPW"
    BOT_NYSE = "NYSE"

    def __init__(self, webhook_url: str, bot_id: str = "NYSE"):
        self.url     = webhook_url
        self.bot_id  = bot_id
        self.other   = self.BOT_GPW if bot_id == self.BOT_NYSE else self.BOT_NYSE
        self.enabled = bool(webhook_url)
        self._cache  = None
        self._cache_ts = 0
        self._CACHE_TTL = 1800  # 30 min
        self._last_published_mode = ""   # ostatni opublikowany tryb
        self._last_publish_ts = 0        # timestamp ostatniej publikacji
        if self.enabled:
            logger.info(f"BotSync: aktywny | bot={bot_id} partner={self.other}")

    def publish(self, market_mode: str, buy_signal: bool,
                reasoning: str, spy_data: dict):
        """Opublikuj sygnał — tylko gdy tryb się zmienia lub co 30 min"""
        if not self.enabled:
            return

        now = time.time()
        mode_changed = (market_mode != self._last_published_mode)
        time_elapsed = (now - self._last_publish_ts) > 1800  # co 30 min nawet bez zmiany

        if not mode_changed and not time_elapsed:
            return  # nic się nie zmieniło — nie zapisuj do Sheets

        # Ustaw od razu — nawet jeśli request się nie uda, nie spamuj Sheets
        self._last_published_mode = market_mode
        self._last_publish_ts = now

        try:
            requests.post(self.url, json={
                "type":        "bot_signal",
                "bot_id":      self.bot_id,
                "timestamp":   utc_now().isoformat(),
                "market_mode": market_mode,
                "buy_signal":  buy_signal,
                "reasoning":   reasoning,
                "spy_rsi":     spy_data.get("rsi14", 0),
                "spy_5d":      spy_data.get("change_5d", 0),
                "spy_price":   spy_data.get("current_price", 0),
                "ema20":       spy_data.get("ema20", 0),
                "ema50":       spy_data.get("ema50", 0),
            }, timeout=10)
            logger.info(f"BotSync ✅ {self.bot_id} → {market_mode} buy={buy_signal} {'(zmiana trybu!)' if mode_changed else '(30min refresh)'}")
        except Exception as e:
            logger.warning(f"BotSync publish błąd: {e}")

    def get_partner(self) -> dict:
        """Pobierz ostatni sygnał partnera (z cache 30 min)"""
        if not self.enabled:
            return {}
        now = time.time()
        if self._cache and (now - self._cache_ts) < self._CACHE_TTL:
            return self._cache
        try:
            r = requests.get(
                self.url,
                params={"type": "get_signal", "bot_id": self.other},
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "ok" and data.get("signal"):
                    self._cache = data["signal"]
                    self._cache_ts = now
                    logger.info(f"BotSync 📥 {self.other}: "
                               f"{self._cache.get('market_mode')} "
                               f"buy={self._cache.get('buy_signal')}")
                    return self._cache
        except Exception as e:
            logger.warning(f"BotSync get_partner błąd: {e}")
        return {}

    def partner_confirms_buy(self) -> tuple:
        """
        Zwraca (potwierdza: bool, powód: str)
        True = partner też widzi DIP → większa pewność
        False = partner widzi CRASH → ostrożniej
        """
        s = self.get_partner()
        if not s:
            return True, "brak sygnału partnera — domyślnie OK"

        mode = s.get("market_mode", "UNKNOWN")
        buy  = s.get("buy_signal", False)
        reason = s.get("reasoning", "")[:80]

        if mode == "CRASH":
            return False, f"{self.other} widzi CRASH — ostrożniej! {reason}"
        if mode == "DIP" and buy:
            return True, f"{self.other} potwierdza DIP+BUY: {reason}"
        return True, f"{self.other}: {mode} (neutralne)"

    def get_claude_context(self) -> str:
        """Kontekst dla Claude AI — co widzi partner"""
        s = self.get_partner()
        if not s:
            return ""
        return (f"\nKontekst od bota {self.other} (ten sam rynek, inna sesja): "
                f"ocena={s.get('market_mode')} buy={s.get('buy_signal')} "
                f"RSI={s.get('spy_rsi')} SPY5d={s.get('spy_5d')}% "
                f"→ {s.get('reasoning','')[:100]}")

# ============================================================================
# BOT v7.0 - ADAPTIVE DIP BUYER + CLAUDE AI
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
        
        # v7.0: Regime Detector
        self.regime = RegimeDetector(
            data_client=self.data,
            data_feed=cfg.data_feed,
            check_interval=cfg.regime_check_interval
        ) if cfg.regime_enabled else None

        # v7.0: Market Analyzer (Claude AI — DIP vs CRASH)
        self.market_analyzer = MarketAnalyzer(
            data_client=self.data,
            data_feed=cfg.data_feed,
            check_interval=cfg.market_analysis_interval,
            anthropic_key=os.getenv("ANTHROPIC_API_KEY", "")
        ) if cfg.market_analysis_enabled else None
        
        # v7.0: ML Analyzer
        self.ml = MLAnalyzer(db.conn)
        
        # v7.0: Google Sheets Webhook
        self.sheets = SheetsWebhook(os.getenv("SHEETS_WEBHOOK_URL", ""))

        # v7.0: Bot Sync (konsultacja z GPW botem przez Google Sheets)
        self.bot_sync = BotSync(
            webhook_url=os.getenv("SHEETS_WEBHOOK_URL", ""),
            bot_id=cfg.bot_id
        ) if cfg.bot_sync_enabled else None
        
        self.last_trade: Dict[str, float] = {s: 0.0 for s in cfg.symbols}
        self.entry_price: Dict[str, float] = {}
        self.start_equity: Optional[float] = None
        self._opp_daily_spent: float = 0.0   # ile $ wydano na OPPORTUNISTIC dziś
        self._opp_daily_date: str = ""        # data resetu licznika
        
        self._load_entry_prices()
        
        logger.info(f"AggressiveBot v7.0 ADAPTIVE | Strategy: DIP BUYING + REGIME DETECTION + CLAUDE AI")
        logger.info(f"Regime Detection: {'ON' if cfg.regime_enabled else 'OFF'}")
        logger.info(f"PDT Fix: GTC orders (stop-lossy działają przez noc!)")

    def _load_entry_prices(self):
        for sym in self.cfg.symbols:
            entry = self.db.get_entry_price(sym)
            if entry:
                self.entry_price[sym] = entry

    def fetch_historical_low(self, sym: str) -> Optional[float]:
        try:
            # Pobierz więcej dni żeby mieć pewność danych (weekend = brak sesji)
            end_dt = utc_now()
            start_dt = end_dt - timedelta(days=7)  # ostatnie 7 dni, znajdziemy ostatnią sesję
            
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start_dt,
                end=end_dt,
                feed=self.cfg.data_feed
            )
            
            df = self.data.get_stock_bars(req).df
            
            if df is None or len(df) == 0:
                logger.warning(f"{sym}: Brak danych historycznych")
                return None
            
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                df = df[df["symbol"] == sym].set_index("timestamp")
            
            df = df.sort_index()
            
            # Potrzebujemy minimum 2 bary (dziś + wczoraj)
            if len(df) < 2:
                logger.warning(f"{sym}: Za mało barów ({len(df)}) - używam dostępnego minimum")
                return float(df['low'].iloc[-1])
            
            # iloc[-1] = dzisiaj (lub ostatnia sesja), iloc[-2] = wczoraj
            yesterday_low = float(df['low'].iloc[-2])
            return yesterday_low
            
        except IndexError as e:
            logger.warning(f"{sym}: IndexError w fetch_historical_low - za mało danych")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {sym}: {e}")
            return None

    def fetch_bars(self, sym: str) -> pd.DataFrame:
        try:
            tf_min = self.cfg.timeframe_minutes
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame(tf_min, TimeFrameUnit.Minute),
                start=utc_now() - timedelta(minutes=300 * tf_min),
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
            # Alpaca wymaga DAY dla fractional shares
            # PDT rozwiązujemy przez całkowite unikanie day-tradingu (Regime Detector)
            qty = round(qty, 4) if self.cfg.fractional_enabled else int(qty)
            tif = TimeInForce.DAY if self.cfg.fractional_enabled else TimeInForce.GTC
            order = self.trading.submit_order(
                MarketOrderRequest(
                    symbol=sym,
                    qty=qty,
                    side=side,
                    time_in_force=tif
                ))
            self.db.upsert_order(order)
            self.last_trade[sym] = time.time()
            logger.info(f"{sym}: {side} {qty} shares [{tif}]")
            return order
        except Exception as e:
            logger.error(f"{sym}: Order failed | {e}")
            return None

    def can_trade_today(self, symbol):
        today = utc_now().strftime("%Y-%m-%d")
        return self.db.get_trades_today(symbol, today) < self.cfg.max_trades_per_day

    def run_once(self):
        # ================================================================
        # v7.0: REGIME GATE - sprawdź rynek PRZED wszystkim
        # ================================================================
        regime_ok = True
        if self.regime:
            regime_ok = self.regime.should_trade(self.cfg.regime_spy_trend_threshold)
            if not regime_ok:
                logger.warning(f"🛑 REGIME GATE: Brak nowych BUY | {self.regime.get_status()}")
                self.db.log_regime("BEAR_OR_VOLATILE", False, self.regime.get_status())
                self.sheets.send_regime("BEAR_OR_VOLATILE", False, self.regime.get_status())
                # NIE wychodzimy! Sprawdzamy czy są otwarte pozycje do zamknięcia
            else:
                self.db.log_regime("BULL_OR_SIDEWAYS", True, self.regime.get_status())
                self.sheets.send_regime("BULL_OR_SIDEWAYS", True, self.regime.get_status())
        
        # v7.0: ML co 6h
        self.ml.maybe_analyze()

        # v7.0: Publikuj sygnał NYSE raz na cykl (nie per symbol)
        if self.market_analyzer and self.bot_sync:
            _mode = self.market_analyzer.analyze()
            _spy  = self.market_analyzer._last_spy_data if hasattr(self.market_analyzer, '_last_spy_data') else {}
            self.bot_sync.publish(
                _mode,
                self.market_analyzer.get_buy_signal(),
                self.market_analyzer._reasoning,
                _spy)
        
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
                self._process_symbol(sym, equity, cash, positions, allow_buy=regime_ok)
            except Exception as e:
                logger.error(f"{sym}: Error | {e}")
                traceback.print_exc()
        
        ts = utc_now()
        for p in self.trading.get_all_positions():
            self.db.insert_position(ts, p)

    def _process_symbol(self, sym, equity, cash, positions, allow_buy=True):
        df = self.fetch_bars(sym)
        if not self.validate_bars(df):
            return
        
        for ts, row in df.tail(3).iterrows():
            ts_dt = ts.to_pydatetime()
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            self.db.upsert_bar(sym, ts_dt, self.cfg.timeframe_str,
                               row["open"], row["high"], row["low"], row["close"], row["volume"])
        
        price = float(df["close"].iloc[-1])
        atr_val = calculate_atr(df, 14)
        
        pos = positions.get(sym)
        pos_qty = float(pos.qty) if pos else 0.0
        
        entry = None
        if pos and hasattr(pos, "avg_entry_price") and pos.avg_entry_price:
            try:
                entry = float(pos.avg_entry_price)
                self.entry_price[sym] = entry
            except Exception:
                pass
        if entry is None and pos_qty == 0:
            entry = self.entry_price.get(sym)
        
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
        
        # === DIP BUYING STRATEGY ===
        action = "HOLD"
        reason = "none"
        yday_low = None
        rebound_pct = None
        tp_hit = False
        sl_hit = False
        
        if pos_qty == 0:
            yday_low = self.fetch_historical_low(sym)
            
            if yday_low is None:
                return
            
            dip_threshold = yday_low * (1 + self.cfg.dip_tolerance_pct / 100.0)
            is_at_dip = price <= dip_threshold
            
            recent_bars = df.tail(self.cfg.rebound_bars)
            avg_recent = recent_bars['close'].mean()
            rebound_pct = ((price - avg_recent) / avg_recent) * 100
            is_rebounding = rebound_pct >= self.cfg.rebound_threshold_pct
            
            entry_signal = is_at_dip and is_rebounding and sentiment_ok
            cooled = (time.time() - self.last_trade[sym]) >= self.cfg.cooldown_sec
            can_trade = self.can_trade_today(sym)

            # PDT: nie kupuj symbolu który już dziś sprzedałeś (day-trade violation)
            sold_today = self.db.sold_today(sym)
            if sold_today:
                logger.info(f"{sym} | ⛔ PDT: sprzedany dziś — pomijam BUY do jutra")

            # v7.0: Pobierz ocenę rynku od Claude AI
            market_mode = MarketAnalyzer.MODE_UNKNOWN
            claude_buy_signal = False
            if self.market_analyzer:
                market_mode = self.market_analyzer.analyze()
                claude_buy_signal = self.market_analyzer.get_buy_signal()

            # v7.0: Konsultacja z GPW botem
            partner_ok = True
            if self.bot_sync:
                partner_ok, partner_reason = self.bot_sync.partner_confirms_buy()

            # v7.0: CRASH BUYING — mała pozycja gdy Claude daje buy_signal
            # Działa niezależnie od regime (nawet w BULL może kupować ostrożnie)
            # Warunek: Claude mówi buy_signal=True (RSI<32 + wolumen maleje)
            #          + lokalne odbicie (is_rebounding)
            #          + nie ma już otwartej pozycji (pos_qty == 0, jesteśmy w tym bloku)
            crash_entry = (
                claude_buy_signal
                and is_rebounding
                and sentiment_ok
                and not allow_buy  # tylko gdy regime blokuje normalny BUY
            )

            # v7.0: OPPORTUNISTIC ENTRY z limitem ryzyka
            # Resetuj licznik dzienny jeśli nowy dzień
            today_str = utc_now().strftime("%Y-%m-%d")
            if self._opp_daily_date != today_str:
                self._opp_daily_spent = 0.0
                self._opp_daily_date  = today_str

            opp_budget_ok   = self._opp_daily_spent < self.cfg.opp_daily_limit
            opp_rebound_ok  = rebound_pct >= self.cfg.opp_min_rebound  # > -0.5%

            opportunistic_entry = (
                claude_buy_signal
                and allow_buy           # regime OK
                and not is_at_dip       # nie ma klasycznego dip sygnału
                and opp_rebound_ok      # nie pikuje mocno w dół
                and sentiment_ok
                and opp_budget_ok       # nie przekroczono dziennego limitu
            )

            logger.info(f"{sym} | ${price:.2f} | YdayLow=${yday_low:.2f} Thresh=${dip_threshold:.2f} | "
                         f"Dip={'YES' if is_at_dip else 'NO'} Rebound={rebound_pct:+.2f}% | "
                         f"Sentiment={sentiment_score:+.2f} | ATR=${atr_val:.2f} | "
                         f"Market={market_mode} BuySignal={claude_buy_signal}")

            if not allow_buy and entry_signal and not crash_entry:
                logger.info(f"{sym} | 🛑 BEAR MARKET - pomijam BUY mimo sygnału")

            # Normalny BUY (bull/sideways + klasyczny dip) — cały kapitał (95%)
            if entry_signal and cooled and can_trade and allow_buy and not sold_today:
                qty = (cash * self.cfg.max_pos_pct) / price
                buy_mode = "NORMAL"
                buy_pct = self.cfg.max_pos_pct

            # Oportunistyczny BUY (BULL + Claude buy_signal) — max $300, limit $500/dzień
            # Blokowany gdy partner widzi CRASH lub symbol sprzedany dziś
            elif opportunistic_entry and cooled and can_trade and partner_ok and not sold_today:
                # Kwota: min(opp_max_position, pozostały dzienny budżet, dostępny cash)
                opp_amount = min(
                    self.cfg.opp_max_position,
                    self.cfg.opp_daily_limit - self._opp_daily_spent,
                    cash * 0.95
                )
                qty = opp_amount / price if opp_amount >= 50 else 0
                buy_mode = "OPPORTUNISTIC"
                buy_pct  = opp_amount / cash if cash > 0 else 0
                if qty == 0:
                    logger.info(f"{sym} | ⏭️ OPPORTUNISTIC pominięty — za mało cash (${opp_amount:.0f} < $50)")
                else:
                    logger.info(
                        f"{sym} | 🔔 OPPORTUNISTIC BUY ${opp_amount:.0f} "
                        f"(dzienny limit: ${self._opp_daily_spent:.0f}+${opp_amount:.0f}/${self.cfg.opp_daily_limit:.0f}) "
                        f"| 🤝 {self.bot_sync.other if self.bot_sync else 'solo'}: {partner_reason}")

            # Crash BUY (BEAR regime + Claude buy_signal) — 30%
            elif crash_entry and cooled and can_trade and not sold_today:
                qty = (cash * self.cfg.crash_buy_pct) / price
                buy_mode = "CRASH_BUY"
                buy_pct = self.cfg.crash_buy_pct
                logger.info(f"{sym} | 🎯 CRASH BUY — regime BEAR ale Claude: RSI oversold + wolumen maleje")
            else:
                qty = 0
                buy_mode = None
                buy_pct = 0

            if qty is not None and qty > 0:
                if not self.cfg.fractional_enabled:
                    qty = int(qty)

                if qty > 0:
                    self.last_trade[sym] = time.time()
                    action = "BUY"
                    reason = f"crash_buy_{self.cfg.crash_buy_pct*100:.0f}pct" if buy_mode == "CRASH_BUY" else f"dip_entry_yday_low+{self.cfg.dip_tolerance_pct}%"

                    if buy_mode in ("OPPORTUNISTIC", "CRASH_BUY"):
                        tp_log = price * (1 + self.cfg.opp_tp_pct / 100)
                        sl_log = price * (1 - self.cfg.opp_sl_pct / 100)
                    else:
                        tp_log = price + atr_val * self.cfg.atr_tp_multiplier
                        sl_log = price - atr_val * self.cfg.atr_sl_multiplier
                    logger.info(
                        f"{sym} | 🎯 BUY {qty:.4f} @${price:.2f} | {buy_mode} ({buy_pct*100:.0f}%) | "
                        f"YdayLow=${yday_low:.2f} Rebound={rebound_pct:+.2f}% | "
                        f"TP=${tp_log:.2f} (+{self.cfg.opp_tp_pct if buy_mode in ('OPPORTUNISTIC','CRASH_BUY') else atr_val*self.cfg.atr_tp_multiplier:.1f}{'%' if buy_mode in ('OPPORTUNISTIC','CRASH_BUY') else '$'}) "
                        f"SL=${sl_log:.2f} (-{self.cfg.opp_sl_pct if buy_mode in ('OPPORTUNISTIC','CRASH_BUY') else atr_val*self.cfg.atr_sl_multiplier:.1f}{'%' if buy_mode in ('OPPORTUNISTIC','CRASH_BUY') else '$'})")

                    order = self.submit_order(sym, OrderSide.BUY, qty)

                    if order:
                        self.entry_price[sym] = price
                        entry = price
                        if buy_mode in ("OPPORTUNISTIC", "CRASH_BUY"):
                            # Stały % TP/SL dla agresywnych wejść
                            tp_price = entry * (1 + self.cfg.opp_tp_pct / 100)
                            sl_price = entry * (1 - self.cfg.opp_sl_pct / 100)
                        elif atr_val > 0:
                            tp_price = entry + (atr_val * self.cfg.atr_tp_multiplier)
                            sl_price = entry - (atr_val * self.cfg.atr_sl_multiplier)
                        cash -= qty * price
                        # Zaktualizuj dzienny licznik OPPORTUNISTIC
                        if buy_mode == "OPPORTUNISTIC":
                            self._opp_daily_spent += qty * price
                            logger.info(f"{sym} | 💰 OPP dzienny wydatek: ${self._opp_daily_spent:.0f}/${self.cfg.opp_daily_limit:.0f}")
                        self.sheets.send_trade(
                            symbol=sym, action="BUY", price=price,
                            entry_price=price, pl_pct=None, reason=reason,
                            atr=atr_val, sentiment=sentiment_score,
                            rebound_pct=rebound_pct, yday_low=yday_low or 0, equity=equity)
        
        else:
            # ----------------------------------------------------------------
            # v7.0: Pobierz ocenę rynku od Claude AI (DIP vs CRASH)
            # ----------------------------------------------------------------
            market_mode = MarketAnalyzer.MODE_UNKNOWN
            if self.market_analyzer:
                market_mode = self.market_analyzer.analyze()

            # ----------------------------------------------------------------
            # v7.0: Trailing stop (aktualizuj gdy cena rośnie)
            # ----------------------------------------------------------------
            if self.market_analyzer and entry:
                effective_sl = self.market_analyzer.update_trailing_sl(
                    sym, price, entry, self.cfg.trailing_stop_pct)
            else:
                effective_sl = sl_price  # fallback: ATR-based SL

            if tp_price and effective_sl:
                logger.info(
                    f"{sym} | pos={pos_qty:.4f} entry=${entry:.2f} "
                    f"TP=${tp_price:.2f} SL=${effective_sl:.2f} (trailing) | "
                    f"price=${price:.2f} | market={market_mode}")

            tp_hit = (tp_price is not None and price >= tp_price)
            cooled = (time.time() - self.last_trade[sym]) >= self.cfg.cooldown_sec

            # ----------------------------------------------------------------
            # SL zależy od trybu rynku:
            # DIP     → trailing SL (szerszy, daje szansę na odbicie)
            # CRASH   → trailing SL + sprzedaj jeśli jesteś na minusie
            # UNKNOWN → trailing SL (zachowaj się jak DIP)
            # ----------------------------------------------------------------
            pnl_now = ((price - entry) / entry * 100) if entry else 0
            sl_hit = (effective_sl is not None and price <= effective_sl)

            if market_mode == MarketAnalyzer.MODE_CRASH and pnl_now < 0:
                # KRACH: sprzedaj wszystkie pozycje na stracie
                if not sl_hit:
                    sl_hit = True
                    logger.warning(
                        f"{sym} | 📉 CRASH MODE — sprzedaję stratną pozycję "
                        f"P/L: {pnl_now:+.2f}%")

            # PDT PROTECTION: Nie sprzedawaj tego samego dnia co kupiłeś
            # Sprawdzamy DB + datę pozycji z Alpaca (na wypadek restartu bota)
            pdt_block = self.db.bought_today(sym)

            # Dodatkowe sprawdzenie przez Alpaca - jeśli DB jest pusta po restarcie
            if not pdt_block and pos:
                try:
                    pos_data = pos.model_dump() if hasattr(pos, 'model_dump') else {}
                    # Alpaca zwraca asset_marginable i inne pola ale nie datę zakupu wprost
                    # Sprawdzamy avg_entry_price - jeśli pozycja z dziś, DB powinna mieć wpis
                    # Fallback: jeśli brak wpisu w DB a pozycja istnieje, sprawdź orders API
                    orders = self.trading.get_orders(filter=dict(symbols=[sym], status='filled', limit=1))
                    if orders:
                        import datetime as _dt
                        order_date = orders[0].filled_at
                        if order_date:
                            today_utc = utc_now().strftime("%Y-%m-%d")
                            order_day = order_date.strftime("%Y-%m-%d") if hasattr(order_date, 'strftime') else str(order_date)[:10]
                            if order_day == today_utc:
                                pdt_block = True
                                logger.info(f"{sym} | ⏳ PDT HOLD (Alpaca orders) - kupione dziś o {order_date}")
                except Exception:
                    pass

            if pdt_block and (tp_hit or sl_hit):
                logger.info(
                    f"{sym} | ⏳ PDT HOLD - kupione dziś, czekam do jutra | "
                    f"P/L: {pnl_now:+.2f}% | market={market_mode}")

            if (tp_hit or sl_hit) and cooled and not pdt_block:
                action = "SELL"
                reason = "take_profit" if tp_hit else "stop_loss"
                if market_mode == MarketAnalyzer.MODE_CRASH and pnl_now < 0:
                    reason = "crash_exit"

                qty = pos_qty if self.cfg.fractional_enabled else int(pos_qty)
                pnl_pct = pnl_now

                logger.info(
                    f"{sym} | 💥 SELL {qty:.4f} @${price:.2f} | "
                    f"{reason} | P/L: {pnl_pct:+.2f}% | market={market_mode}")

                order = self.submit_order(sym, OrderSide.SELL, qty)

                if order:
                    self.entry_price.pop(sym, None)
                    if self.market_analyzer:
                        self.market_analyzer.reset_trailing_sl(sym)
                    self.sheets.send_trade(
                        symbol=sym, action="SELL", price=price,
                        entry_price=entry, pl_pct=pnl_pct, reason=reason,
                        atr=atr_val, sentiment=sentiment_score,
                        rebound_pct=rebound_pct or 0, yday_low=yday_low or 0, equity=equity)
        
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
        # v7.0: Wyślij summary do Google Sheets
        self.sheets.send_summary(today, start_equity, end_equity, total, wins)
        return report

    def run(self):
        logger.info("="*60)
        logger.info("ADAPTIVE BOT v7.0 - DIP BUYING + REGIME DETECTION + CLAUDE AI")
        logger.info("="*60)
        logger.info(f"Strategy: Buy at yesterday's low + {self.cfg.dip_tolerance_pct}% when rebounding")
        logger.info(f"Regime Detection: {'ENABLED' if self.cfg.regime_enabled else 'DISABLED'}")
        logger.info(f"PDT Fix: GTC orders aktywne")
        logger.info(f"ML Analysis: co 6h w logach")
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
    print("ADAPTIVE TRADING BOT v7.0")
    print("Strategy: Dip Buying + Regime Detection + ML")
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
