# optimizer.py — TRADING BOT OPTIMIZER v1.0
# Runs alongside bot_ultimate.py as a SEPARATE process
# Analyzes trading history and adjusts parameters
#
# ============================================================================
# HOW IT WORKS:
# ============================================================================
# 1. Reads bot's SQLite database (signals, eod_reports)
# 2. Analyzes which parameter combinations led to wins/losses
# 3. Writes optimized parameters to optimizer_params.json
# 4. Bot reads this file on each cycle and applies overrides
#
# RUNS: Once per day at market close (or on-demand)
# SAFETY: If optimizer fails, bot uses default env variables
# ============================================================================
#
# Usage:
#   python optimizer.py                  # Run full analysis
#   python optimizer.py --report         # Show report only (no changes)
#   python optimizer.py --reset          # Reset to defaults
#
# Railway: Add as separate service or run via cron
# ============================================================================

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | OPTIMIZER | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('optimizer.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================
DB_PATH = os.getenv("DB_PATH", "bot_ultimate.db")
PARAMS_FILE = "optimizer_params.json"
MIN_TRADES_FOR_OPTIMIZATION = 10  # Need at least this many trades to optimize

# Parameter bounds — optimizer won't go outside these
BOUNDS = {
    "rsi_buy":            (20, 40),
    "rsi_sell":           (45, 70),
    "entry_bps":          (10, 40),
    "atr_sl_multiplier":  (1.0, 3.0),
    "atr_tp_multiplier":  (1.5, 5.0),
    "risk_per_trade_pct": (0.5, 3.0),
    "vwap_window":        (15, 60),
    "bb_period":          (15, 30),
    "bb_std":             (1.5, 3.0),
    "ema_trend_period":   (30, 100),
}


@dataclass
class TradeRecord:
    """Single completed trade (buy→sell pair)."""
    symbol: str
    buy_time: str
    sell_time: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    rsi_at_entry: float
    vwap_at_entry: float
    atr_at_entry: float
    ema50_at_entry: float
    bb_lower_at_entry: float
    trend_ok: bool
    bb_ok: bool
    reason_exit: str
    sentiment: float


@dataclass
class AnalysisResult:
    """Results of parameter analysis."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy: float
    max_consecutive_losses: int
    best_rsi_range: Tuple[float, float]
    recommended_params: Dict


# ============================================================================
# DATABASE READER
# ============================================================================
def load_trades(db_path: str, days_back: int = 30) -> List[TradeRecord]:
    """Load completed trades from bot's database."""
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()

    # Get all SELL signals (completed trades) with their data
    trades = []
    try:
        rows = conn.execute("""
            SELECT symbol, ts_utc, entry_price, price, rsi, vwap, 
                   atr, ema50, bb_lower, trend_ok, bb_ok,
                   reason, sentiment
            FROM signals 
            WHERE action='SELL' 
              AND entry_price IS NOT NULL 
              AND price IS NOT NULL
              AND ts_utc > ?
            ORDER BY ts_utc ASC
        """, (cutoff,)).fetchall()

        for row in rows:
            entry_p = float(row['entry_price'])
            exit_p = float(row['price'])
            if entry_p <= 0:
                continue

            pnl_pct = ((exit_p - entry_p) / entry_p) * 100

            trades.append(TradeRecord(
                symbol=row['symbol'],
                buy_time="",  # Could look up from BUY signals if needed
                sell_time=row['ts_utc'],
                entry_price=entry_p,
                exit_price=exit_p,
                pnl_pct=pnl_pct,
                rsi_at_entry=float(row['rsi']) if row['rsi'] else 0,
                vwap_at_entry=float(row['vwap']) if row['vwap'] else 0,
                atr_at_entry=float(row['atr']) if row['atr'] else 0,
                ema50_at_entry=float(row['ema50']) if row['ema50'] else 0,
                bb_lower_at_entry=float(row['bb_lower']) if row['bb_lower'] else 0,
                trend_ok=bool(row['trend_ok']) if row['trend_ok'] is not None else True,
                bb_ok=bool(row['bb_ok']) if row['bb_ok'] is not None else True,
                reason_exit=row['reason'] or "",
                sentiment=float(row['sentiment']) if row['sentiment'] else 0,
            ))

        logger.info(f"Loaded {len(trades)} completed trades from last {days_back} days")

    except Exception as e:
        logger.error(f"Failed to load trades: {e}")

    conn.close()
    return trades


# ============================================================================
# ANALYSIS ENGINE
# ============================================================================
def analyze_trades(trades: List[TradeRecord]) -> AnalysisResult:
    """Analyze trade history and find optimal parameters."""
    if not trades:
        return _empty_result()

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t.pnl_pct) for t in losses]) if losses else 0

    # Profit factor = gross wins / gross losses
    gross_wins = sum(t.pnl_pct for t in wins) if wins else 0
    gross_losses = sum(abs(t.pnl_pct) for t in losses) if losses else 1
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 0

    # Expectancy = avg_win * win_rate - avg_loss * loss_rate
    loss_rate = len(losses) / len(trades) * 100 if trades else 0
    expectancy = (avg_win * win_rate / 100) - (avg_loss * loss_rate / 100)

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for t in trades:
        if t.pnl_pct <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    # === PARAMETER OPTIMIZATION ===
    recommended = {}

    # 1. Optimal RSI buy level
    if len(trades) >= MIN_TRADES_FOR_OPTIMIZATION:
        recommended.update(_optimize_rsi(trades))
        recommended.update(_optimize_atr_multipliers(trades))
        recommended.update(_optimize_risk_per_trade(trades))
        recommended.update(_analyze_filters(trades))

    # Find best RSI range from winning trades
    winning_rsi = [t.rsi_at_entry for t in wins if t.rsi_at_entry > 0]
    if winning_rsi:
        best_rsi = (float(np.percentile(winning_rsi, 10)), float(np.percentile(winning_rsi, 90)))
    else:
        best_rsi = (25.0, 35.0)

    return AnalysisResult(
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        max_consecutive_losses=max_consec,
        best_rsi_range=best_rsi,
        recommended_params=recommended,
    )


def _optimize_rsi(trades: List[TradeRecord]) -> Dict:
    """Find optimal RSI buy threshold by testing ranges."""
    best_score = -999
    best_rsi = 30

    for rsi_thresh in range(20, 41, 2):
        # Simulate: which trades would have been taken at this RSI?
        filtered = [t for t in trades if t.rsi_at_entry <= rsi_thresh and t.rsi_at_entry > 0]
        if len(filtered) < 3:
            continue

        wins = [t for t in filtered if t.pnl_pct > 0]
        wr = len(wins) / len(filtered) if filtered else 0
        avg_pnl = np.mean([t.pnl_pct for t in filtered]) if filtered else 0

        # Score = win_rate * avg_pnl * sqrt(n_trades) — rewards consistency + volume
        score = wr * avg_pnl * np.sqrt(len(filtered))

        if score > best_score:
            best_score = score
            best_rsi = rsi_thresh

    return {"rsi_buy": _clamp(float(best_rsi), *BOUNDS["rsi_buy"])}


def _optimize_atr_multipliers(trades: List[TradeRecord]) -> Dict:
    """Analyze stop-loss and take-profit outcomes to tune ATR multipliers."""
    params = {}

    # Analyze stop-loss trades
    sl_trades = [t for t in trades if "stop_loss" in t.reason_exit]
    tp_trades = [t for t in trades if "take_profit" in t.reason_exit]
    other_exits = [t for t in trades if "technical_exit" in t.reason_exit]

    # If too many stop losses, widen the stop
    if len(trades) > 0:
        sl_rate = len(sl_trades) / len(trades)
        tp_rate = len(tp_trades) / len(trades)

        # Load current multipliers
        current_sl_mult = float(os.getenv("ATR_SL_MULTIPLIER", "1.5"))
        current_tp_mult = float(os.getenv("ATR_TP_MULTIPLIER", "3.0"))

        # If > 40% of exits are stop losses, widen SL by 0.25
        if sl_rate > 0.40:
            new_sl = current_sl_mult + 0.25
            params["atr_sl_multiplier"] = _clamp(new_sl, *BOUNDS["atr_sl_multiplier"])
            logger.info(f"SL hit rate {sl_rate:.0%} > 40% → widening SL to {params['atr_sl_multiplier']}x ATR")

        # If < 15% of exits are stop losses, can tighten
        elif sl_rate < 0.15 and len(trades) >= 10:
            new_sl = current_sl_mult - 0.15
            params["atr_sl_multiplier"] = _clamp(new_sl, *BOUNDS["atr_sl_multiplier"])
            logger.info(f"SL hit rate {sl_rate:.0%} < 15% → tightening SL to {params['atr_sl_multiplier']}x ATR")

        # If TP is rarely hit, lower it
        if tp_rate < 0.10 and len(trades) >= 10:
            new_tp = current_tp_mult - 0.25
            params["atr_tp_multiplier"] = _clamp(new_tp, *BOUNDS["atr_tp_multiplier"])
            logger.info(f"TP hit rate {tp_rate:.0%} < 10% → lowering TP to {params['atr_tp_multiplier']}x ATR")

        # If TP hits frequently and avg TP trade is big, raise it
        elif tp_rate > 0.30 and tp_trades:
            avg_tp_gain = np.mean([t.pnl_pct for t in tp_trades])
            if avg_tp_gain > 0.5:
                new_tp = current_tp_mult + 0.25
                params["atr_tp_multiplier"] = _clamp(new_tp, *BOUNDS["atr_tp_multiplier"])
                logger.info(f"TP hit rate {tp_rate:.0%} > 30% with avg +{avg_tp_gain:.2f}% → raising TP to {params['atr_tp_multiplier']}x ATR")

    return params


def _optimize_risk_per_trade(trades: List[TradeRecord]) -> Dict:
    """Adjust risk per trade based on recent win rate."""
    if len(trades) < 10:
        return {}

    recent = trades[-20:]  # Last 20 trades
    wins = [t for t in recent if t.pnl_pct > 0]
    wr = len(wins) / len(recent)

    current_risk = float(os.getenv("RISK_PER_TRADE_PCT", "1.5"))

    # High win rate → can increase risk slightly
    if wr > 0.60 and len(recent) >= 15:
        new_risk = min(current_risk + 0.25, BOUNDS["risk_per_trade_pct"][1])
        logger.info(f"Win rate {wr:.0%} > 60% → increasing risk to {new_risk}%")
        return {"risk_per_trade_pct": new_risk}

    # Low win rate → decrease risk
    if wr < 0.35:
        new_risk = max(current_risk - 0.25, BOUNDS["risk_per_trade_pct"][0])
        logger.info(f"Win rate {wr:.0%} < 35% → decreasing risk to {new_risk}%")
        return {"risk_per_trade_pct": new_risk}

    return {}


def _analyze_filters(trades: List[TradeRecord]) -> Dict:
    """Check if trend and BB filters are helping."""
    params = {}

    # Compare trades WITH trend filter vs WITHOUT
    trend_yes = [t for t in trades if t.trend_ok]
    trend_no = [t for t in trades if not t.trend_ok]

    if trend_yes and trend_no:
        wr_yes = len([t for t in trend_yes if t.pnl_pct > 0]) / len(trend_yes)
        wr_no = len([t for t in trend_no if t.pnl_pct > 0]) / len(trend_no)
        logger.info(f"Trend filter: WITH={wr_yes:.0%} win rate ({len(trend_yes)} trades) | WITHOUT={wr_no:.0%} ({len(trend_no)} trades)")

        # If trades WITHOUT trend filter actually do better, consider disabling
        # But keep a high bar — trend filter is a safety net
        if wr_no > wr_yes + 0.15 and len(trend_no) >= 10:
            logger.warning("Trades WITHOUT trend filter have significantly higher win rate — review EMA period")
            # Don't disable, but try adjusting period
            params["ema_trend_period"] = _clamp(40, *BOUNDS["ema_trend_period"])

    # Check BB filter effectiveness
    bb_yes = [t for t in trades if t.bb_ok]
    bb_no = [t for t in trades if not t.bb_ok]

    if bb_yes and bb_no:
        wr_bb_yes = len([t for t in bb_yes if t.pnl_pct > 0]) / len(bb_yes)
        wr_bb_no = len([t for t in bb_no if t.pnl_pct > 0]) / len(bb_no)
        logger.info(f"BB filter: WITH={wr_bb_yes:.0%} ({len(bb_yes)} trades) | WITHOUT={wr_bb_no:.0%} ({len(bb_no)} trades)")

    return params


def _empty_result() -> AnalysisResult:
    return AnalysisResult(
        total_trades=0, winning_trades=0, losing_trades=0,
        win_rate=0, avg_win_pct=0, avg_loss_pct=0,
        profit_factor=0, expectancy=0, max_consecutive_losses=0,
        best_rsi_range=(25, 35), recommended_params={})


def _clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


# ============================================================================
# PARAMETER WRITER
# ============================================================================
def save_params(params: Dict, filepath: str = PARAMS_FILE):
    """Save optimized parameters to JSON file for bot to read."""
    if not params:
        logger.info("No parameter changes recommended")
        return

    # Load existing params to merge
    existing = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                existing = json.load(f)
        except Exception:
            pass

    # Merge — new values override old
    existing.update(params)
    existing["_last_updated"] = datetime.now(timezone.utc).isoformat()
    existing["_version"] = "optimizer_v1.0"

    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info(f"Saved {len(params)} parameter(s) to {filepath}")
    for k, v in params.items():
        if not k.startswith("_"):
            logger.info(f"  {k}: {v}")


def reset_params(filepath: str = PARAMS_FILE):
    """Reset to defaults by removing optimizer file."""
    if os.path.exists(filepath):
        os.remove(filepath)
        logger.info(f"Removed {filepath} — bot will use default env variables")
    else:
        logger.info("No optimizer params file found — already using defaults")


# ============================================================================
# REPORT PRINTER
# ============================================================================
def print_report(result: AnalysisResult):
    """Print human-readable analysis report."""
    print()
    print("=" * 60)
    print("  TRADING BOT OPTIMIZER — ANALYSIS REPORT")
    print("=" * 60)
    print()
    print(f"  Total Trades:          {result.total_trades}")
    print(f"  Winning:               {result.winning_trades} ({result.win_rate:.1f}%)")
    print(f"  Losing:                {result.losing_trades}")
    print(f"  Avg Win:               +{result.avg_win_pct:.2f}%")
    print(f"  Avg Loss:              -{result.avg_loss_pct:.2f}%")
    print(f"  Profit Factor:         {result.profit_factor:.2f}")
    print(f"  Expectancy/Trade:      {result.expectancy:+.3f}%")
    print(f"  Max Consec. Losses:    {result.max_consecutive_losses}")
    print(f"  Best RSI Entry Range:  {result.best_rsi_range[0]:.0f} - {result.best_rsi_range[1]:.0f}")
    print()

    if result.recommended_params:
        print("  RECOMMENDED CHANGES:")
        print("  " + "-" * 40)
        for k, v in result.recommended_params.items():
            print(f"    {k}: {v}")
    else:
        print("  No parameter changes recommended (need more data)")

    print()
    print("=" * 60)

    # Warnings
    if result.win_rate < 40:
        print("  ⚠️  Win rate below 40% — strategy needs attention")
    if result.profit_factor < 1.0:
        print("  ⚠️  Profit factor < 1.0 — bot is losing money")
    if result.max_consecutive_losses >= 5:
        print("  ⚠️  5+ consecutive losses detected — consider pausing")
    if result.expectancy < 0:
        print("  ⚠️  Negative expectancy — each trade loses money on average")

    if result.win_rate >= 50 and result.profit_factor >= 1.5:
        print("  ✅  Strategy looks healthy!")

    print()


# ============================================================================
# MAIN
# ============================================================================
def main():
    logger.info("=" * 60)
    logger.info("OPTIMIZER START")
    logger.info("=" * 60)

    # Parse args
    report_only = "--report" in sys.argv
    reset = "--reset" in sys.argv

    if reset:
        reset_params()
        return

    # Load trades
    trades = load_trades(DB_PATH, days_back=30)

    if not trades:
        logger.warning("No trades found in database — nothing to optimize")
        print("\nNo trades found. Bot needs to make at least 10 trades before optimizer can help.\n")
        return

    # Analyze
    result = analyze_trades(trades)

    # Print report
    print_report(result)

    # Save params (unless report-only mode)
    if not report_only:
        if result.total_trades >= MIN_TRADES_FOR_OPTIMIZATION:
            save_params(result.recommended_params)
        else:
            logger.info(f"Only {result.total_trades} trades — need {MIN_TRADES_FOR_OPTIMIZATION} for optimization")
            print(f"Need {MIN_TRADES_FOR_OPTIMIZATION} trades minimum. Currently: {result.total_trades}")

    logger.info("OPTIMIZER COMPLETE")


if __name__ == "__main__":
    main()
