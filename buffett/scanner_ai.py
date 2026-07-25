"""
Weekly scanner for Buffett Monitor.
Orchestrates fetching, scoring, and change logging for the entire universe.
"""

import logging
import sys
from datetime import date
from typing import Dict, List, Tuple
from pathlib import Path

from buffett.fetchers import fetch_fundamentals
from buffett.scorer import compute_intrinsic_value, compute_quant_score, decide_signal, calculate_graham_number
from buffett.moat_llm import judge_moat
from buffett.change_log import diff_previous
from data.init_db import init_database  # Fixed import path
from alerts.alert_system import price_alert, signal_alert, fundamental_alert
import yfinance as yf
from ml.signal_enhancer import SignalEnhancer


import yfinance as yf
from ml.signal_enhancer import SignalEnhancer

logger = logging.getLogger(__name__)


# Initialize ML signal enhancer (once per scan)
try:
    enhancer = SignalEnhancer()
    if not enhancer.is_ready:
        enhancer = None
        logger.info("ML signal enhancer not ready. Using rule-based signals only.")
    else:
        logger.info("ML signal enhancer initialized successfully.")
except Exception as e:
    enhancer = None
    logger.warning(f"Failed to initialize ML signal enhancer: {e}")

# Initialize ML signal enhancer (once per scan)
try:
    enhancer = SignalEnhancer()
    if not enhancer.is_ready:
        enhancer = None
        logger.info("ML signal enhancer not ready. Using rule-based signals only.")
    else:
        logger.info("ML signal enhancer initialized successfully.")
except Exception as e:
    enhancer = None
    logger.warning(f"Failed to initialize ML signal enhancer: {e}")

def run_weekly_scan(db_path: str = "data/buffett.db") -> Dict:
    """
    Run a weekly scan over the entire universe.

    Args:
        db_path: Path to SQLite database

    Returns:
        Summary dictionary with scan results
    """
    logger.info("Starting weekly scan...")

    # Ensure database is initialized
    init_database(db_path)

    # Get universe of tickers
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        # Using custom AI stock list for targeted scan
        tickers = ['BE', 'CRWV', 'INTC', 'LITE', 'CORZ', 'IREN', 'APLD', 'SNDK', 'CIFR', 'EQT', 'COHR', 'SOI', 'TSEM', 'RIOT', 'KRC', 'HUT', 'WYFI', 'PSIX', 'BTDR', 'CLSK', 'BITF', 'LBRT', 'INFY', 'PUMP', 'BW', 'INTC', 'CRWV', 'CORZ', 'IREN', 'NVDA', 'VST', 'GDX', 'APLD', 'GLXY', 'AVGO', 'TSM', 'RIOT', 'EQT', 'MU', 'SOI', 'TSEM', 'HUT', 'COHR', 'BTDR', 'SNDK', 'STX', 'INTC', 'CRWV', 'SNPS', 'COHR', 'NOK', 'YNDX']
    finally:
        conn.close()

    logger.info(f"Scanning {len(tickers)} tickers...")

    # Results tracking
    results = {
        "scan_date": date.today().isoformat(),
        "total_tickers": len(tickers),
        "successful": 0,
        "failed": 0,
        "buy_signals": 0,
        "hold_signals": 0,
        "sell_signals": 0,
        "avoid_signals": 0,
        "errors": [],
        "summary_by_signal": {}
    }

    # Process each ticker
    for i, ticker in enumerate(tickers, 1):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(tickers)}")

        try:
            # Fetch fundamentals
            fundamentals = fetch_fundamentals(ticker)

            if fundamentals is None:
                results["failed"] += 1
                results["errors"].append(f"{ticker}: Failed to fetch fundamentals")
                continue

            # Calculate intrinsic value
            fcf = fundamentals.get("eps_ttm", 0) * fundamentals.get("shares_outstanding", 0)  # Simplified
            if fcf > 0:
                iv = compute_intrinsic_value(
                    fcf=fcf,
                    growth_rate=0.05,  # TODO: get from fundamentals or config
                    discount_rate=0.10
                )
                fundamentals["intrinsic_value"] = iv

                price = fundamentals.get("price", 0)
                if iv > 0:
                    fundamentals["margin_of_safety"] = (iv - price) / iv if price > 0 else 0
                    fundamentals["implied_return_pct"] = (iv / price - 1) if price > 0 else 0

            # Calculate Graham number
            eps = fundamentals.get("eps_ttm", 0)
            bvps = fundamentals.get("book_value_per_share", 0)
            if eps > 0 and bvps > 0:
                fundamentals["graham_number"] = calculate_graham_number(eps, bvps)

            # Quantitative score
            quant_score, passed_criteria = compute_quant_score(fundamentals)
            fundamentals["quant_score"] = quant_score

            # Qualitative moat judgment (Pillars 1&2)
            moat_judgment = judge_moat(ticker, fundamentals)
            fundamentals.update(moat_judgment)  # Add pillar1, pillar2, moat_strength, etc.

            # Decide signal
            signal = decide_signal(
                quant_score=quant_score,
                moat_strength=moat_judgment.get("moat_strength"),
                fundamentals_flag=fundamentals.get("fundamentals_flag", "NORMAL"),
                price=fundamentals.get("price", 0),
                intrinsic_value=fundamentals.get("intrinsic_value", 0)
            )
            rule_based_signal = signal
            rule_based_confidence = 0.8  # placeholder confidence for rule-based signal
            fundamentals["signal_reason"] = _generate_signal_reason(
                quant_score, moat_judgment.get("moat_strength"),
                fundamentals.get("margin_of_safety", 0), fundamentals.get("fundamentals_flag")
            )
            # Enhance the signal using ML if enhancer is ready
            if enhancer is not None:
                try:
                    # Get historical price data for technical indicators
                    price_data = yf.download(ticker, period="100d", progress=False)
                    if not price_data.empty:
                        # Use the enhancer to get the enhanced signal
                        enhanced_signal, confidence = enhancer.enhance_signal(
                            ticker=ticker,
                            price_df=price_data,
                            fundamentals=fundamentals,
                            rule_based_signal=rule_based_signal,
                            rule_based_confidence=rule_based_confidence
                        )
                        # Use the enhanced signal
                        signal = enhanced_signal
                        logger.debug(f"{ticker}: Enhanced signal {enhanced_signal} (conf={confidence:.2f}), rule-based {rule_based_signal}")
                    else:
                        logger.warning(f"{ticker}: No price data available for ML enhancement")
                        signal = rule_based_signal
                except Exception as e:
                    logger.warning(f"{ticker}: ML enhancement failed: {e}")
                    signal = rule_based_signal
            else:
                signal = rule_based_signal
            rule_based_signal = signal
            rule_based_confidence = 0.8  # placeholder
            # Store additional info for debugging/training
            fundamentals["rule_based_signal"] = rule_based_signal
            fundamentals["ml_confidence"] = 0.0 if "confidence" not in locals() else confidence
            # Enhance the signal using ML if enhancer is ready
            if enhancer is not None:
                try:
                    # Get historical price data for technical indicators
                    price_data = yf.download(ticker, period="100d", progress=False)
                    if not price_data.empty:
                        # Use the enhancer to get the enhanced signal
                        enhanced_signal, confidence = enhancer.enhance_signal(
                            ticker=ticker,
                            price_df=price_data,
                            fundamentals=fundamentals,
                            rule_based_signal=rule_based_signal,
                            rule_based_confidence=rule_based_confidence
                        )
                        # Use the enhanced signal
                        signal = enhanced_signal
                        logger.debug(f"{ticker}: Enhanced signal {enhanced_signal} (conf={confidence:.2f}), rule-based {rule_based_signal}")
                    else:
                        logger.warning(f"{ticker}: No price data available for ML enhancement")
                        signal = rule_based_signal
                except Exception as e:
                    logger.warning(f"{ticker}: ML enhancement failed: {e}")
                    signal = rule_based_signal
            else:
                signal = rule_based_signal
            fundamentals["signal"] = signal
            fundamentals["enhancement_used"] = "signal" in locals() and signal != rule_based_signal

            # Determine which Buffett pillars passed (for compatibility with existing schema)
            # Determine which Buffett pillars passed (for compatibility with existing schema)
            fundamentals["pillar1_understandable"] = moat_judgment.get("pillar1") in ["STRONG", "WEAK"]  # Simplified
            fundamentals["pillar2_longterm"] = moat_judgment.get("pillar2") in ["STRONG", "WEAK"]
            fundamentals["pillar3_leadership"] = True  # Placeholder - would need management data
            fundamentals["pillar4_undervalued"] = fundamentals.get("margin_of_safety", 0) > 0.3
            fundamentals["pillars_passed"] = sum([
                fundamentals["pillar1_understandable"],
                fundamentals["pillar2_longterm"],
                fundamentals["pillar3_leadership"],
                fundamentals["pillar4_undervalued"]
            ])

            # Moat fields for compatibility
            fundamentals["moat_strength"] = moat_judgment.get("moat_strength", "UNKNOWN")
            fundamentals["moat_rationale"] = moat_judgment.get("moat_rationale", "")
            fundamentals["mgmt_quality"] = moat_judgment.get("mgmt_quality", "UNKNOWN")
            fundamentals["mgmt_rationale"] = moat_judgment.get("mgmt_rationale", "")

            # Save to database
            _save_snapshot(ticker, fundamentals, db_path)
            _save_scores(ticker, fundamentals, db_path)

            # Log changes
            changes_logged = diff_previous(ticker, fundamentals, db_path)

            # Update counters
            results["successful"] += 1
            signal_count_key = f"{signal.lower()}_signals"
            if signal_count_key in results:
                results[signal_count_key] += 1

            # Log significant signals
            if signal in ["BUY", "SELL"]:
                logger.info(f"{ticker}: {signal} signal (QS: {quant_score:.1f}, Moat: {moat_judgment.get('moat_strength')})")

                # Send alerts for significant signal changes
                # We would need to compare with previous signal to send alert
                # For now, we'll skip alert generation in scanner to avoid duplicates
                # Alerts will be generated by a separate process that checks for changes

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            results["failed"] += 1
            results["errors"].append(f"{ticker}: {str(e)}")

    # Final summary
    logger.info(f"Scan complete. Successful: {results['successful']}, Failed: {results['failed']}")
    logger.info(f"Signals - BUY: {results['buy_signals']}, HOLD: {results['hold_signals']}, "
                f"SELL: {results['sell_signals']}, AVOID: {results['avoid_signals']}")

    return results


def _save_snapshot(ticker: str, fundamentals: Dict, db_path: str):
    """Save fundamentals snapshot to database."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        # Get the columns from buffett_fundamentals table
        cursor = conn.execute("PRAGMA table_info(buffett_fundamentals)")
        table_columns = [row[1] for row in cursor.fetchall()]  # column names

        # Remove the auto-increment id column if present
        if "id" in table_columns:
            table_columns.remove("id")

        # Prepare values for each column in the table (using fundamentals dict, defaulting to None)
        values = []
        for col in table_columns:
            values.append(fundamentals.get(col))  # Returns None if key not present

        placeholders = ", ".join(["?"] * len(values))
        columns_str = ", ".join(table_columns)

        sql = f"""
            INSERT OR REPLACE INTO buffett_fundamentals
            ({columns_str})
            VALUES ({placeholders})
        """

        conn.execute(sql, values)
        conn.commit()
    finally:
        conn.close()


def _save_scores(ticker: str, fundamentals: dict, db_path: str):
    """Save Buffett scores to database."""
    import sqlite3
    from datetime import date

    conn = sqlite3.connect(db_path)
    try:
        # Get the columns from buffett_scores table
        cursor = conn.execute("PRAGMA table_info(buffett_scores)")
        table_columns = [row[1] for row in cursor.fetchall()]  # column names

        # Remove the auto-increment id column if present
        if "id" in table_columns:
            table_columns.remove("id")

        # Prepare values for each column in the table (using fundamentals dict, defaulting to None)
        values = []
        for col in table_columns:
            if col == "ticker":
                values.append(ticker)
            elif col == "snapshot_date":
                values.append(date.today().isoformat())
            else:
                values.append(fundamentals.get(col))  # Returns None if key not present

        placeholders = ", ".join(["?"] * len(values))
        columns_str = ", ".join(table_columns)

        sql = f"""
            INSERT OR REPLACE INTO buffett_scores
            ({columns_str})
            VALUES ({placeholders})
        """

        conn.execute(sql, values)
        conn.commit()
    except Exception as e:
        print(f"Error saving scores for {ticker}: {e}")
        raise
    finally:
        conn.close()




def _record_signal_outcome(ticker: str, signal_date: str, rule_based_signal: str,
                       ml_signal: str, ml_confidence: float, final_signal: str,
                       db_path: str):
    """Record signal outcomes for ML training."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            INSERT OR REPLACE INTO ml_signal_outcomes
            (ticker, signal_date, rule_based_signal, ml_signal, ml_confidence, final_signal)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker, signal_date, rule_based_signal, ml_signal, ml_confidence, final_signal))
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to record signal outcome for {ticker}: {e}")
    finally:
        conn.close()
def _generate_signal_reason(quant_score: float, moat_strength: str,
                           margin_of_safety: float, fundamentals_flag: str) -> str:
    """Generate human-readable reason for the signal."""
    reasons = []

    if fundamentals_flag in ["DATA_SUSPECT", "DELISTED"]:
        reasons.append("Poor data quality")
    elif fundamentals_flag == "LOSS_MAKING":
        reasons.append("Company reporting losses")

    if quant_score >= 80:
        reasons.append("Strong financials")
    elif quant_score >= 60:
        reasons.append("Acceptable financials")
    else:
        reasons.append("Weak financials")

    if moat_strength == "STRONG":
        reasons.append("Strong moat")
    elif moat_strength == "WEAK":
        reasons.append("Weak moat")
    else:
        reasons.append("Average/unknown moat")

    if margin_of_safety >= 0.3:
        reasons.append("Significant margin of safety (>30%)")
    elif margin_of_safety > 0:
        reasons.append("Some margin of safety")
    else:
        reasons.append("No margin of safety (overvalued)")

    return "; ".join(reasons)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the scan
    summary = run_weekly_scan()
    print("\n=== SCAN SUMMARY ===")
    for key, value in summary.items():
        if key != "errors":
            print(f"{key}: {value}")
    if summary["errors"]:
        print(f"\nErrors ({len(summary['errors'])}):")
        for error in summary["errors"][:5]:  # Show first 5 errors
            print(f"  - {error}")