"""
Weekly scanner for Buffett Monitor.
Orchestrates fetching, scoring, and change logging for the entire universe.
"""

import logging
import sqlite3
import sys
from datetime import date
from typing import Dict, List, Tuple
from pathlib import Path

from buffett.fetchers import fetch_fundamentals
from buffett.scorer import calculate_graham_number, compute_enhanced_score
from buffett.moat_llm import judge_moat
from buffett.sector_stats import compute_sector_stats
from buffett.change_log import diff_previous
from data.init_db import init_database  # Fixed import path
from alerts.alert_system import price_alert, signal_alert, fundamental_alert, alert_manager
import yfinance as yf
from ml.signal_enhancer import SignalEnhancer

logger = logging.getLogger(__name__)


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


def run_weekly_scan(db_path: str = "data/buffett.db", tickers: list[str] | None = None) -> Dict:
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
        if tickers is not None and len(tickers) > 0:
            # Use specific tickers if provided
            # Check which tickers are in the universe
            cursor = conn.execute("SELECT ticker FROM buffett_universe WHERE is_active = 1")
            universe_tickers = set([row[0] for row in cursor.fetchall()])
            unknown_tickers = [t for t in tickers if t not in universe_tickers]
            if unknown_tickers:
                logger.warning(f"The following tickers are not in the universe and will be scanned anyway: {unknown_tickers}")
            # Use the provided tickers (no change to tickers variable)
        else:
            # Use full universe from database
            cursor = conn.execute("SELECT ticker FROM buffett_universe WHERE is_active = 1")
            tickers = [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()

    if tickers is not None and len(tickers) > 0:
        logger.info(f"Scanning {len(tickers)} specified tickers...")
    else:
        logger.info(f"Scanning {len(tickers)} tickers from universe...")
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

    # Compute sector-relative peer thresholds once per scan (not per ticker)
    # so scoring judges "cheap"/"profitable" against comparable peers rather
    # than a single fixed global threshold. Falls back to no sector data
    # (compute_quant_score uses its fixed constants) if this fails.
    try:
        sector_stats_by_sector = compute_sector_stats(db_path, as_of_date=date.today().isoformat())
        logger.info(f"Computed sector-relative stats for {len(sector_stats_by_sector)} sectors")
    except Exception as e:
        sector_stats_by_sector = {}
        logger.warning(f"Failed to compute sector-relative stats: {e}")

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

            # Calculate Graham number
            eps = fundamentals.get("eps_ttm", 0)
            bvps = fundamentals.get("book_value_per_share", 0)
            if eps > 0 and bvps > 0:
                fundamentals["graham_number"] = calculate_graham_number(eps, bvps)

            # Enhanced scoring (AI + classic + news sentiment)
            sector = fundamentals.get("sector", "")
            industry = fundamentals.get("industry", "")
            moat_judgment = judge_moat(ticker, fundamentals, db_path=db_path)
            moat_strength = moat_judgment.get("moat_strength", "UNKNOWN")
            
            quant_score, passed_criteria, signal, scoring_metadata = compute_enhanced_score(
                fundamentals=fundamentals,
                moat_strength=moat_strength,
                sector=sector,
                industry=industry,
                db_path=db_path,
                sector_stats=sector_stats_by_sector.get(sector)
            )
            
            fundamentals["quant_score"] = quant_score
            fundamentals["passed_criteria"] = passed_criteria
            fundamentals["scoring_metadata"] = scoring_metadata
            fundamentals.update(moat_judgment)  # Add pillar1, pillar2, moat_strength, etc.

            # Single source of truth for intrinsic value / margin of safety:
            # whichever valuation path actually drove the signal (AI or classic).
            price = fundamentals.get("price", 0)
            if scoring_metadata.get("ai_valuation"):
                intrinsic_value = scoring_metadata["ai_valuation"]["intrinsic_value_ai"]
                margin_of_safety = scoring_metadata["ai_valuation"]["margin_of_safety_ai"]
            else:
                intrinsic_value = scoring_metadata.get("classic_intrinsic", 0.0)
                margin_of_safety = (intrinsic_value - price) / intrinsic_value if intrinsic_value > 0 and price > 0 else 0.0
            fundamentals["intrinsic_value"] = intrinsic_value
            fundamentals["margin_of_safety"] = margin_of_safety
            fundamentals["implied_return_pct"] = (intrinsic_value / price - 1) if price > 0 and intrinsic_value > 0 else 0.0
            rule_based_signal = signal
            rule_based_confidence = 0.8  # placeholder confidence for rule-based signal
            fundamentals["signal_reason"] = _generate_signal_reason(
                quant_score, moat_strength,
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
                        fundamentals["ml_signal"] = enhanced_signal
                        fundamentals["ml_confidence"] = confidence
                        logger.debug(f"{ticker}: Enhanced signal {enhanced_signal} (conf={confidence:.2f}), rule-based {rule_based_signal}")
                    else:
                        logger.warning(f"{ticker}: No price data available for ML enhancement")
                        signal = rule_based_signal
                        fundamentals["ml_signal"] = ""
                        fundamentals["ml_confidence"] = 0.0
                except Exception as e:
                    logger.warning(f"{ticker}: ML enhancement failed: {e}")
                    signal = rule_based_signal
                    fundamentals["ml_signal"] = ""
                    fundamentals["ml_confidence"] = 0.0

            # Apply regime-aware signal override
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute('''SELECT mr.regime, mr.confidence, ma.qs_buy_threshold, ma.qs_sell_threshold,
                                   ma.position_size_multiplier, ma.signal_confidence
                                FROM market_regime mr
                                LEFT JOIN market_regime_adaptations ma ON ma.regime_id = mr.id
                                ORDER BY mr.recorded_at DESC
                                LIMIT 1''')
                regime_row = cursor.fetchone()
                conn.close()
                if regime_row:
                    current_regime = regime_row[0]
                    regime_confidence = regime_row[1]
                    qs_buy_threshold = regime_row[2] or 60
                    qs_sell_threshold = regime_row[3] or 20
                    signal_conf = regime_row[5] or 0.5
                    
                    # Adjust signal in extreme regimes
                    if regime_confidence >= 50 and signal_conf < 0.6:
                        # Low regime confidence → downgrade BUY signals, upgrade SELL
                        if signal == 'BUY' and quant_score < qs_buy_threshold:
                            signal = 'HOLD'
                            logger.debug(f"{ticker}: Regime-adjusted {rule_based_signal} → HOLD (QS {quant_score:.0f} < {qs_buy_threshold})")
                        elif signal == 'SELL' and quant_score > qs_sell_threshold:
                            signal = 'HOLD'
                            logger.debug(f"{ticker}: Regime-adjusted {rule_based_signal} → HOLD (QS {quant_score:.0f} > {qs_sell_threshold})")
                    
                    if current_regime in ('BEAR_STRONG', 'HIGH_VOLATILITY') and regime_confidence >= 50:
                        # In bear/high-vol: only BUY if QS is very high
                        if signal == 'BUY' and quant_score < qs_buy_threshold:
                            previous = signal
                            signal = 'HOLD'
                            logger.debug(f"{ticker}: {current_regime} regime → downgraded {previous} → HOLD (QS {quant_score:.0f} < {qs_buy_threshold})")
                    
                    fundamentals["regime"] = current_regime
                    fundamentals["regime_confidence"] = regime_confidence
                else:
                    signal = rule_based_signal
                    fundamentals["ml_signal"] = ""
                    fundamentals["ml_confidence"] = 0.0
            except Exception as e:
                logger.debug(f"Regime adjustment unavailable: {e}")
                signal = rule_based_signal
                fundamentals["regime"] = "UNKNOWN"
                fundamentals["regime_confidence"] = 0.0
                fundamentals["ml_signal"] = ""
                fundamentals["ml_confidence"] = 0.0

            # Store additional info for debugging/training
            fundamentals["signal"] = signal  # FIX: _save_scores reads fundamentals["signal"]
            fundamentals["rule_based_signal"] = rule_based_signal
            # fundamentals["ml_confidence"] already set above

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

            # Record signal outcome for ML training
            ml_confidence_val = fundamentals.get("ml_confidence", 0.0)
            _record_signal_outcome(
                ticker=ticker,
                signal_date=date.today().isoformat(),
                rule_based_signal=rule_based_signal,
                ml_signal=fundamentals.get("ml_signal", ""),
                ml_confidence=ml_confidence_val,
                final_signal=signal,
                db_path=db_path
            )

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

    _check_scan_health(results)

    return results


def _check_scan_health(results: Dict) -> None:
    """
    Post-scan invariant checks.

    A prior incident let fundamentals["signal"] go unset for six weeks,
    silently producing all-NULL signals with no visible failure -- nothing
    paged anyone because every ticker still counted as "successful". This
    check catches that whole class of degenerate-but-not-erroring scan
    outcome and pages via the existing alert channel (Telegram, if
    configured) instead of relying on someone noticing the dashboard looks
    empty.
    """
    total = results.get("total_tickers", 0)
    successful = results.get("successful", 0)
    failed = results.get("failed", 0)
    signal_total = (
        results.get("buy_signals", 0)
        + results.get("hold_signals", 0)
        + results.get("sell_signals", 0)
        + results.get("avoid_signals", 0)
    )

    problems = []

    if total > 0 and successful == 0:
        problems.append(f"0/{total} tickers scanned successfully")
    elif total > 0 and failed / total > 0.5:
        problems.append(f"{failed}/{total} tickers failed ({failed / total:.0%} failure rate)")

    if successful > 0 and signal_total == 0:
        problems.append(
            f"{successful} tickers scanned successfully but produced 0 categorized "
            f"signals (BUY/HOLD/SELL/AVOID) -- signals may be NULL or malformed"
        )
    elif successful > 0 and signal_total < successful * 0.5:
        problems.append(
            f"Only {signal_total}/{successful} successful scans produced a categorized signal"
        )

    if problems:
        message = "Weekly scan health check FAILED:\n- " + "\n- ".join(problems)
        if results.get("errors"):
            message += "\nSample errors:\n- " + "\n- ".join(results["errors"][:3])
        logger.error(message)
        try:
            alert_manager.add_alert(
                ticker="SYSTEM",
                alert_type="pipeline_health",
                message=message,
                priority="urgent",
                data={k: v for k, v in results.items() if k != "errors"},
            )
        except Exception as e:
            logger.error(f"Failed to send pipeline health alert: {e}")
    else:
        logger.info("Scan health check passed.")


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