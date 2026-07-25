#!/usr/bin/env python3
"""
Label signal outcomes based on forward returns.
Creates training labels for ML model retraining.

Labeling logic (20d primary, 60d/252d secondary):
- BUY  + fwd > +3% → CORRECT (1)
- SELL + fwd < -2% → CORRECT (1)
- HOLD signal |fwd| <= 2% → NEUTRAL (0)
- All other cases → INCORRECT (-1)
- UNKNOWN signal → NEUTRAL (0)  (treated as weak label: +return -> 1, -return -> -1, else 0)
"""

import sys
import os
import sqlite3
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "data/buffett.db"

BUY_THRESHOLD  =  0.03   # +3%
SELL_THRESHOLD = -0.02   # -2%
HOLD_THRESHOLD =  0.02   # +/- 2%

def label_signal_outcome(signal, forward_return):
    """Return 1 (correct), 0 (neutral), or -1 (incorrect) based on signal and forward return."""
    if forward_return is None:
        return None
    s = (signal or '').upper()
    if s == 'BUY':
        if forward_return > BUY_THRESHOLD:
            return 1
        elif forward_return > 0:
            return 0
        else:
            return -1
    elif s == 'SELL':
        if forward_return < SELL_THRESHOLD:
            return 1
        elif forward_return < 0:
            return 0
        else:
            return -1
    elif s == 'HOLD':
        if abs(forward_return) <= HOLD_THRESHOLD:
            return 0
        else:
            return -1  # Incorrect if outside the band
    elif s == 'AVOID':
        if forward_return < 0:
            return 1
        else:
            return -1
    elif s == 'UNKNOWN':
        # Treat as weak signal: positive return -> correct, negative -> incorrect, zero -> neutral
        if forward_return > HOLD_THRESHOLD:
            return 1
        elif forward_return < -HOLD_THRESHOLD:
            return -1
        else:
            return 0
    return None

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure outcome_label columns exist
    cursor.execute("PRAGMA table_info(ml_signal_outcomes)")
    columns = [row[1] for row in cursor.fetchall()]
    for col in ('outcome_label_20d', 'outcome_label_60d', 'outcome_label_252d'):
        if col not in columns:
            cursor.execute(f"ALTER TABLE ml_signal_outcomes ADD COLUMN {col} INTEGER")
            logger.info(f"Added column {col}")
    conn.commit()

    # Select rows where we have at least one forward return not null
    query = """
    SELECT id, ticker, signal_date, final_signal,
           forward_20d_return, forward_60d_return, forward_252d_return,
           outcome_label_20d, outcome_label_60d, outcome_label_252d
    FROM ml_signal_outcomes
    WHERE forward_20d_return IS NOT NULL
       OR forward_60d_return IS NOT NULL
       OR forward_252d_return IS NOT NULL
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    logger.info(f"Found {len(rows)} signals with at least one forward return")

    updated_20d = updated_60d = updated_252d = 0
    skipped_no_return = 0

    for row in rows:
        (row_id, ticker, signal_date, final_signal,
         ret_20d, ret_60d, ret_252d,
         label_20d, label_60d, label_252d) = row

        # Skip if no forward returns at all (should not happen due to WHERE, but safe)
        if ret_20d is None and ret_60d is None and ret_252d is None:
            skipped_no_return += 1
            continue

        updates = []
        values = []

        # 20d
        if ret_20d is not None:
            label = label_signal_outcome(final_signal, ret_20d)
            if label is not None:
                updates.append("outcome_label_20d = ?")
                values.append(label)
                updated_20d += 1

        # 60d
        if ret_60d is not None:
            label = label_signal_outcome(final_signal, ret_60d)
            if label is not None:
                updates.append("outcome_label_60d = ?")
                values.append(label)
                updated_60d += 1

        # 252d
        if ret_252d is not None:
            label = label_signal_outcome(final_signal, ret_252d)
            if label is not None:
                updates.append("outcome_label_252d = ?")
                values.append(label)
                updated_252d += 1

        if updates:
            values.append(row_id)
            sql = f"UPDATE ml_signal_outcomes SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(sql, values)

    conn.commit()
    conn.close()

    logger.info(f"Updated labels: 20d={updated_20d}, 60d={updated_60d}, 252d={updated_252d}")
    if skipped_no_return:
        logger.info(f"Skipped {skipped_no_return} signals (no forward returns)")

if __name__ == "__main__":
    main()