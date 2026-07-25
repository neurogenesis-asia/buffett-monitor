"""
Change log functionality for tracking changes in fundamentals and signals.
Implements diff_previous(ticker, new_snapshot) -> writes buffett_change_log rows.
"""

import json
import sqlite3
from datetime import date
from typing import Dict, Optional, Any, List
from pathlib import Path


def diff_previous(ticker: str, new_snapshot: Dict, db_path: str = "data/buffett.db") -> List[Dict]:
    """
    Compare new snapshot with the most recent previous snapshot and log changes.
    
    Args:
        ticker: Stock ticker
        new_snapshot: Dictionary containing the new fundamentals data
        db_path: Path to the SQLite database
       
    Returns:
        List of change log entries created
    """
    # Ensure data directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    try:
        # Get the most recent previous snapshot for this ticker
        cursor = conn.execute("""
            SELECT * FROM buffett_fundamentals 
            WHERE ticker = ? 
            ORDER BY snapshot_date DESC 
            LIMIT 1 OFFSET 1
        """, (ticker,))
        prev_row = cursor.fetchone()
        
        if prev_row is None:
            # No previous snapshot, this is either the first or second snapshot
            # Get the most recent to see if this is the first
            cursor = conn.execute("""
                SELECT snapshot_date FROM buffett_fundamentals 
                WHERE ticker = ? 
                ORDER BY snapshot_date DESC 
                LIMIT 1
            """, (ticker,))
            most_recent = cursor.fetchone()
            
            if most_recent is None:
                # This is the very first snapshot - log as NEW_TICKER
                return _log_new_ticker(ticker, new_snapshot, conn)
            else:
                # This is the second snapshot - compare with the first
                cursor = conn.execute("""
                    SELECT * FROM buffett_fundamentals 
                    WHERE ticker = ? AND snapshot_date = ?
                """, (ticker, most_recent[0]))
                prev_row = cursor.fetchone()
        
        # Convert previous row to dictionary for comparison
        if prev_row:
            prev_snapshot = _row_to_dict(prev_row, conn)
            changes = _compare_snapshots(prev_snapshot, new_snapshot)
            
            # Log each change
            changes_logged = []
            for change in changes:
                _log_change(conn, ticker, new_snapshot.get('snapshot_date'), change)
                changes_logged.append(change)
            
            conn.commit()
            return changes_logged
        else:
            # Should not happen, but handle gracefully
            return []
            
    finally:
        conn.close()


def _row_to_dict(row, conn) -> Dict:
    """Convert a database row to a dictionary using column names."""
    cursor = conn.execute("SELECT * FROM buffett_fundamentals LIMIT 1")
    column_names = [description[0] for description in cursor.description]
    return dict(zip(column_names, row))


def _compare_snapshots(prev: Dict, new: Dict) -> List[Dict]:
    """
    Compare two snapshots and return a list of changes.
    
    Each change is a dict with: field_name, old_value, new_value, change_type, severity
    """
    changes = []
    
    # Define which fields to track for changes and their significance
    tracked_fields = {
        # Critical changes that might signal deteriorating fundamentals
        'pe_ratio': {'threshold': 0.20, 'type': 'THRESHOLD_BREACH', 'severity': 'WARN'},
        'pb_ratio': {'threshold': 0.20, 'type': 'THRESHOLD_BREACH', 'severity': 'WARN'},
        'de_ratio': {'threshold': 0.25, 'type': 'THRESHOLD_BREACH', 'severity': 'WARN'},
        'current_ratio': {'threshold': 0.20, 'type': 'THRESHOLD_BREACH', 'severity': 'WARN'},
        'roe_latest': {'threshold': 0.15, 'type': 'THRESHOLD_BREACH', 'severity': 'WARN'},
        'roe_5yr_avg': {'threshold': 0.15, 'type': 'THRESHOLD_BREACH', 'severity': 'WARN'},
        'dividend_yield': {'threshold': 0.25, 'type': 'THRESHOLD_BREACH', 'severity': 'INFO'},
        'operating_cf': {'threshold': 0.30, 'type': 'THRESHOLD_BREACH', 'severity': 'WARN'},
        
        # Fields that are either same/different (no threshold)
        'signal': {'type': 'SIGNAL_CHANGE', 'severity': 'ALERT'},
        'moat_strength': {'type': 'SIGNAL_CHANGE', 'severity': 'WARN'},
        'mgmt_quality': {'type': 'SIGNAL_CHANGE', 'severity': 'INFO'},
        'data_sources_json': {'type': 'DATA_CORRECTION', 'severity': 'INFO'},
        'fetch_errors_json': {'type': 'DATA_CORRECTION', 'severity': 'WARN'},
    }
    
    snapshot_date = new.get('snapshot_date')
    
    for field_name, rules in tracked_fields.items():
        old_val = prev.get(field_name)
        new_val = new.get(field_name)
        
        # Skip if both are None
        if old_val is None and new_val is None:
            continue
            
        # Handle JSON fields specially
        if field_name in ['data_sources_json', 'fetch_errors_json']:
            try:
                old_parsed = json.loads(old_val) if old_val else {}
                new_parsed = json.loads(new_val) if new_val else {}
                if old_parsed != new_parsed:
                    changes.append({
                        'field_name': field_name,
                        'old_value': old_val,
                        'new_value': new_val,
                        'change_type': rules['type'],
                        'severity': rules['severity']
                    })
            except (json.JSONDecodeError, TypeError):
                if str(old_val) != str(new_val):
                    changes.append({
                        'field_name': field_name,
                        'old_value': str(old_val) if old_val is not None else None,
                        'new_value': str(new_val) if new_val is not None else None,
                        'change_type': rules['type'],
                        'severity': rules['severity']
                    })
            continue
            
        # Handle numeric fields with thresholds
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            # Avoid division by zero
            if old_val == 0:
                change_pct = float('inf') if new_val != 0 else 0
            else:
                change_pct = abs(new_val - old_val) / abs(old_val)
                
            if change_pct >= rules['threshold']:
                changes.append({
                    'field_name': field_name,
                    'old_value': old_val,
                    'new_value': new_val,
                    'change_type': rules['type'],
                    'severity': rules['severity']
                })
        else:
            # For non-numeric or mixed types, check for exact inequality
            if old_val != new_val:
                changes.append({
                    'field_name': field_name,
                    'old_value': old_val,
                    'new_value': new_val,
                    'change_type': rules['type'],
                    'severity': rules['severity']
                })
    
    return changes


def _log_change(conn, ticker: str, snapshot_date: str, change: Dict):
    """Log a single change to the buffett_change_log table."""
    conn.execute("""
        INSERT INTO buffett_change_log 
        (ticker, snapshot_date, field_name, old_value, new_value, change_type, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        ticker,
        snapshot_date,
        change['field_name'],
        str(change['old_value']) if change['old_value'] is not None else None,
        str(change['new_value']) if change['new_value'] is not None else None,
        change['change_type'],
        change['severity']
    ))


def _log_new_ticker(ticker: str, new_snapshot: Dict, conn) -> List[Dict]:
    """Log the first snapshot of a new ticker as a NEW_TICKER change."""
    # Log a few key fields as NEW_TICKER changes
    key_fields = ['price', 'pe_ratio', 'pb_ratio', 'roe_latest', 'de_ratio', 'current_ratio']
    changes_logged = []
    
    for field in key_fields:
        if field in new_snapshot and new_snapshot[field] is not None:
            conn.execute("""
                INSERT INTO buffett_change_log 
                (ticker, snapshot_date, field_name, old_value, new_value, change_type, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                new_snapshot.get('snapshot_date'),
                field,
                None,  # No old value
                str(new_snapshot[field]),
                'NEW_TICKER',
                'INFO'
            ))
            changes_logged.append({
                'field_name': field,
                'old_value': None,
                'new_value': new_snapshot[field],
                'change_type': 'NEW_TICKER',
                'severity': 'INFO'
            })
    
    return changes_logged


def get_recent_changes(ticker: str = None, limit: int = 50, db_path: str = "data/buffett.db") -> List[Dict]:
    """
    Get recent change log entries.
    
    Args:
        ticker: Optional ticker to filter by
        limit: Maximum number of entries to return
        db_path: Path to the SQLite database
       
    Returns:
        List of change log entries as dictionaries
    """
    conn = sqlite3.connect(db_path)
    try:
        if ticker:
            cursor = conn.execute("""
                SELECT * FROM buffett_change_log 
                WHERE ticker = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (ticker, limit))
        else:
            cursor = conn.execute("""
                SELECT * FROM buffett_change_log 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
        
        # Convert rows to dictionaries
        column_names = [description[0] for description in cursor.description]
        return [dict(zip(column_names, row)) for row in cursor.fetchall()]
    finally:
        conn.close()


if __name__ == "__main__":
    # Simple test
    print("Change log module loaded successfully")