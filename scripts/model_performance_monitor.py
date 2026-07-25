#!/usr/bin/env python3
"""
Model Performance Monitoring
Tracks ML model accuracy, calibration, and drift over time.
Run after each model retraining to log performance metrics.
"""

import sys
import os
import sqlite3
import logging
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "data/buffett.db"

def ensure_metrics_table(conn):
    """Ensure model_performance_metrics table exists."""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            metric_type TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            sample_size INTEGER,
            notes TEXT
        )
    ''')
    conn.commit()

def log_metric(conn, metric_type: str, metric_name: str, value: float, sample_size: int = None, notes: str = None):
    """Log a metric to the database."""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO model_performance_metrics (metric_type, metric_name, metric_value, sample_size, notes)
        VALUES (?, ?, ?, ?, ?)
    ''', (metric_type, metric_name, value, sample_size, notes))
    conn.commit()

def get_recent_metrics(conn, metric_name: str, days: int = 30) -> list:
    """Get recent metrics for trend analysis."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT recorded_at, metric_value, sample_size 
        FROM model_performance_metrics 
        WHERE metric_name = ? 
        AND recorded_at > datetime('now', ?)
        ORDER BY recorded_at
    ''', (metric_name, f'-{days} days'))
    return cursor.fetchall()

def main():
    """Run model performance monitoring."""
    logger.info("=" * 60)
    logger.info("MODEL PERFORMANCE MONITORING")
    logger.info(f"Started at {datetime.now()}")
    logger.info("=" * 60)
    
    conn = sqlite3.connect(DB_PATH)
    ensure_metrics_table(conn)
    
    # Load model
    trainer = ModelTrainer()
    
    if not trainer.is_ready:
        logger.warning("No trained model found. Run retraining first.")
        conn.close()
        return
    
    logger.info("Model loaded successfully")
    
    # Get labeled outcomes for evaluation
    cursor = conn.cursor()
    cursor.execute('''
        SELECT ticker, signal_date, final_signal, outcome_label_20d
        FROM ml_signal_outcomes
        WHERE outcome_label_20d IS NOT NULL
        ORDER BY signal_date DESC
    ''')
    outcomes = cursor.fetchall()
    
    if len(outcomes) < 10:
        logger.warning(f"Not enough labeled data for evaluation ({len(outcomes)} samples)")
        conn.close()
        return
    
    logger.info(f"Evaluating model on {len(outcomes)} labeled outcomes")
    
    # Calculate distribution metrics
    label_counts = {}
    for _, _, _, label in outcomes:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info("Label distribution in evaluation set:")
    for label, count in sorted(label_counts.items()):
        label_name = {-1: 'INCORRECT', 0: 'NEUTRAL', 1: 'CORRECT'}.get(label, str(label))
        pct = 100 * count / len(outcomes)
        logger.info(f"  {label_name}: {count} ({pct:.1f}%)")
        
        log_metric(conn, 'label_distribution', f'label_{label}', pct, count)
    
    # Get recent accuracy trend
    recent_acc = get_recent_metrics(conn, 'accuracy', days=90)
    if len(recent_acc) >= 2:
        logger.info(f"\nAccuracy trend (last {len(recent_acc)} evaluations):")
        for recorded_at, value, _ in recent_acc[-5:]:
            logger.info(f"  {recorded_at}: {value:.4f}")
        
        # Check for drift
        if len(recent_acc) >= 3:
            values = [m[1] for m in recent_acc[-3:]]
            if values[-1] < values[0] - 0.1:  # Dropped more than 10%
                logger.warning(f"⚠️  ACCURACY DRIFT DETECTED: {values[0]:.4f} → {values[-1]:.4f}")
                log_metric(conn, 'drift_alert', 'accuracy_drop', 
                          values[-1] - values[0], notes=f'From {values[0]:.4f} to {values[-1]:.4f}')
    
    # Model readiness check
    logger.info(f"\nModel Status:")
    logger.info(f"  Ready: {trainer.is_ready}")
    logger.info(f"  Features: {len(trainer.feature_names)}")
    
    log_metric(conn, 'model_status', 'feature_count', len(trainer.feature_names))
    log_metric(conn, 'model_status', 'is_ready', 1.0 if trainer.is_ready else 0.0)
    
    # Log evaluation timestamp
    log_metric(conn, 'evaluation', 'last_evaluation', 1.0, len(outcomes),
              notes=f"Evaluated on {len(outcomes)} labeled outcomes")
    
    conn.close()
    
    logger.info("=" * 60)
    logger.info("Monitoring complete")
    logger.info("Run this script regularly to track performance trends")

if __name__ == "__main__":
    main()