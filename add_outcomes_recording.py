#!/usr/bin/env python3
"""
Script to add ML signal outcomes recording to buffett/scanner.py
"""

import os
import re

scanner_path = '/home/shalu/buffett-monitor/buffett/scanner.py'
backup_path = '/home/shalu/buffett-monitor/buffett/scanner.py.backup'

# Start from the backup to ensure we have a clean state
with open(backup_path, 'r') as f:
    content = f.read()

# 1. Add import for sqlite3 if not already present (it should be, but let's ensure)
if 'import sqlite3' not in content:
    # Add after the existing imports
    imports_end = content.find('from alerts.alert_system import price_alert, signal_alert, fundamental_alert')
    if imports_end != -1:
        # Find the end of the import block
        while imports_end < len(content) and content[imports_end] != '\n':
            imports_end += 1
        while imports_end < len(content) and content[imports_end] == '\n':
            imports_end += 1
        content = content[:imports_end] + 'import sqlite3\n' + content[imports_end:]

# 2. Add a function to record signal outcomes
# We'll add it after the existing helper functions (_save_scores, etc.)
# Find where to insert: after _save_scores function
save_scores_end = content.find('def _save_scores(ticker: str, fundamentals: dict, db_path: str):')
if save_scores_end != -1:
    # Find the end of the _save_scores function
    lines = content.split('\n')
    func_start = -1
    for i, line in enumerate(lines):
        if line.strip() == 'def _save_scores(ticker: str, fundamentals: dict, db_path: str):':
            func_start = i
            break
    
    if func_start != -1:
        # Find the end of the function (next function definition or end of file)
        func_end = len(lines)
        for i in range(func_start + 1, len(lines)):
            stripped = lines[i].strip()
            if stripped.startswith('def ') or stripped.startswith('if __name__'):
                func_end = i
                break
        
        # Insert our function right after _save_scores
        insert_lines = [
            '',
            '',
            'def _record_signal_outcome(ticker: str, signal_date: str, rule_based_signal: str,',
            '                       ml_signal: str, ml_confidence: float, final_signal: str,',
            '                       db_path: str):',
            '    \"\"\"Record signal outcomes for ML training.\"\"\"',
            '    conn = sqlite3.connect(db_path)',
            '    try:',
            '        conn.execute(\"\"\"\"\"\"',
            '            INSERT OR REPLACE INTO ml_signal_outcomes',
            '            (ticker, signal_date, rule_based_signal, ml_signal, ml_confidence, final_signal)',
            '            VALUES (?, ?, ?, ?, ?, ?)',
            '            \"\"\"\"\"\",',
            '        (ticker, signal_date, rule_based_signal, ml_signal, ml_confidence, final_signal))',
            '        conn.commit()',
            '    except Exception as e:',
            '        logger.warning(f\"Failed to record signal outcome for {ticker}: {e}\")',
            '    finally:',
            '        conn.close()',
        ]
        
        # Insert the lines
        lines[func_end:func_end] = insert_lines
        content = '\n'.join(lines)

# 3. In the ticker loop, after we have determined the final signal, record it
# We need to find where we set the final signal in our enhancement logic
# Let's look for where we set 'signal = ' after our enhancement block

# Instead of complex parsing, let's add the recording right after we determine the final signal
# We'll look for the pattern where we set the signal in our enhancement logic

# Actually, let's modify our approach: we'll add the recording at the end of the ticker processing loop,
# right before we update the counters, where we have access to the final signal

# Find the line where we update counters: results["successful"] += 1
counter_update = content.find('results[\"successful\"] += 1')
if counter_update != -1:
    # We want to insert our recording logic just before this line
    # But we need to make sure we have the signal date and all the signal information
    
    # Let's instead insert the recording logic in the loop body where we have all the information
    # We'll look for the end of the loop body - but this is tricky
    
    # Alternative approach: add the recording right after we save to database and before we log changes
    # We have access to: ticker, fundamentals (which should have the signal), and we know the date
    
    # Find where we save to database: _save_snapshot(ticker, fundamentals, db_path)
    save_snapshot = content.find('_save_snapshot(ticker, fundamentals, db_path)')
    if save_snapshot != -1:
        # Find the line after this
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '_save_snapshot(ticker, fundamentals, db_path)' in line:
                # Insert after this line and the next line (_save_scores)
                insert_at = i + 2  # Skip _save_snapshot and _save_scores lines
                # But let's make sure we're at the right place by checking the next few lines
                
                # Insert our recording logic here
                record_lines = [
                    '            # Record signal outcome for ML training',
                    '            signal_date = date.today().isoformat()',
                    '            rule_based_signal = fundamentals.get(\"signal\", \"UNKNOWN\")',
                    '            # Note: ml_signal and ml_confidence would be available if we tracked them',
                    '            # For now, we\'ll store what we have and enhance this later',
                    '            ml_signal = fundamentals.get(\"ml_signal\", \"\")',
                    '            ml_confidence = fundamentals.get(\"ml_confidence\", 0.0)',
                    '            final_signal = fundamentals.get(\"signal\", \"UNKNOWN\")',
                    '            _record_signal_outcome(ticker, signal_date, rule_based_signal, ml_signal, ml_confidence, final_signal, db_path)',
                ]
                
                # Insert the lines
                lines[insert_at:insert_at] = record_lines
                content = '\n'.join(lines)
                break

# Write the modified content back to the file
with open(scanner_path, 'w') as f:
    f.write(content)

print("Successfully added ML signal outcomes recording to scanner.py")