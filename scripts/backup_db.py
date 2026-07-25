"""
Backup utility for Buffett Monitor database.
"""

import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def backup_database(db_path: str = "data/buffett.db", 
                   backup_dir: str = "data/backups") -> bool:
    """
    Create a timestamped backup of the database.
    
    Args:
        db_path: Path to the SQLite database
        backup_dir: Directory to store backups
        
    Returns:
        True if backup successful, False otherwise
    """
    try:
        # Ensure backup directory exists
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if database exists
        if not os.path.exists(db_path):
            logger.warning(f"Database not found at {db_path}")
            return False
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_name = Path(db_path).stem
        backup_filename = f"{db_name}_{timestamp}.db"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Copy the database file
        shutil.copy2(db_path, backup_path)
        
        logger.info(f"Database backed up to: {backup_path}")
        
        # Optional: clean up old backups (keep last 10)
        _cleanup_old_backups(backup_dir, keep=10)
        
        return True
        
    except Exception as e:
        logger.error(f"Error backing up database: {e}")
        return False


def _cleanup_old_backups(backup_dir: str, keep: int = 10):
    """Remove old backup files, keeping only the most recent 'keep' files."""
    try:
        backup_files = []
        for f in os.listdir(backup_dir):
            if f.endswith(".db"):
                full_path = os.path.join(backup_dir, f)
                backup_files.append((os.path.getmtime(full_path), full_path))
        
        # Sort by modification time (oldest first)
        backup_files.sort(key=lambda x: x[0])
        
        # Remove excess files
        if len(backup_files) > keep:
            files_to_remove = backup_files[:-keep]
            for _, file_path in files_to_remove:
                os.remove(file_path)
                logger.debug(f"Removed old backup: {file_path}")
                
    except Exception as e:
        logger.warning(f"Error cleaning up old backups: {e}")


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    success = backup_database()
    print(f"Backup {'successful' if success else 'failed'}")