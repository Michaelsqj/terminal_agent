"""SQLite database schema and utilities for storing rollout metrics."""

import sqlite3
import os
import json
import gzip
import base64
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


def get_db_path() -> str:
    """Get the database path from environment or use default."""
    # Check for custom path
    db_path = os.environ.get('METRICS_DB_PATH')
    if db_path:
        return db_path
    
    # Check if JUDGE_IO_LOG_DIR is set
    log_dir = os.environ.get('JUDGE_IO_LOG_DIR')
    if log_dir:
        return os.path.join(log_dir, 'training_metrics.db')
    
    # Default to project root
    return './training_metrics.db'


@contextmanager
def get_db_connection():
    """Get a database connection with proper error handling."""
    conn = None
    try:
        conn = sqlite3.connect(get_db_path())
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        if conn:
            conn.close()


def init_database():
    """Initialize the database with the required schema."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rollout_metrics (
                env_id TEXT PRIMARY KEY,
                task_name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                test_score REAL,
                judge_score REAL,
                final_reward REAL,
                messages_compressed TEXT,
                messages_size INTEGER
            )
        ''')
        
        # SQLite doesn't support INDEX in CREATE TABLE, so create indexes separately
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_task_name ON rollout_metrics(task_name)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON rollout_metrics(timestamp)
        ''')
        
        conn.commit()


def compress_messages(messages: List[Dict[str, Any]], max_size_mb: float = 10.0) -> Optional[str]:
    """
    Compress messages to a base64-encoded gzipped JSON string.
    
    Args:
        messages: List of message dictionaries
        max_size_mb: Maximum size in MB before compression (default 10MB)
        
    Returns:
        Base64-encoded compressed string, or None if messages are too large
    """
    if not messages:
        return None
    
    # Convert to JSON
    json_str = json.dumps(messages, separators=(',', ':'))
    
    # Check size before compression
    size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
    if size_mb > max_size_mb:
        # Messages too large, store a truncated version
        truncated_messages = messages[:10] + [{"role": "system", "content": f"[Truncated - {len(messages)} total messages]"}]
        json_str = json.dumps(truncated_messages, separators=(',', ':'))
    
    # Compress with gzip
    compressed = gzip.compress(json_str.encode('utf-8'))
    
    # Encode to base64 for storage
    return base64.b64encode(compressed).decode('utf-8')


def decompress_messages(compressed_data: str) -> Optional[List[Dict[str, Any]]]:
    """Decompress messages from base64-encoded gzipped JSON string."""
    if not compressed_data:
        return None
    
    try:
        # Decode from base64
        compressed = base64.b64decode(compressed_data.encode('utf-8'))
        
        # Decompress
        json_str = gzip.decompress(compressed).decode('utf-8')
        
        # Parse JSON
        return json.loads(json_str)
    except Exception:
        return None


def insert_rollout_metrics(
    env_id: str,
    task_name: str,
    test_score: float,
    judge_score: float,
    final_reward: float,
    timestamp: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None
):
    """Insert a single rollout's metrics into the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Compress messages if provided
        messages_compressed = None
        messages_size = 0
        if messages:
            messages_compressed = compress_messages(messages)
            messages_size = len(messages)
        
        if timestamp:
            cursor.execute('''
                INSERT OR REPLACE INTO rollout_metrics 
                (env_id, task_name, timestamp, test_score, judge_score, final_reward, messages_compressed, messages_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (env_id, task_name, timestamp, test_score, judge_score, final_reward, messages_compressed, messages_size))
        else:
            cursor.execute('''
                INSERT OR REPLACE INTO rollout_metrics 
                (env_id, task_name, test_score, judge_score, final_reward, messages_compressed, messages_size)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (env_id, task_name, test_score, judge_score, final_reward, messages_compressed, messages_size))
        
        conn.commit()