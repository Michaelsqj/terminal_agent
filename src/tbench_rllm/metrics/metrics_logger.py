"""Thread-safe async metrics logging module for rollout results."""

import logging
import threading
import queue
import atexit
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from .metrics_db import init_database, insert_rollout_metrics


logger = logging.getLogger(__name__)


@dataclass
class MetricsLogEntry:
    """Data class for queued metrics entries."""
    env_id: str
    task_name: str
    test_score: float
    judge_score: float
    final_reward: float
    timestamp: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None


class AsyncMetricsLogger:
    """Asynchronous metrics logger with background worker thread."""
    
    def __init__(self, max_queue_size: int = 10000):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._worker_thread = None
        self._shutdown = threading.Event()
        self._initialized = False
        self._init_lock = threading.Lock()
        
    def _ensure_initialized(self):
        """Initialize the logger and start the worker thread if not already done."""
        with self._init_lock:
            if not self._initialized:
                try:
                    # Initialize database
                    init_database()
                    
                    # Start worker thread
                    self._worker_thread = threading.Thread(
                        target=self._worker_loop,
                        daemon=True,
                        name="MetricsLoggerWorker"
                    )
                    self._worker_thread.start()
                    
                    # Register shutdown handler
                    atexit.register(self.shutdown)
                    
                    self._initialized = True
                    logger.info("Async metrics logger initialized successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize async metrics logger: {e}")
    
    def _worker_loop(self):
        """Background worker that processes the queue."""
        logger.debug("Metrics logger worker thread started")
        
        while not self._shutdown.is_set():
            try:
                # Get entry from queue with timeout
                entry = self._queue.get(timeout=1.0)
                
                # Process the entry
                self._process_entry(entry)
                
                # Mark task as done
                self._queue.task_done()
                
            except queue.Empty:
                # No items to process, continue
                continue
            except Exception as e:
                # Log error but continue processing
                logger.error(f"Error in metrics worker thread: {e}", exc_info=True)
        
        # Process remaining items before shutdown
        self._drain_queue()
        logger.debug("Metrics logger worker thread stopped")
    
    def _process_entry(self, entry: MetricsLogEntry):
        """Process a single metrics entry."""
        try:
            insert_rollout_metrics(
                env_id=entry.env_id,
                task_name=entry.task_name,
                test_score=entry.test_score,
                judge_score=entry.judge_score,
                final_reward=entry.final_reward,
                timestamp=entry.timestamp,
                messages=entry.messages
            )
            logger.debug(f"Logged metrics for rollout {entry.env_id} (task: {entry.task_name})")
        except Exception as e:
            # Log error but don't crash - this is fire and forget
            logger.error(f"Failed to log metrics for rollout {entry.env_id}: {e}")
    
    def _drain_queue(self):
        """Process all remaining items in the queue."""
        while not self._queue.empty():
            try:
                entry = self._queue.get_nowait()
                self._process_entry(entry)
                self._queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error draining queue: {e}")
    
    def log_metrics(
        self,
        env_id: str,
        task_name: str,
        test_score: float,
        judge_score: float,
        final_reward: float,
        timestamp: Optional[datetime] = None,
        messages: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Log metrics asynchronously (fire and forget).
        
        This method returns immediately after queuing the metrics.
        If the queue is full, the metrics are silently dropped.
        """
        self._ensure_initialized()
        
        if self._shutdown.is_set():
            return
        
        try:
            # Convert timestamp to string if provided
            timestamp_str = timestamp.isoformat() if timestamp else None
            
            # Create log entry
            entry = MetricsLogEntry(
                env_id=env_id,
                task_name=task_name,
                test_score=test_score,
                judge_score=judge_score,
                final_reward=final_reward,
                timestamp=timestamp_str,
                messages=messages
            )
            
            # Try to add to queue (non-blocking)
            self._queue.put_nowait(entry)
            
        except queue.Full:
            # Queue is full, drop the metrics (fire and forget)
            logger.warning(f"Metrics queue full, dropping metrics for rollout {env_id}")
        except Exception as e:
            # Any other error, just log and continue
            logger.error(f"Error queuing metrics for rollout {env_id}: {e}")
    
    def shutdown(self, timeout: float = 30.0):
        """Gracefully shutdown the logger."""
        if self._shutdown.is_set():
            return
        
        logger.info("Shutting down async metrics logger...")
        self._shutdown.set()
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            
            if self._worker_thread.is_alive():
                logger.warning("Metrics worker thread did not stop cleanly")


# Global singleton instance
_logger_instance = None
_logger_lock = threading.Lock()


def get_logger() -> AsyncMetricsLogger:
    """Get the global async metrics logger instance."""
    global _logger_instance
    
    with _logger_lock:
        if _logger_instance is None:
            _logger_instance = AsyncMetricsLogger()
    
    return _logger_instance


def log_rollout_metrics(
    env_id: str,
    task_name: str,
    test_score: float,
    judge_score: float,
    final_reward: float,
    timestamp: Optional[datetime] = None,
    messages: Optional[List[Dict[str, Any]]] = None
):
    """
    Log metrics for a single rollout asynchronously.
    
    This is a convenience function that uses the global logger instance.
    Returns immediately (fire and forget).
    """
    logger_instance = get_logger()
    logger_instance.log_metrics(
        env_id=env_id,
        task_name=task_name,
        test_score=test_score,
        judge_score=judge_score,
        final_reward=final_reward,
        timestamp=timestamp,
        messages=messages
    )