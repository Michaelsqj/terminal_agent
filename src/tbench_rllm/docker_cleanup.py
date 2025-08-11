"""Docker cleanup utilities for managing resources during training."""

import logging
import subprocess
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class DockerResourceCleaner:
    """Periodically cleans up Docker resources to prevent exhaustion."""
    
    def __init__(self, cleanup_interval_seconds: int = 120):
        """
        Initialize the cleaner.
        
        Args:
            cleanup_interval_seconds: How often to run cleanup (default: 2 minutes)
        """
        self.cleanup_interval = cleanup_interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._force_cleanup_on_start = True
        
    def start(self):
        """Start the background cleanup thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Docker cleaner already running")
            return
            
        # Run immediate cleanup on start
        if self._force_cleanup_on_start:
            logger.info("Running immediate Docker cleanup on start")
            try:
                self._run_cleanup()
            except Exception as e:
                logger.error(f"Initial cleanup error: {e}")
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started Docker cleanup thread (interval: {self.cleanup_interval}s)")
        
    def stop(self):
        """Stop the background cleanup thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Stopped Docker cleanup thread")
        
    def _cleanup_loop(self):
        """Main cleanup loop that runs in background thread."""
        while not self._stop_event.is_set():
            try:
                self._run_cleanup()
            except Exception as e:
                logger.error(f"Docker cleanup error: {e}")
            
            # Sleep in small increments to allow quick shutdown
            for _ in range(self.cleanup_interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)
    
    def _run_cleanup(self):
        """Run the actual cleanup commands."""
        logger.debug("Running Docker cleanup...")
        
        # Clean up stopped containers
        stopped_containers = self._run_command(
            ["docker", "ps", "-a", "--filter", "status=exited", "--filter", "status=dead", "-q"]
        )
        if stopped_containers.strip():
            container_count = len(stopped_containers.strip().split('\n'))
            logger.info(f"Removing {container_count} stopped containers")
            self._run_command(["docker", "container", "prune", "-f"])
        
        # Count networks before cleanup
        networks_before = self._run_command(["docker", "network", "ls", "-q"])
        network_count_before = len(networks_before.strip().split('\n')) if networks_before.strip() else 0
        
        # Remove ALL unused networks (docker network prune handles this safely)
        self._run_command(["docker", "network", "prune", "-f"])
        
        # Count networks after cleanup
        networks_after = self._run_command(["docker", "network", "ls", "-q"])
        network_count_after = len(networks_after.strip().split('\n')) if networks_after.strip() else 0
        networks_removed = network_count_before - network_count_after
        
        if networks_removed > 0:
            logger.info(f"Removed {networks_removed} unused networks")
        
        # Clean up dangling images (optional, less aggressive)
        # self._run_command(["docker", "image", "prune", "-f"])
        
        logger.debug("Docker cleanup completed")
    
    def _run_command(self, cmd: list, check: bool = True) -> str:
        """Run a command and return output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check,
                timeout=30
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            if check:
                logger.error(f"Command failed: {' '.join(cmd)}: {e.stderr}")
            return ""
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            return ""


# Global singleton instance
_docker_cleaner: Optional[DockerResourceCleaner] = None


def start_docker_cleanup(cleanup_interval_seconds: int = 120):
    """Start the global Docker cleanup service."""
    global _docker_cleaner
    if _docker_cleaner is None:
        _docker_cleaner = DockerResourceCleaner(cleanup_interval_seconds)
        _docker_cleaner.start()
    return _docker_cleaner


def stop_docker_cleanup():
    """Stop the global Docker cleanup service."""
    global _docker_cleaner
    if _docker_cleaner:
        _docker_cleaner.stop()
        _docker_cleaner = None