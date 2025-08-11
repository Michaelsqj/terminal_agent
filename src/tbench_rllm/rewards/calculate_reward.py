"""Main reward calculation function that combines test and judge scores."""

from dataclasses import dataclass
import logging
from typing import List, Dict, Any

from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.terminal.terminal import Terminal

from src.tbench_rllm.metrics.metrics_logger import log_rollout_metrics


from .test_reward import calculate_test_score
from .judge_reward import calculate_judge_score

logger = logging.getLogger(__name__)

@dataclass
class RewardResults:
    raw_judge_score: float
    raw_test_score: float
    final_weighted_reward: float


async def calculate_reward(
    terminal: Terminal,
    trial_handler: TrialHandler,
    task_name: str,
    test_weights: Dict[str, float],
    dockerfile_contents: str,
    max_test_timeout_sec: float,
    messages: List[Dict[str, Any]],
    rollout_id: str,
    test_weight: float = 0.65,
    judge_weight: float = 0.35
) -> RewardResults:
    """
    Calculate the final reward by combining test and judge scores.
    
    Args:
        terminal: Terminal instance for the container
        trial_handler: Trial handler with task information
        task_name: Name of the task
        test_weights: Test name to weight mapping
        dockerfile_contents: Contents of the dockerfile
        max_test_timeout_sec: Maximum timeout for tests
        messages: Conversation messages for judge evaluation
        rollout_id: Unique identifier for this rollout
        test_weight: Weight for test-based scoring (default 0.65)
        judge_weight: Weight for judge-based scoring (default 0.35)
        
    Returns:
        RewardResults with test score, judge score, and final weighted reward
    """
    try:
        # Calculate test-based score
        test_score = await calculate_test_score(
            terminal=terminal,
            trial_handler=trial_handler,
            task_name=task_name,
            test_weights=test_weights,
            max_test_timeout_sec=max_test_timeout_sec,
            rollout_id=rollout_id
        )
        logger.info(f"Test score for {task_name} (rollout {rollout_id}): {test_score:.3f}")
        
        # Calculate judge-based score
        judge_score = await calculate_judge_score(
            messages=messages,
            task_name=task_name,
            dockerfile_contents=dockerfile_contents,
            rollout_id=rollout_id
        )
        logger.info(f"Judge score for {task_name} (rollout {rollout_id}): {judge_score:.3f}")
        
        # Calculate final weighted reward
        final_reward = (test_weight * test_score) + (judge_weight * judge_score)
        logger.info(f"Final reward for {task_name} (rollout {rollout_id}): {final_reward:.3f}")
        
        # Log metrics to database
        log_rollout_metrics(
            env_id=rollout_id,
            task_name=task_name,
            test_score=test_score,
            judge_score=judge_score,
            final_reward=final_reward,
            messages=messages
        )
        
        return RewardResults(
            raw_judge_score=judge_score,
            raw_test_score=test_score,
            final_weighted_reward=final_reward
        )
        
    except Exception as e:
        logger.error(f"Error calculating reward for {task_name} (rollout {rollout_id}): {e}", exc_info=True)
        return RewardResults(
            raw_judge_score=0.0,
            raw_test_score=0.0,
            final_weighted_reward=0.0
        )