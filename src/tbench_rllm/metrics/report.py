"""Generate reports from the metrics database."""

import argparse
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Any, List, Dict, Optional
import csv
import sys
import os
import pandas as pd

from ..rewards.metrics_db import get_db_path, get_db_connection


def group_rollouts_by_timestamp(
    rollouts: List[sqlite3.Row], 
    time_window_seconds: int = 300 # Default 5 minutes
) -> List[List[sqlite3.Row]]:
    """
    Group rollouts that occurred within a time window of each other.
    
    Args:
        rollouts: List of rollout records sorted by timestamp
        time_window_seconds: Maximum seconds between rollouts in same group
        
    Returns:
        List of grouped rollouts
    """
    if not rollouts:
        return []
    
    groups = []
    current_group = [rollouts[0]]
    
    for i in range(1, len(rollouts)):
        # Parse timestamps
        current_ts = datetime.fromisoformat(rollouts[i]['timestamp'])
        prev_ts = datetime.fromisoformat(rollouts[i-1]['timestamp'])
        
        # Check if within time window
        if (current_ts - prev_ts).total_seconds() <= time_window_seconds:
            current_group.append(rollouts[i])
        else:
            # Start new group
            groups.append(current_group)
            current_group = [rollouts[i]]
    
    # Add last group
    if current_group:
        groups.append(current_group)
    
    return groups


def get_total_tasks_from_parquet() -> Optional[int]:
    """Get total number of tasks from train.parquet file."""
    parquet_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'train.parquet')
    try:
        df = pd.read_parquet(parquet_path)
        return len(df)
    except Exception:
        return None


def calculate_group_metrics(group: List[sqlite3.Row]) -> Dict[str, Any]:
    """Calculate average metrics for a group of rollouts."""
    test_scores = [r['test_score'] for r in group if r['test_score'] is not None]
    judge_scores = [r['judge_score'] for r in group if r['judge_score'] is not None]
    final_rewards = [r['final_reward'] for r in group if r['final_reward'] is not None]
    
    # Count how many rollouts have messages
    messages_count = sum(1 for r in group if r['messages_size'] is not None and r['messages_size'] > 0)
    
    return {
        'avg_test_score': sum(test_scores) / len(test_scores) if test_scores else 0.0,
        'avg_judge_score': sum(judge_scores) / len(judge_scores) if judge_scores else 0.0,
        'avg_final_reward': sum(final_rewards) / len(final_rewards) if final_rewards else 0.0,
        'min_final_reward': min(final_rewards) if final_rewards else 0.0,
        'max_final_reward': max(final_rewards) if final_rewards else 0.0,
        'num_rollouts': len(group),
        'rollouts_with_messages': messages_count
    }


def generate_task_instance_report(
    task_filter: Optional[str] = None,
    since_date: Optional[str] = None,
    time_window_seconds: int = 300
) -> List[Dict[str, Any]]:
    """
    Generate task instance report from the metrics database.
    
    Args:
        task_filter: Optional task name to filter by
        since_date: Optional date string (YYYY-MM-DD) to filter from
        time_window_seconds: Time window for grouping rollouts into task instances
        
    Returns:
        List of task instance records with aggregated metrics
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM rollout_metrics WHERE 1=1"
        params = []
        
        if task_filter:
            query += " AND task_name = ?"
            params.append(task_filter)
        
        if since_date:
            query += " AND timestamp >= ?"
            params.append(since_date)
        
        query += " ORDER BY task_name, timestamp"
        
        cursor.execute(query, params)
        rollouts = cursor.fetchall()
    
    # Group by task name first
    task_rollouts = defaultdict(list)
    for rollout in rollouts:
        task_rollouts[rollout['task_name']].append(rollout)
    
    # Process each task's rollouts
    task_instances = []
    for task_name, task_rollout_list in task_rollouts.items():
        # Group rollouts by timestamp
        groups = group_rollouts_by_timestamp(task_rollout_list, time_window_seconds)
        
        for group in groups:
            metrics = calculate_group_metrics(group)
            task_instances.append({
                'task_name': task_name,
                'batch_timestamp': group[0]['timestamp'],
                'num_rollouts': metrics['num_rollouts'],
                'avg_test_score': metrics['avg_test_score'],
                'avg_judge_score': metrics['avg_judge_score'],
                'avg_final_reward': metrics['avg_final_reward'],
                'min_final_reward': metrics['min_final_reward'],
                'max_final_reward': metrics['max_final_reward']
            })
    
    # Sort by timestamp
    task_instances.sort(key=lambda x: x['batch_timestamp'])
    
    return task_instances


def print_task_instance_report(task_instances: List[Dict[str, Any]]):
    """Print task instance report to console in a formatted table."""
    if not task_instances:
        print("No task instances found")
        return
    
    # Calculate timing information
    first_timestamp = datetime.fromisoformat(task_instances[0]['batch_timestamp'])
    last_timestamp = datetime.fromisoformat(task_instances[-1]['batch_timestamp'])
    elapsed_time = last_timestamp - first_timestamp
    
    # Format elapsed time
    total_seconds = int(elapsed_time.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    elapsed_str = f"{hours}h {minutes}m {seconds}s"
    
    # Get unique tasks completed
    unique_tasks_completed = len(set(t['task_name'] for t in task_instances))
    
    # Get total tasks from parquet
    total_tasks = get_total_tasks_from_parquet()
    
    # Calculate ETA if we have total tasks
    eta_str = "N/A"
    if total_tasks and unique_tasks_completed > 0:
        # Calculate average time per task
        if elapsed_time.total_seconds() > 0:
            avg_time_per_task = elapsed_time.total_seconds() / unique_tasks_completed
            remaining_tasks = total_tasks - unique_tasks_completed
            eta_seconds = remaining_tasks * avg_time_per_task
            
            # Format ETA
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)
            eta_seconds = int(eta_seconds % 60)
            eta_str = f"{eta_hours}h {eta_minutes}m {eta_seconds}s"
    
    # Header
    print(f"\n{'Task Name':<30} {'Timestamp':<20} {'N':<3} {'Test':<6} {'Judge':<6} {'Final':<6} {'Min':<6} {'Max':<6}")
    print("-" * 100)
    
    # Rows
    for instance in task_instances:
        timestamp = datetime.fromisoformat(instance['batch_timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{instance['task_name']:<30} {timestamp:<20} {instance['num_rollouts']:<3} "
              f"{instance['avg_test_score']:<6.3f} {instance['avg_judge_score']:<6.3f} "
              f"{instance['avg_final_reward']:<6.3f} {instance['min_final_reward']:<6.3f} "
              f"{instance['max_final_reward']:<6.3f}")
    
    # Summary statistics
    total_rollouts = sum(instance['num_rollouts'] for instance in task_instances)
    print(f"\nTask instances: {len(task_instances)}")
    print(f"Total rollouts: {total_rollouts}")
    print(f"Unique tasks completed: {unique_tasks_completed}")
    if total_tasks:
        print(f"Total tasks in dataset: {total_tasks}")
        print(f"Progress: {unique_tasks_completed}/{total_tasks} ({unique_tasks_completed/total_tasks*100:.1f}%)")
    print(f"Average final reward across instances: {sum(t['avg_final_reward'] for t in task_instances) / len(task_instances):.3f}")
    
    # Timing information
    print(f"\nTime elapsed: {elapsed_str}")
    print(f"First datapoint: {first_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Last datapoint: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    if total_tasks and unique_tasks_completed > 0:
        print(f"Estimated time remaining: {eta_str}")


def export_to_csv(task_instances: List[Dict[str, Any]], output_path: str):
    """Export task instance report to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'task_name', 'batch_timestamp', 'num_rollouts',
            'avg_test_score', 'avg_judge_score', 'avg_final_reward',
            'min_final_reward', 'max_final_reward'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for instance in task_instances:
            writer.writerow(instance)
    
    print(f"Report exported to {output_path}")


def main():
    """Main entry point for the report generator."""
    parser = argparse.ArgumentParser(description='Generate reports from training metrics')
    parser.add_argument('--task', help='Filter by task name')
    parser.add_argument('--since', help='Filter by date (YYYY-MM-DD)')
    parser.add_argument('--output', help='Export to CSV file')
    parser.add_argument('--window', type=int, default=300, 
                       help='Time window in seconds for grouping rollouts (default: 300)')
    
    args = parser.parse_args()
    
    # Check if database exists
    db_path = get_db_path()
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rollout_metrics")
            count = cursor.fetchone()[0]
            print(f"Database: {db_path} ({count} rollouts)")
    except sqlite3.OperationalError:
        print(f"Error: Database not found or table doesn't exist at {db_path}")
        print("Make sure training has been run with metrics logging enabled")
        sys.exit(1)
    
    # Generate report
    task_instances = generate_task_instance_report(
        task_filter=args.task,
        since_date=args.since,
        time_window_seconds=args.window
    )
    
    if args.output:
        export_to_csv(task_instances, args.output)
    else:
        print_task_instance_report(task_instances)


if __name__ == '__main__':
    main()