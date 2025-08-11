"""Judge-based reward calculation using LLM evaluation."""

import asyncio
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import litellm

from src.tbench_rllm.rewards.judge_backend import get_judge_backend


logger = logging.getLogger(__name__)




async def calculate_judge_score(
    messages: List[Dict[str, Any]],
    dockerfile_contents: str,
    task_name: Optional[str] = None,
    rollout_id: Optional[str] = None,
    max_retries: int = 3,
    enable_io_logging: bool = True,
    llm_judge_name: Optional[str] = None,
    judge_version: str = "latest",
) -> float:
    """
    Get LLM judge evaluation score.

    Args:
        messages: Conversation messages to evaluate
        dockerfile_contents: Contents of the dockerfile
        task_name: Name of the task (optional, used for logging)
        rollout_id: Unique identifier for this rollout (optional, used for logging)
        max_retries: Maximum number of retry attempts
        enable_io_logging: Whether to log judge inputs/outputs for debugging

    Returns:
        Judge score between 0.0 and 1.0
    """
    # Assert that both are provided when logging is enabled
    if enable_io_logging:
        assert task_name is not None and rollout_id is not None, (
            "Both task_name and rollout_id must be provided when enable_io_logging is True"
        )

    try:
        # Load judge prompt
        judge_prompt, actual_version = _load_judge_prompt(judge_version)
        if not judge_prompt:
            raise FileNotFoundError(
                f"Judge prompt not found for version: {judge_version}"
            )

        # Format conversation and create judge message
        judge_user_message = _format_judge_message(messages, dockerfile_contents)

        # Log judge input if enabled
        if enable_io_logging:
            _log_judge_input(rollout_id, task_name, messages, judge_user_message)

        llm_judge_name = llm_judge_name or os.environ.get("LLM_JUDGE_NAME")
        if not llm_judge_name:
            logger.error("LLM_JUDGE_NAME not set")
            return 0.0

        # Get judge backend
        backend = get_judge_backend(llm_judge_name)
        
        # Try to get valid judge response with retries
        conversation_history = []  # Track conversation for format correction
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    # First attempt - normal judge call
                    response = await backend.call_judge(judge_prompt, judge_user_message)
                else:
                    # Retry attempt - use format correction if previous response was invalid
                    if conversation_history:
                        response = await backend.call_judge_with_correction(judge_prompt, conversation_history)
                    else:
                        # Regular retry if no conversation history
                        response = await backend.call_judge(judge_prompt, judge_user_message)

                # Log raw response if enabled
                if enable_io_logging:
                    _log_judge_output(rollout_id, response, attempt)

                # Parse score from response
                score = _parse_judge_score(response)
                if score is not None:
                    logger.info(
                        f"Judge score for rollout {rollout_id or 'unknown'}: {score}"
                    )

                    # Log successful result if enabled
                    if enable_io_logging:
                        _log_judge_result(
                            rollout_id,
                            task_name,
                            score,
                            success=True,
                            judge_version=actual_version,
                        )

                    return score
                else:
                    # Invalid format - prepare for correction retry
                    logger.warning(
                        f"Judge attempt {attempt + 1} returned invalid format"
                    )
                    conversation_history = [
                        {"role": "user", "content": judge_user_message},
                        {"role": "assistant", "content": response},
                    ]

            except Exception as e:
                logger.warning(f"Judge attempt {attempt + 1} failed: {e}")
                conversation_history = []  # Reset on exception
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff

        logger.error(f"Failed to get valid judge score after {max_retries} attempts")
        if enable_io_logging:
            _log_judge_result(
                rollout_id,
                task_name,
                0.0,
                success=False,
                error="Max retries exceeded",
                judge_version=actual_version,
            )
        return 0.0

    except Exception as e:
        logger.error(f"Error in calculate_judge_score: {e}", exc_info=True)
        if enable_io_logging:
            _log_judge_result(
                rollout_id,
                task_name,
                0.0,
                success=False,
                error=str(e),
                judge_version=actual_version,
            )
        return 0.0


def _load_judge_prompt(version: str = "latest") -> tuple[Optional[str], str]:
    """Load the judge prompt from file.

    Args:
        version: Version to load (e.g., 'v4.0', 'latest')

    Returns:
        Tuple of (prompt_content, actual_version)
    """
    judge_prompts_dir = Path(__file__).parent / "judge_prompts"

    # Load versions.json to resolve version
    versions_file = judge_prompts_dir / "versions.json"
    if versions_file.exists():
        with open(versions_file, "r", encoding="utf-8") as f:
            versions_data = json.load(f)

        # Resolve 'latest' to actual version
        if version == "latest":
            actual_version = versions_data.get("latest", "v4.0")
        else:
            actual_version = version

        # Get filename for version
        if actual_version in versions_data.get("versions", {}):
            filename = versions_data["versions"][actual_version]["file"]
            judge_prompt_path = judge_prompts_dir / filename
        else:
            # Try direct filename
            judge_prompt_path = (
                judge_prompts_dir / f"terminal_agent_judge_prompt_{version}.md"
            )
    else:
        raise FileNotFoundError(f"versions.json not found in {judge_prompts_dir}")

    if not judge_prompt_path.exists():
        logger.error(f"Judge prompt not found at {judge_prompt_path}")
        raise FileNotFoundError(f"Judge prompt file not found: {judge_prompt_path}")

    prompt_content = judge_prompt_path.read_text()

    # Extract version from prompt header if available
    import re

    version_match = re.search(r"Judge Instructions v(\d+\.\d+)", prompt_content)
    if version_match:
        actual_version = f"v{version_match.group(1)}"

    return prompt_content, actual_version


def _format_judge_message(
    messages: List[Dict[str, Any]], dockerfile_contents: str
) -> str:
    """Format conversation messages for the judge."""
    formatted_parts = []
    msg_counter = 1
    user_message_count = 0

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Skip system messages
        if role == "system":
            continue

        # Determine display role
        if role == "assistant":
            display_role = "Agent"
        elif role == "user":
            user_message_count += 1
            # First user message stays as "User", others become "Env Response"
            display_role = "User" if user_message_count == 1 else "Env Response"
        else:
            display_role = role.capitalize()

        formatted_parts.append(f"[Message {msg_counter} by {display_role}]")
        formatted_parts.append(content)
        formatted_parts.append("")
        msg_counter += 1

    conversation_text = "\n".join(formatted_parts)

    # Create complete message with dockerfile
    return f"""# Dockerfile contents:
```dockerfile
{dockerfile_contents}
```

# Agent trajectory:
```
{conversation_text}
```"""


def _parse_judge_score(response_text: str) -> Optional[float]:
    """Parse single holistic score from judge response."""
    try:
        import yaml
        import re

        # Strategy 1: Try to extract YAML (with or without markdown blocks)
        yaml_text = response_text.strip()

        # Check for markdown code blocks
        if "```yaml" in response_text:
            start = response_text.find("```yaml") + 7
            end = response_text.find("```", start)
            if end > start:
                yaml_text = response_text[start:end].strip()
        elif "```" in response_text:
            # Try generic code block
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end > start:
                yaml_text = response_text[start:end].strip()

        # Try to parse as YAML
        try:
            data = yaml.safe_load(yaml_text)
            if isinstance(data, dict) and "score" in data:
                score = float(data["score"])
                if 0.0 <= score <= 1.0:
                    return score
        except (yaml.YAMLError, ValueError, TypeError):
            pass

        # Strategy 2: Regex patterns for score
        score_patterns = [
            r"score:\s*([\d.]+)",  # YAML style
            r'"score":\s*([\d.]+)',  # JSON style
            r"score\s*[:=]\s*([\d.]+)",  # Various separators
            r"Score:\s*([\d.]+)",  # Capitalized
            r"\bscore\s+(?:is\s+)?([\d.]+)",  # Natural language
        ]

        for pattern in score_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    if 0.0 <= score <= 1.0:
                        return score
                except ValueError:
                    continue

        # Strategy 3: Look for any float between 0 and 1 after "score" keyword
        score_section = re.search(
            r"score[^0-9]*?(0?\.\d+|1\.0|0|1)(?![0-9])", response_text, re.IGNORECASE
        )
        if score_section:
            try:
                score = float(score_section.group(1))
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                pass

        logger.warning(f"Could not extract valid score from response")
        return None

    except Exception as e:
        logger.error(f"Failed to parse judge score: {e}")
        return None


# I/O Logging helpers (lightweight version)


def _get_log_dir() -> Path:
    """Get the judge I/O log directory."""
    base_dir = Path(os.environ.get("JUDGE_IO_LOG_DIR", "./judge_io_logs"))
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _log_judge_input(
    rollout_id: str,
    task_name: str,
    messages: List[Dict[str, Any]],
    formatted_message: str,
):
    """Log judge input for debugging."""
    try:
        log_dir = _get_log_dir() / f"rollout_{rollout_id}"
        log_dir.mkdir(exist_ok=True)

        # Save metadata
        metadata = {
            "rollout_id": rollout_id,
            "task_name": task_name,
            "timestamp": datetime.now().isoformat(),
            "num_messages": len(messages),
        }

        with open(log_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Save formatted input
        with open(log_dir / "judge_input.txt", "w", encoding="utf-8") as f:
            f.write(formatted_message)

        logger.debug(f"Logged judge input for rollout {rollout_id}")

    except Exception as e:
        logger.debug(f"Failed to log judge input: {e}")


def _log_judge_output(rollout_id: str, response: str, attempt: int):
    """Log judge raw response."""
    try:
        log_dir = _get_log_dir() / f"rollout_{rollout_id}"
        log_dir.mkdir(exist_ok=True)

        filename = f"judge_response_attempt_{attempt}.txt"
        with open(log_dir / filename, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Attempt: {attempt}\n")
            f.write("=" * 80 + "\n\n")
            f.write(response)

        logger.debug(
            f"Logged judge response for rollout {rollout_id}, attempt {attempt}"
        )

    except Exception as e:
        logger.debug(f"Failed to log judge output: {e}")


def _log_judge_result(
    rollout_id: str,
    task_name: str,
    final_score: float,
    success: bool,
    error: Optional[str] = None,
    judge_version: str = "unknown",
):
    """Log final judge result."""
    try:
        log_dir = _get_log_dir() / f"rollout_{rollout_id}"
        log_dir.mkdir(exist_ok=True)

        result = {
            "rollout_id": rollout_id,
            "task_name": task_name,
            "timestamp": datetime.now().isoformat(),
            "final_score": final_score,
            "judge_version": judge_version,
            "success": success,
            "error": error,
        }

        with open(log_dir / "final_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # Also append to daily log
        daily_log = (
            _get_log_dir()
            / f"judge_results_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        )
        with open(daily_log, "a", encoding="utf-8") as f:
            json.dump(result, f)
            f.write("\n")

        status = "success" if success else "failure"
        logger.debug(
            f"Logged judge result for rollout {rollout_id}: {status}, score={final_score}"
        )

    except Exception as e:
        logger.debug(f"Failed to log judge result: {e}")
