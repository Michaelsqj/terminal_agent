import os
import asyncio
import shlex
import time

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import litellm
from pathlib import Path
import logging
logger = logging.getLogger(__name__)



class JudgeBackend(ABC):
    """Abstract base class for judge backend implementations."""
    
    @abstractmethod
    async def call_judge(self, system_prompt: str, user_message: str) -> str:
        """Make a judge call with the given prompts."""
        pass
    
    @abstractmethod
    async def call_judge_with_correction(
        self, system_prompt: str, conversation_history: List[Dict[str, str]]
    ) -> str:
        """Call judge with format correction based on conversation history."""
        pass


class LiteLLMJudgeBackend(JudgeBackend):
    """Judge backend using LiteLLM API."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    async def call_judge(self, system_prompt: str, user_message: str) -> str:
        """Make the actual LLM API call."""
        sys_msg = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        } if self.model_name.startswith("anthropic/") else {
            "role": "system",
            "content": system_prompt,
        }

        response = await asyncio.to_thread(
            litellm.completion,
            model=self.model_name,
            messages=[
                sys_msg,
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1000,
            timeout=60.0,
        )
        return response.choices[0].message.content
    
    async def call_judge_with_correction(
        self, system_prompt: str, conversation_history: List[Dict[str, str]]
    ) -> str:
        """Call judge with format correction message."""
        # Build correction message
        correction_message = _build_format_correction_message()
        
        # Build full message history
        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_history,
            {"role": "user", "content": correction_message},
        ]
        
        response = await asyncio.to_thread(
            litellm.completion,
            model=self.model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
            timeout=60.0,
        )
        return response.choices[0].message.content


class ClaudeCodeCLIJudgeBackend(JudgeBackend):
    """Judge backend using Claude Code CLI."""
    
    async def call_judge(self, system_prompt: str, user_message: str) -> str:
        """Call judge using Claude Code CLI."""
        # Get claude path from env var or use default
        claude_path = os.environ.get('CLAUDE_CLI_PATH')

        prompt = f'Assume the role of {shlex.quote(system_prompt)}\nJust write the above format to me with no other words surrounding it, then I will decide what to do\n\n{user_message}'

        cmd = [
            claude_path,
            '-p', prompt,
            '--dangerously-skip-permissions',
            '--model', "claude-sonnet-4-20250514"
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env=os.environ.copy()  # Pass current environment
        )
        
        stdout, _ = await asyncio.wait_for(
            proc.communicate(),
            timeout=60.0
        )            
        return stdout.decode('utf-8', errors='replace').strip()
            

    
    async def call_judge_with_correction(
        self, system_prompt: str, conversation_history: List[Dict[str, str]]
    ) -> str:
        """Call judge with format correction via CLI."""
        # Build correction message
        correction_message = _build_format_correction_message()
        
        # Combine conversation history into a single user message
        combined_message_parts = []
        for msg in conversation_history:
            role = msg["role"].upper()
            content = msg["content"]
            combined_message_parts.append(f"[{role}]\n{content}")
        
        combined_message_parts.append(f"[USER]\n{correction_message}")
        combined_user_message = "\n\n".join(combined_message_parts)
        
        # Call judge with combined message
        return await self.call_judge(system_prompt, combined_user_message)


# Cache for backend configuration
_backend_cache = {
    "last_check": 0,
    "backend_type": None,
    "model_name": None,
    "env_overrides": None,
    "cache_duration": 5.0  # Check file every 5 seconds max
}


def _read_backend_config() -> Optional[Tuple[str, str, Optional[Dict[str, str]]]]:
    """Read backend configuration from .judge_backend file.
    
    Returns:
        Tuple of (backend_type, model_name, env_overrides) or None if file doesn't exist
    """
    config_path = Path(".judge_backend")
    
    if not config_path.exists():
        return None
    
    try:
        content = config_path.read_text().strip()
        if not content:
            return None
        
        lines = content.splitlines()
        main_line = lines[0]
        env_overrides = None
        
        # Parse environment overrides if present
        if len(lines) > 1:
            for line in lines[1:]:
                if line.startswith("env:"):
                    import json
                    env_overrides = json.loads(line[4:])
                    break
        
        # Parse the main content
        if main_line == "ccode-4-sonnet":
            return ("ccode", "ccode-4-sonnet", env_overrides)
        elif ":" in main_line:
            backend_type, model_name = main_line.split(":", 1)
            if backend_type == "litellm":
                return (backend_type, model_name, env_overrides)
            else:
                logger.warning(f"Unknown backend type in .judge_backend: {backend_type}")
                return None
        else:
            logger.warning(f"Invalid format in .judge_backend: {main_line}")
            return None
            
    except Exception as e:
        logger.error(f"Error reading .judge_backend file: {e}")
        return None


def get_judge_backend(llm_judge_name: str) -> JudgeBackend:
    """Get the appropriate judge backend based on model name.
    
    First checks .judge_backend file, then falls back to the provided llm_judge_name.
    """
    global _backend_cache
    
    # Check if we need to refresh the cache
    current_time = time.time()
    if current_time - _backend_cache["last_check"] > _backend_cache["cache_duration"]:
        # Read from file
        config = _read_backend_config()
        
        if config:
            backend_type, model_name, env_overrides = config
            
            # Check if backend has changed
            if (backend_type != _backend_cache["backend_type"] or 
                model_name != _backend_cache["model_name"] or
                env_overrides != _backend_cache["env_overrides"]):
                logger.info(f"Switching judge backend from {_backend_cache['backend_type']}:{_backend_cache['model_name']} to {backend_type}:{model_name}")
                
                # Apply environment overrides if provided
                if env_overrides:
                    logger.info(f"Applying environment overrides for {backend_type}: {list(env_overrides.keys())}")
                    for key, value in env_overrides.items():
                        if value == "unset":
                            # "unset" means remove the variable
                            if key in os.environ:
                                del os.environ[key]
                                logger.debug(f"Unset environment variable: {key}")
                        else:
                            os.environ[key] = value
                            logger.debug(f"Set environment variable: {key}")
            
            _backend_cache["backend_type"] = backend_type
            _backend_cache["model_name"] = model_name
            _backend_cache["env_overrides"] = env_overrides
        else:
            # No file or invalid file, use provided name
            if llm_judge_name == "ccode-4-sonnet":
                _backend_cache["backend_type"] = "ccode"
                _backend_cache["model_name"] = llm_judge_name
            else:
                _backend_cache["backend_type"] = "litellm"
                _backend_cache["model_name"] = llm_judge_name
            _backend_cache["env_overrides"] = None
        
        _backend_cache["last_check"] = current_time
    
    # Return the appropriate backend based on cache
    if _backend_cache["backend_type"] == "ccode":
        logger.debug("Using CLI judge backend for ccode-4-sonnet model")
        return ClaudeCodeCLIJudgeBackend()
    else:
        logger.debug(f"Using LiteLLM judge backend for model: {_backend_cache['model_name']}")
        return LiteLLMJudgeBackend(_backend_cache["model_name"])



def _build_format_correction_message() -> str:
    """Build a message to correct invalid judge output format."""
    return """Your previous response could not be parsed. Please provide your evaluation in the EXACT format specified:

```yaml
thoughts: "[2-3 sentences analyzing: (1) action output & validity, (2) phase adherence & todo usage, (3) overall task progress. Note major violations like missing todos or skipped phases. Highlight any significant recovery from errors.]"
score: 0.00
```

CRITICAL REQUIREMENTS:
- The score MUST be a single number between 0.00 and 1.00 (two decimal places)
- The thoughts field should be 2-3 sentences as specified
- Use the exact YAML format shown above
- Do not include any other fields or text outside the YAML block

Please re-evaluate and provide your response in the correct format."""