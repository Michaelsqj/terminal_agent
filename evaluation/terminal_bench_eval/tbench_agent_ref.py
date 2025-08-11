import os
import time
import asyncio
import logging
import json
from pathlib import Path
from typing import List, Tuple
from terminal_bench.agents.base_agent import BaseAgent, AgentResult
from terminal_bench.harness_models import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession
from terminal_bench.llms.chat import Chat
from terminal_bench.llms.lite_llm import LiteLLM
from src.agent_core.load_sys_msg import load_sys_msg

from src.agent_core.env_components.command_executor import DockerExecutor
from src.agent_core.env_components.action_parser import ActionParserXMLYaml
from src.agent_core.env_components.state_managers import TodoManager, ScratchpadManager
from src.agent_core.env_components.action_handler import ActionHandler
from src.agent_core.env_components.step_executor import StepExecutor, StepResult


logger = logging.getLogger(__name__)

class TerminalBenchAgent(BaseAgent):

    def __init__(self, **kwargs):
        self.sys_msg = load_sys_msg()
        
        self.model_name = os.getenv("LITELLM_MODEL", None)
        if not self.model_name:
            raise ValueError("LITELLM_MODEL environment variable must be set")
        self.temperature = float(os.getenv("LITELLM_TEMPERATURE", "0.1"))
        
        self._llm = LiteLLM(model_name=self.model_name, temperature=self.temperature,)
        
        super().__init__(**kwargs)

    
    @staticmethod
    def name() -> str:
        return "TerminalBenchAgent"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        """Execute a task using the Terminal Bench harness.
        
        Args:
            instruction: The task instruction to execute
            session: TmuxSession object for command execution
            logging_dir: Optional directory for logging
            
        Returns:
            AgentResult with token counts and failure mode
        """

        container_name = session.container.name
        if not container_name:
            raise ValueError("Container name is required for DockerExecutor")
        executor = DockerExecutor(container_name=container_name)

        if logging_dir:
            logging_dir = Path(logging_dir)
            logging_dir.mkdir(exist_ok=True, parents=True)
            
            # Create a unique log file name with timestamp
            log_file = logging_dir / f"agent_conversation_{int(time.time())}.json"
            logger.info(f"Logging conversation to: {log_file}")
        else:
            log_file = None
        
        
        
        action_parser = ActionParserXMLYaml()
        todo_manager = TodoManager()
        scratchpad_manager = ScratchpadManager()
        action_handler = ActionHandler(
            executor=executor,
            todo_manager=todo_manager,
            scratchpad_manager=scratchpad_manager
        )
        
        # Track timestamped markers for Terminal Bench
        timestamped_markers: List[Tuple[float, str]] = []
        
        # Initialize failure mode
        failure_mode = FailureMode.NONE
        
        # Create step executor with callbacks
        step_executor = StepExecutor(
            action_parser=action_parser,
            action_handler=action_handler,
            max_consecutive_parse_errors=3,
            on_no_action=lambda: logger.info("Episode done: no actions attempted"),
            on_parse_error_limit=lambda: logger.info("Episode done: parse error limit reached"),
            on_finish_action=lambda action: timestamped_markers.append((time.time(), "FINISH")),
            on_bash_success=lambda action: logger.info("Episode done: success cmd executed"),
        )
        
        # Initialize Chat instance
        chat = Chat(self._llm)
        
        # Add system message to chat history
        chat._messages.append({"role": "system", "content": self.sys_msg})
        
        current_prompt = instruction
        
        step = 0
        while True:
            step += 1
            try:
                log_path = None
                if logging_dir:
                    log_path = logging_dir / f"step_{step}_debug.json"
                    
                assistant_response = chat.chat(
                    prompt=current_prompt,
                    logging_path=log_path,
                    api_key=os.getenv("LITE_LLM_API_KEY"),
                    api_base=os.getenv("LITE_LLM_API_BASE"),
                )
                
                if not assistant_response:
                    raise ValueError("Received empty response from the model")
                
                # Execute the step synchronously
                step_result = asyncio.run(step_executor.execute_step(assistant_response))
                
                # Prepare next prompt from responses
                if step_result.responses:
                    current_prompt = "\n\n".join(step_result.responses)
                else:
                    current_prompt = ""
                
                # Handle different step results
                if step_result.result == StepResult.NO_ACTION or step_result.result == StepResult.FINISH:
                    failure_mode = FailureMode.NONE
                    break
                elif step_result.result == StepResult.PARSE_ERROR_LIMIT:
                    failure_mode = FailureMode.PARSE_ERROR
                    break
                elif step_result.result == StepResult.ERROR and step_result.has_error:
                    # Continue on error, don't break
                    pass
                    
            except asyncio.TimeoutError:
                # Handle timeout from tmux operations
                error_msg = f"Timeout error in step {step}"
                current_prompt = f"<error>\n{error_msg}\n</error>"
                failure_mode = FailureMode.AGENT_TIMEOUT
                break
            except Exception as e:
                # Handle unexpected errors
                error_msg = f"Unexpected error in step {step}: {type(e).__name__}: {str(e)}"
                logger.exception(f"Error in step {step}")
                current_prompt = f"<error>\n{error_msg}\n</error>"
                
                # Check if it's an API error
                if "api" in str(e).lower() or "rate" in str(e).lower():
                    failure_mode = FailureMode.UNKNOWN_AGENT_ERROR
                else:
                    failure_mode = FailureMode.UNKNOWN_AGENT_ERROR
                break
        
        if log_file:
            messages = chat._messages if hasattr(chat, '_messages') else []
            data = {
                "messages": messages,
                "total_input_tokens": chat.total_input_tokens,
                "total_output_tokens": chat.total_output_tokens,
                "failure_mode": failure_mode.value,
                "timestamped_markers": timestamped_markers,
                "num_steps": step,
            }
            # Get path
            path = log_file.resolve().parent
            with open(path / 'conversation_log.json', 'w') as f:
                json.dump(data, f, indent=2)

        return AgentResult(
            total_input_tokens=chat.total_input_tokens,
            total_output_tokens=chat.total_output_tokens,
            failure_mode=failure_mode,
            timestamped_markers=timestamped_markers,
        )
