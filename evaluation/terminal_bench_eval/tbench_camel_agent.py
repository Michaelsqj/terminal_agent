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

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkitDocker
from camel.types import ModelPlatformType, ModelType

logger = logging.getLogger(__name__)

class TerminalBenchAgent(BaseAgent):

    def __init__(self, **kwargs):
        # Get current script directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Define workspace directory for the toolkit
        workspace_dir = os.path.join(
            os.path.dirname(os.path.dirname(base_dir)), "workspace"
        )
        print(f"Using workspace directory: {workspace_dir}")
        # Define system message
        sys_msg = (
            "You are a System Administrator helping with log management tasks. "
            "You have access to terminal tools that can help you execute "
            "shell commands and search files. "
        )

        # Set model config
        tools = TerminalToolkitDocker(working_directory=workspace_dir).get_tools()

        model_config_dict = ChatGPTConfig(
            temperature=0.0,
        ).as_dict()

        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=model_config_dict,
        )

        # Set agent
        camel_agent = ChatAgent(
            system_message=sys_msg,
            model=model,
            tools=tools,
        )
        camel_agent.reset()
        
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
        
        
        # Track timestamped markers for Terminal Bench
        timestamped_markers: List[Tuple[float, str]] = []
        
        # Initialize failure mode
        failure_mode = FailureMode.NONE
        

        usr_msg = f"{instruction}\n"

        # Get response information
        response = camel_agent.step(usr_msg)

        total_input_tokens = response.info['usage']['prompt_tokens']
        total_output_tokens = response.info['usage']['completion_tokens']
        
        memory_list = camel_agent._memory._chat_history_block.storage.memory_list

        def create_timestamped_marker_from_memory(records: List[dict]) -> Tuple[float, str]:
            """Create a timestamped marker from memory records."""
            results = []
            print(f"Total records: {len(records)}")
            for record in records:

                if 'func_name' in record['message'].keys():
                    timestamp = record['timestamp']
                    func_name = record['message']['func_name']
                    args = record['message'].get('args', {})
                    if args:
                        command = args.get('command', '')
                    else:
                        command = ''
                    results.append((timestamp, f"Called tool: {func_name} with args: {command}"))
            return results

        timestamped_markers = create_timestamped_marker_from_memory(memory_list)


        return AgentResult(
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            failure_mode=failure_mode,
            timestamped_markers=timestamped_markers,
        )
