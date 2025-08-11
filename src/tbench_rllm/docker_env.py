"""Docker-isolated environment for RLLM training."""

import asyncio
import logging
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.terminal.terminal import Terminal

from rllm.environments.base.base_env import BaseEnv # pylint: disable=import-error # type: ignore
from src.agent_core.env_components.action_handler import ActionHandler
from src.agent_core.env_components.action_parser import ActionParserXMLYaml
from src.agent_core.env_components.actions import BashAction, FinishAction
from src.agent_core.env_components.state_managers import ScratchpadManager, TodoManager
from src.agent_core.env_components.command_executor import DockerExecutor
from src.agent_core.env_components.step_executor import StepExecutor, StepResult
from src.tbench_rllm.rewards.calculate_reward import calculate_reward
from src.tbench_rllm.agent_env_dtos import ActionPayload, StepObservation

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for a specific terminal bench task."""
    task_name: str
    task_path: str
    instruction: str
    test_weights: Dict[str, float]
    dockerfile_contents: str
    py_test_file_contents: str
    max_test_timeout_sec: float = 300.0


@dataclass
class ContainerConfig:
    """Configuration for Docker container runtime settings."""
    no_rebuild: bool = False
    timeout: int = 600  # seconds


class DockerIsolatedEnv(BaseEnv):
    """Environment that runs each trajectory in an isolated Docker container."""
    
    def __init__(
        self,
        task_config: TaskConfig,
        container_config: ContainerConfig,
        env_id: Optional[str] = None,
    ):
        self.task_config = task_config
        self.container_config = container_config
        self.env_id = env_id or str(uuid.uuid4())
        
        # Terminal bench specific
        self.trial_handler = None
        self.terminal = None
        self.session = None
        
        # Track state
        self.done = False
        self.steps_taken = 0
        
        # Action handler components
        self.action_parser = None
        self.action_handler = None
        self.step_executor = None
        
    def reset(self) -> Tuple[Dict, Dict]:
        """Reset environment and start a new Docker container."""
        logger.info(f"[{self.env_id[:8]}] Reset for task: {self.task_config.task_name}")
        
        # Clean up any existing container
        if self.terminal:
            self._cleanup()
        
        # Initialize trial handler
        self.trial_handler = TrialHandler(
            trial_name=f"rllm_{self.env_id}_{self.task_config.task_name}",
            input_path=Path(self.task_config.task_path),
            output_path=Path(f"/tmp/rllm_docker_output/{self.env_id}/"),
        )
        
        # Start terminal/container
        self.terminal = Terminal(
            client_container_name=self.trial_handler.client_container_name,
            client_image_name=self.trial_handler.client_image_name,
            docker_compose_path=self.trial_handler.task_paths.docker_compose_path,
            sessions_path=self.trial_handler.trial_paths.sessions_path,
            commands_path=self.trial_handler.trial_paths.commands_path,
            no_rebuild=self.container_config.no_rebuild,
            cleanup=False,  # We'll handle cleanup ourselves
        )
        
        try:
            self.terminal.start()
            logger.debug(f"[{self.env_id[:8]}] Container started: {self.trial_handler.client_container_name}")
        except Exception as e:
            logger.error(f"[{self.env_id[:8]}] Container start failed: {e}")
            raise
        
        # Get or create session
        try:
            self.session = self.terminal.get_session("agent")
        except ValueError:
            self.session = self.terminal.create_session("agent")
        
        # Initialize action handling components
        self._initialize_components()
        
        # Reset state
        self.done = False
        self.steps_taken = 0
        
        # Capture pre-agent pane
        self._capture_pre_agent_pane()
        
        step_observation = StepObservation(
            msg=self.task_config.instruction,
            status="success"
        )

        return step_observation.model_dump(), {}
    
    def step(self, action: Dict) -> Tuple[Any, float, bool, Dict]:        
        if self.done:
            logger.warning(f"[{self.env_id[:8]}] Step on completed env")
            return {}, 0.0, True, {"error": "Environment is done"}
        
        logger.debug(f"[{self.env_id[:8]}] Step {self.steps_taken + 1}")

        # RLLM execution engine passes action.action (dict), not Action object
        action_payload = ActionPayload.model_validate(action)
        if not action_payload or not action_payload.recent_model_resp:
            raise ValueError(f"[{self.env_id[:8]}] Invalid action payload: {action}")
        
        observation = self._execute_action(action_payload.recent_model_resp)
        self.steps_taken += 1

        reward = 0.0
        info = {}
        if self.done:
            # Capture post-agent pane before calculating reward
            self._capture_post_agent_pane()
            
            try:
                reward_results = asyncio.run(
                    calculate_reward(
                        terminal=self.terminal, # type: ignore
                        trial_handler=self.trial_handler, # type: ignore
                        task_name=self.task_config.task_name,
                        test_weights=self.task_config.test_weights,
                        dockerfile_contents=self.task_config.dockerfile_contents,
                        max_test_timeout_sec=self.task_config.max_test_timeout_sec,
                        messages=action_payload.convo_history,
                        rollout_id=self.env_id,
                    )
                )
                reward = reward_results.final_weighted_reward
                info["reward_details"] = {
                    "test_score": reward_results.raw_test_score,
                    "judge_score": reward_results.raw_judge_score,
                    "final_reward": reward_results.final_weighted_reward,
                }
                logger.info(f"[{self.env_id[:8]}] Complete: reward={reward:.3f}, test={reward_results.raw_test_score:.3f}, judge={reward_results.raw_judge_score:.3f}")
            except Exception as e:
                logger.error(f"[{self.env_id[:8]}] Reward calc failed: {e}")
                reward = 0.0
                info["reward_error"] = str(e)
        
        return observation.model_dump(), reward, self.done, info
    
    def close(self):
        """Clean up Docker container and resources."""
        logger.debug(f"[{self.env_id[:8]}] Closing")
        self._cleanup()
    
    
    @staticmethod
    def from_dict(info: Dict) -> "DockerIsolatedEnv":
        """Create environment from configuration dict.
        
        Expects flattened structure from dataset extra_info plus command line overrides:
        - From dataset: task_name, task_path, instruction, test_weights, etc.
        - From command line: max_steps, no_rebuild, timeout (via env.env_args)
        """
        # Create TaskConfig from dataset fields
        task_config = TaskConfig(
            task_name=info['task_name'],
            task_path=info['task_path'],
            instruction=info['instruction'],
            test_weights=info['test_weights'],
            dockerfile_contents=info['dockerfile_contents'],
            py_test_file_contents=info['py_test_file_contents'],
            max_test_timeout_sec=info.get('max_test_timeout_sec', 300.0),
        )
        
        # Create ContainerConfig from command line overrides (or defaults)
        container_config = ContainerConfig(
            no_rebuild=info.get('no_rebuild', False),
            timeout=info.get('timeout', 600),
        )
        
        return DockerIsolatedEnv(
            task_config=task_config,
            container_config=container_config,
            env_id=info.get('env_id')
        )
    
    @staticmethod
    def is_multithread_safe() -> bool:
        """Each env has its own container, so thread-safe."""
        return True
    
    def _initialize_components(self):
        """Initialize action parsing and handling components."""
        # Initialize components
        self.action_parser = ActionParserXMLYaml()
        self.todo_manager = TodoManager()
        self.scratchpad_manager = ScratchpadManager()
        
        # Get container name for action handler
        if not self.terminal or not self.terminal.container:
            logger.error(f"[{self.env_id[:8]}] Terminal/container not initialized")
            raise RuntimeError("Terminal or container not initialized")
        container_name = self.terminal.container.name if self.terminal.container else None
        if not container_name:
            raise RuntimeError("Container not initialized")
            
        executor = DockerExecutor(container_name)
        
        self.action_handler = ActionHandler(
            executor=executor,
            todo_manager=self.todo_manager,
            scratchpad_manager=self.scratchpad_manager,
        )
        
        # Create step executor with callbacks
        self.step_executor = StepExecutor(
            action_parser=self.action_parser,
            action_handler=self.action_handler,
            max_consecutive_parse_errors=3,
            on_no_action=lambda: self._handle_no_action(),
            on_parse_error_limit=lambda: self._handle_parse_error_limit(),
            on_finish_action=lambda action: self._handle_finish_action(action),
            on_bash_success=lambda action: self._handle_bash_success(action),
        )
    
    def _handle_no_action(self):
        """Handle when no action is attempted."""
        self.done = True
        logger.info(f"[{self.env_id[:8]}] Episode done: no actions attempted")
    
    def _handle_parse_error_limit(self):
        """Handle when parse error limit is reached."""
        self.done = True
        logger.info(f"[{self.env_id[:8]}] Episode done: 3 consecutive parsing errors")
    
    def _handle_finish_action(self, action: FinishAction):
        """Handle finish action."""
        self.done = True
        logger.info(f"[{self.env_id[:8]}] Episode done: finish action called - {action.message}")
    
    def _handle_bash_success(self, action: BashAction):
        """Handle successful bash action with end_after_cmd_success."""
        self.done = True
        logger.info(f"[{self.env_id[:8]}] Episode done: success cmd executed")
    
    def _execute_action(self, action: str) -> StepObservation:
        """Execute the action and return observation."""
        # Execute the step
        step_result = asyncio.run(self.step_executor.execute_step(action))
        
        # Determine status based on result
        status = "error" if step_result.has_error else "success"
        
        # Handle specific results
        msg = "\n\n".join(step_result.responses) if step_result.responses else "No actions parsed"
        
        if step_result.result == StepResult.NO_ACTION:
            msg = "No actions were attempted. The trajectory has ended."
            status = "error"
        elif step_result.result == StepResult.PARSE_ERROR_LIMIT:
            # Message already includes the error about consecutive parsing errors
            status = "error"
        
        return StepObservation(
            msg=msg,
            status=status,
        )
    
    def _cleanup(self):
        """Cleanup Docker container and resources asynchronously."""
        container_name = None
        try:
            if self.terminal:
                # Store container name before stopping
                if self.terminal.container:
                    container_name = self.terminal.container.name
                
                logger.debug(f"[{self.env_id[:8]}] Stopping terminal")
                self.terminal.stop()
                logger.info(f"[{self.env_id[:8]}] Terminal stopped")
                self.terminal = None
                self.session = None
                
                # Verify container is actually stopped/removed
                if container_name:
                    self._verify_container_stopped(container_name)
        except Exception as e:
            logger.debug(f"[{self.env_id[:8]}] Cleanup error (non-critical): {e}")
    
    def _verify_container_stopped(self, container_name: str):
        """Verify Docker container is stopped/removed, force stop if necessary."""
        try:
            # Check if container exists
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if container_name in result.stdout:
                logger.warning(f"[{self.env_id[:8]}] Container {container_name} still exists after cleanup")
                
                # Check if it's still running
                running_result = subprocess.run(
                    ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if container_name in running_result.stdout:
                    logger.warning(f"[{self.env_id[:8]}] Container {container_name} is still running - forcing stop")
                    # Force stop the container
                    subprocess.run(
                        ["docker", "stop", "-t", "0", container_name],
                        capture_output=True,
                        timeout=10
                    )
                    logger.info(f"[{self.env_id[:8]}] Force stopped container {container_name}")
                
                # Remove the container
                logger.debug(f"[{self.env_id[:8]}] Removing container {container_name}")
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                    timeout=10
                )
                logger.info(f"[{self.env_id[:8]}] Removed container {container_name}")
        except subprocess.TimeoutExpired:
            logger.error(f"[{self.env_id[:8]}] Timeout while verifying/stopping container {container_name}")
        except Exception as e:
            logger.error(f"[{self.env_id[:8]}] Error verifying container cleanup: {e}")
    
    def _capture_pre_agent_pane(self):
        """Capture the terminal pane before agent starts."""
        try:
            if self.session:
                pane_content = self.session.capture_pane(capture_entire=True)
                # Ensure directory exists
                self.trial_handler.trial_paths.pre_agent_pane_path.parent.mkdir(parents=True, exist_ok=True)
                self.trial_handler.trial_paths.pre_agent_pane_path.write_text(pane_content)
                logger.debug(f"[{self.env_id[:8]}] Captured pre-agent pane")
        except Exception as e:
            logger.warning(f"[{self.env_id[:8]}] Failed to capture pre-agent pane: {e}")
            # Write empty file to prevent downstream errors
            try:
                self.trial_handler.trial_paths.pre_agent_pane_path.parent.mkdir(parents=True, exist_ok=True)
                self.trial_handler.trial_paths.pre_agent_pane_path.write_text("")
            except Exception:
                pass
    
    def _capture_post_agent_pane(self):
        """Capture the terminal pane after agent completes."""
        try:
            if self.session:
                pane_content = self.session.capture_pane(capture_entire=True)
                # Ensure directory exists
                self.trial_handler.trial_paths.post_agent_pane_path.parent.mkdir(parents=True, exist_ok=True)
                self.trial_handler.trial_paths.post_agent_pane_path.write_text(pane_content)
                logger.debug(f"[{self.env_id[:8]}] Captured post-agent pane")
        except Exception as e:
            logger.warning(f"[{self.env_id[:8]}] Failed to capture post-agent pane: {e}")
            # Write empty file to prevent downstream errors
            try:
                self.trial_handler.trial_paths.post_agent_pane_path.parent.mkdir(parents=True, exist_ok=True)
                self.trial_handler.trial_paths.post_agent_pane_path.write_text("")
            except Exception:
                pass