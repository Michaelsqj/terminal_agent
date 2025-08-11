"""Terminal bench agent for RLLM."""

import logging
from typing import Dict, List, Any, Optional
from src.agent_core.load_sys_msg import load_sys_msg
from rllm.agents.agent import BaseAgent, Action, Trajectory, Step # pylint: disable=import-error # type: ignore
from src.tbench_rllm.agent_env_dtos import ActionPayload, StepObservation

logger = logging.getLogger(__name__)


class TerminalBenchAgent(BaseAgent):
    """Agent that interacts with terminal environments."""

    def __init__(
        self,
        agent_id: Optional[str] = None,
    ):
        super().__init__()
        self.agent_id = agent_id or "unknown"
        self.system_prompt = load_sys_msg()
        logger.debug(f"[{self.agent_id[:8]}] Agent initialized")

        # Initialize conversation with system prompt
        self.reset()

    def reset(self):
        """Reset agent state for new trajectory."""
        self._trajectory = Trajectory()
        self.convo_history = [{"role": "system", "content": self.system_prompt}]
        self.current_step = None
        self.step_count = 0
        logger.debug(f"[{self.agent_id[:8]}] Agent reset")

    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        **kwargs,
    ):
        """Update agent state based on environment feedback."""
        # Update previous step's outcome if it exists
        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info.update(info)
            # Store next observation in info if needed
            if observation is not None:
                prior_step.info["next_observation"] = observation

        # If episode is done, don't create a new step
        if done:
            if reward != 0:
                logger.info(f"[{self.agent_id[:8]}] Episode complete: reward={reward:.3f}")
            return

        observation = StepObservation.model_validate(observation)
        # Add to conversation history
        self.convo_history.append(
            {
                "role": "user",
                "content": observation.msg,
            }
        )
        
        # Log key observation events
        msg_preview = observation.msg[:100] + "..." if len(observation.msg) > 100 else observation.msg
        if self.step_count == 0:
            logger.info(f"[{self.agent_id[:8]}] Task: {msg_preview}")
        else:
            logger.debug(f"[{self.agent_id[:8]}] Obs: {observation.status}")

        # Create new step for this observation
        self.current_step = Step(
            observation=observation.msg, info={"step_number": self.step_count}
        )
        self._trajectory.steps.append(self.current_step)

    def update_from_model(self, response: str, **kwargs) -> Action:
        """Update agent state with model response."""
        assert self.current_step is not None, (
            "No current step to update with model response"
        )
        self.current_step.model_response = response

        # Add assistant response to conversation history
        self.convo_history.append({"role": "assistant", "content": response})

        self.step_count += 1
        
        # Log action generation
        action_preview = response[:80] + "..." if len(response) > 80 else response
        logger.debug(f"[{self.agent_id[:8]}] Action {self.step_count}: {action_preview.strip()}")

        payload = ActionPayload(
            recent_model_resp=response,
            convo_history=self.convo_history.copy(),  # Copy to avoid mutations
        ).model_dump()

        action = Action(action=payload)
        self.current_step.action = action
        return action

    def get_current_state(self) -> Optional[Step]:
        """Get current step state."""
        if not self.trajectory.steps:
            return None
        return self._trajectory.steps[-1]

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Return conversation history for model interaction."""
        return self.convo_history

    @property
    def trajectory(self) -> Trajectory:
        """Return complete interaction trajectory."""
        return self._trajectory
