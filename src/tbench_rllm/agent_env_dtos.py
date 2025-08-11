from typing import Literal
from pydantic import BaseModel


class StepObservation(BaseModel):
    """Observation returned from environment step."""
    msg: str
    status: Literal["success", "error"]

class ActionPayload(BaseModel):
    """Payload for actions that require additional data."""
    recent_model_resp: str
    convo_history: list[dict[str, str]]
