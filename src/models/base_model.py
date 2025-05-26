from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union


class BaseAIModel(ABC):
    """Base class for all AI models used in the library assistant."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    async def generate_response(
        self, prompt: str, context: Optional[List[str]] = None, **kwargs
    ) -> str:
        """Generate a response for the given prompt."""
        pass

    @abstractmethod
    async def create_study_plan(
        self, topic: str, available_resources: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """Create a study plan using available resources."""
        pass

    @abstractmethod
    def get_cost_estimate(self, prompt_length: int, response_length: int) -> float:
        """Estimate the cost of generating a response."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the model."""
        pass
