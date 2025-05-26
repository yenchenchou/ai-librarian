from typing import Dict, List, Optional, Union

import openai

from .base_model import BaseAIModel


class OpenAIModel(BaseAIModel):
    """OpenAI model implementation for library assistant."""

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        if api_key:
            openai.api_key = api_key
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
        }

    async def generate_response(
        self, prompt: str, context: Optional[List[str]] = None, **kwargs
    ) -> str:
        """Generate a response using OpenAI's API."""
        messages = []

        # Add context if provided
        if context:
            for ctx in context:
                messages.append({"role": "system", "content": ctx})

        # Add the main prompt
        messages.append({"role": "user", "content": prompt})

        response = await openai.ChatCompletion.acreate(
            model=self.model_name, messages=messages, **kwargs
        )

        return response.choices[0].message.content

    async def create_study_plan(
        self, topic: str, available_resources: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """Create a study plan using OpenAI's API."""
        # Format resources for the prompt
        resources_str = "\n".join(
            [
                f"- {r['title']}: {r.get('description', 'No description')}"
                for r in available_resources
            ]
        )

        prompt = f"""Create a comprehensive study plan for {topic} using the following resources:
{resources_str}

Please structure the response as a JSON with the following fields:
- prerequisites: List of required knowledge
- learning_path: List of steps with estimated time
- resources: List of resources to use for each step
- timeline: Overall timeline with milestones
"""

        response = await self.generate_response(prompt, **kwargs)
        # TODO: Parse response into structured format
        return {"raw_response": response}

    def get_cost_estimate(self, prompt_length: int, response_length: int) -> float:
        """Estimate the cost based on token usage."""
        if self.model_name not in self.pricing:
            return 0.0

        pricing = self.pricing[self.model_name]
        input_cost = (prompt_length / 1000) * pricing["input"]
        output_cost = (response_length / 1000) * pricing["output"]

        return input_cost + output_cost

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the model."""
        return {
            "name": self.model_name,
            "provider": "OpenAI",
            "type": "commercial",
            "pricing": str(self.pricing.get(self.model_name, "Unknown")),
        }
