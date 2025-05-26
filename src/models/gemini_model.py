from typing import Dict, List, Optional, Union

import google.generativeai as genai

from .base_model import BaseAIModel


class GeminiModel(BaseAIModel):
    """Google's Gemini model implementation for library assistant."""

    def __init__(self, model_name: str = "gemini-pro", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        # Note: Gemini's pricing is not publicly available yet
        self.pricing = {"gemini-pro": {"input": 0.0, "output": 0.0}}  # Placeholder

    async def generate_response(
        self, prompt: str, context: Optional[List[str]] = None, **kwargs
    ) -> str:
        """Generate a response using Gemini's API."""
        # Combine context and prompt
        full_prompt = ""
        if context:
            full_prompt = "Context:\n" + "\n".join(context) + "\n\n"
        full_prompt += prompt

        response = await self.model.generate_content_async(full_prompt, **kwargs)

        return response.text

    async def create_study_plan(
        self, topic: str, available_resources: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """Create a study plan using Gemini's API."""
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
            "provider": "Google",
            "type": "commercial",
            "pricing": "Not publicly available",
        }
