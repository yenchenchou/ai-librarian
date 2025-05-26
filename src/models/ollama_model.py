import json
from typing import Dict, List, Optional, Union

import aiohttp

from .base_model import BaseAIModel


class OllamaModel(BaseAIModel):
    """Ollama model implementation for local and API-based models."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, api_key)
        self.base_url = base_url.rstrip("/")
        self.session = None

        # Model configurations
        self.model_configs = {
            "llama2": {
                "family": "LLaMA",
                "license": "Meta AI Research License",
                "paper": "https://arxiv.org/abs/2307.09288",
            },
            "qwen": {
                "family": "Qwen",
                "license": "Tongyi Qianwen License",
                "paper": "https://arxiv.org/abs/2309.16609",
            },
            "deepseek": {
                "family": "DeepSeek",
                "license": "DeepSeek License",
                "paper": "https://arxiv.org/abs/2401.02954",
            },
            "mistral": {
                "family": "Mistral",
                "license": "Apache 2.0",
                "paper": "https://arxiv.org/abs/2310.06825",
            },
        }

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers=(
                    {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                )
            )

    async def generate_response(
        self, prompt: str, context: Optional[List[str]] = None, **kwargs
    ) -> str:
        """Generate a response using Ollama's API."""
        await self._ensure_session()

        # Combine context and prompt
        full_prompt = ""
        if context:
            full_prompt = "Context:\n" + "\n".join(context) + "\n\n"
        full_prompt += prompt

        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            **kwargs,
        }

        async with self.session.post(
            f"{self.base_url}/api/generate", json=payload
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Ollama API error: {await response.text()}")

            result = await response.json()
            return result["response"]

    async def create_study_plan(
        self, topic: str, available_resources: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """Create a study plan using Ollama."""
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
        """Estimate the cost (0 for local models)."""
        return 0.0  # Local models have no API costs

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the model."""
        base_info = {
            "name": self.model_name,
            "provider": "Ollama",
            "type": "local",
            "base_url": self.base_url,
        }

        # Add model-specific information if available
        model_type = self.model_name.split("-")[0].lower()  # Extract base model name
        if model_type in self.model_configs:
            base_info.update(self.model_configs[model_type])

        return base_info

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
