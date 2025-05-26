from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_model import BaseAIModel


class OpenSourceModel(BaseAIModel):
    """Base class for open-source language models."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        super().__init__(model_name)
        self.device = device
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model(**kwargs)

    def _load_model(self, **kwargs):
        """Load the model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")

    async def generate_response(
        self, prompt: str, context: Optional[List[str]] = None, **kwargs
    ) -> str:
        """Generate a response using the open-source model."""
        # Combine context and prompt
        full_prompt = ""
        if context:
            full_prompt = "Context:\n" + "\n".join(context) + "\n\n"
        full_prompt += prompt

        # Tokenize input
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                do_sample=kwargs.get("do_sample", True),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and return response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return response

    async def create_study_plan(
        self, topic: str, available_resources: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """Create a study plan using the open-source model."""
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
        """Estimate the cost (0 for open-source models)."""
        return 0.0  # Open-source models have no API costs

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the model."""
        return {
            "name": self.model_name,
            "provider": "Open Source",
            "type": "open_source",
            "model_path": self.model_path,
            "device": self.device,
        }
