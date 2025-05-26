import os
from typing import Dict, Optional, Union

from .gemini_model import GeminiModel
from .ollama_model import OllamaModel
from .openai_model import OpenAIModel


class ModelFactory:
    """Factory for creating AI model instances."""

    @staticmethod
    def create_model(
        model_type: str, model_name: Optional[str] = None, **kwargs
    ) -> Union[OpenAIModel, GeminiModel, OllamaModel]:
        """
        Create a model instance based on the type.

        Args:
            model_type: Type of model ('openai', 'gemini', 'ollama')
            model_name: Specific model name (optional)
            **kwargs: Additional arguments for model initialization
        """
        model_type = model_type.lower()

        if model_type == "openai":
            api_key = kwargs.pop("api_key", os.getenv("OPENAI_API_KEY"))
            return OpenAIModel(
                model_name=model_name or "gpt-4", api_key=api_key, **kwargs
            )

        elif model_type == "gemini":
            api_key = kwargs.pop("api_key", os.getenv("GOOGLE_API_KEY"))
            return GeminiModel(
                model_name=model_name or "gemini-pro", api_key=api_key, **kwargs
            )

        elif model_type == "ollama":
            base_url = kwargs.pop("base_url", "http://localhost:11434")
            return OllamaModel(
                model_name=model_name or "llama2", base_url=base_url, **kwargs
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_available_models() -> Dict[str, list]:
        """Get list of available models for each type."""
        return {
            "openai": ["gpt-4", "gpt-3.5-turbo"],
            "gemini": ["gemini-pro"],
            "ollama": [
                "llama2",
                "llama2:13b",
                "llama2:70b",
                "qwen",
                "deepseek",
                "mistral",
                "mixtral",
            ],
        }
