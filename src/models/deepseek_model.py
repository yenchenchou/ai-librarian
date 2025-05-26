from typing import Dict, Optional

from .open_source_model import OpenSourceModel


class DeepSeekModel(OpenSourceModel):
    """DeepSeek model implementation."""

    def __init__(
        self,
        model_name: str = "deepseek-7b-chat",
        model_path: str = "deepseek-ai/deepseek-llm-7b-chat",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(model_name, model_path, device, **kwargs)

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the DeepSeek model."""
        info = super().get_model_info()
        info.update(
            {
                "family": "DeepSeek",
                "license": "DeepSeek License",
                "paper": "https://arxiv.org/abs/2401.02954",
            }
        )
        return info
