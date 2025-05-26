from typing import Dict, Optional

from .open_source_model import OpenSourceModel


class LLaMAModel(OpenSourceModel):
    """LLaMA model implementation."""

    def __init__(
        self,
        model_name: str = "llama-2-7b-chat",
        model_path: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(model_name, model_path, device, **kwargs)

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the LLaMA model."""
        info = super().get_model_info()
        info.update(
            {
                "family": "LLaMA",
                "license": "Meta AI Research License",
                "paper": "https://arxiv.org/abs/2307.09288",
            }
        )
        return info
