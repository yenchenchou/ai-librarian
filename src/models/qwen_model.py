from typing import Dict, Optional

from .open_source_model import OpenSourceModel


class QwenModel(OpenSourceModel):
    """Qwen (通义千问) model implementation."""

    def __init__(
        self,
        model_name: str = "qwen-7b-chat",
        model_path: str = "Qwen/Qwen-7B-Chat",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(model_name, model_path, device, **kwargs)

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the Qwen model."""
        info = super().get_model_info()
        info.update(
            {
                "family": "Qwen",
                "license": "Tongyi Qianwen License",
                "paper": "https://arxiv.org/abs/2309.16609",
            }
        )
        return info
