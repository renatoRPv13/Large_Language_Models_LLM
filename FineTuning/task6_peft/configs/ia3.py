from custom_helpers import get_config
from .base import BaseConfig
from peft import (
    IA3Config,
    TaskType,
    PeftConfig
)
from typing import Dict


class PeftIA3Config(BaseConfig):
    """Configuration class for IA3 PEFT model training."""
    
    @property
    def model_config(self) -> Dict:
        return {
            "model_name": self.conf["model"],
            "quantize": False,
            "is_prompt_tuning": False
        }
    
    @property
    def peft_config(self) -> PeftConfig:
        return IA3Config(
            # TODO: fill in the config
            inference_mode=False
        )

    @property
    def training_config(self) -> Dict:
        return {
            **self.DEFAULT_TRAINING_CONFIG,
            "output_dir": self.conf.exp.ia3_dir
        }
