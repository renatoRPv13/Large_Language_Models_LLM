from custom_helpers import get_config
from .base import BaseConfig
from peft import (
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    PeftConfig
)
from typing import Dict


class PeftPromptTuningConfig(BaseConfig):
    """Configuration class for PEFT model training"""
    @property
    def model_config(self) -> Dict:
        """Model loading configuration"""
        return {
            "model_name": self.conf["model"],
            "quantize": True,
            "is_prompt_tuning": True
        }

    @property
    def peft_config(self) -> PeftConfig:
        """Create PEFT configuration"""
        return PromptTuningConfig(
            # TODO: fill in the config
            tokenizer_name_or_path=self.conf["model"]
        )

    @property
    def training_config(self) -> Dict:
        """Training process configuration"""
        return {
            **self.DEFAULT_TRAINING_CONFIG,
            # "num_train_epochs": 3,
            # YOU CAN OVERWRITE DEFAULT PARAMS HERE
            "output_dir": self.conf.exp.prompt_tuning_dir
        }
