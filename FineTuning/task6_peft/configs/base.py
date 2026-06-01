from abc import ABC, abstractmethod
from custom_helpers import get_config
from typing import Dict, List
from peft import PeftConfig


class BaseConfig:
    """Base configuration class for PEFT model training"""
    DEFAULT_TRAINING_CONFIG = {
        # "output_dir": "./peft_prompt_tuning",  # intentionally commented out
        "learning_rates": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        "batch_size": 16,
        "num_epochs": 5,
        "eval_steps": 25
    }
    def __init__(self, conf_path: str):
        self.conf = get_config(conf_path)
        self._validate()
    
    def __check_dict(self, d: Dict, keys: List[str], config_name: str):
        """Check if all keys are present in the dictionary"""
        for key in keys:
            if key not in d:
                raise ValueError(f"Missing required key: {key} in {config_name}")
            
    def _validate(self):
        """Validate configuration parameters"""
        required_keys = ["model"]
        self.__check_dict(self.conf, required_keys, "self.config")

        # check model_config
        required_keys = ["model_name", "quantize", "is_prompt_tuning"]
        self.__check_dict(self.model_config, required_keys, "self.model_config")

        # check training_config
        required_keys = ["output_dir", "batch_size", "num_epochs", "eval_steps"]
        self.__check_dict(self.training_config, required_keys, "self.training_config")
    
    @property
    @abstractmethod
    def model_config(self) -> Dict:
        """Model loading configuration"""
        pass
    
    @property
    @abstractmethod
    def peft_config(self) -> PeftConfig:
        """Create PEFT configuration"""
        pass

    @property
    @abstractmethod
    def training_config(self) -> Dict:
        """Training process configuration"""
        pass
