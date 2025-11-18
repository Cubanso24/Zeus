"""
Configuration management utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ModelConfig:
    """Model configuration."""
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    model_max_length: int = 2048
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = False
    load_in_8bit: bool = False


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    output_dir: str = "models/splunk-query-llm"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    logging_dir: str = "logs"
    logging_strategy: str = "steps"
    logging_steps: int = 10
    report_to: list = field(default_factory=lambda: ["tensorboard"])
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    group_by_length: bool = True
    dataloader_num_workers: int = 4


@dataclass
class DataConfig:
    """Data configuration."""
    train_file: str = "data/processed/train_alpaca.jsonl"
    val_file: str = "data/processed/val_alpaca.jsonl"
    test_file: str = "data/processed/test_alpaca.jsonl"
    format_type: str = "alpaca"
    max_seq_length: int = 2048


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    enabled: bool = False
    project: str = "splunk-query-llm"
    entity: Optional[str] = None
    name: Optional[str] = None


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    system_prompt: str = ""
    seed: int = 42


class ConfigLoader:
    """Load and manage configuration."""

    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config_dict

    @staticmethod
    def dict_to_config(config_dict: Dict[str, Any]) -> Config:
        """
        Convert dictionary to Config object.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config object
        """
        config = Config()

        # Parse model config
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])

        # Parse LoRA config
        if 'lora' in config_dict:
            config.lora = LoRAConfig(**config_dict['lora'])

        # Parse training config
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])

        # Parse data config
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])

        # Parse wandb config
        if 'wandb' in config_dict:
            config.wandb = WandbConfig(**config_dict['wandb'])

        # Other settings
        config.system_prompt = config_dict.get('system_prompt', '')
        config.seed = config_dict.get('seed', 42)

        return config

    @classmethod
    def load(cls, config_path: str) -> Config:
        """
        Load configuration from file.

        Args:
            config_path: Path to config file

        Returns:
            Config object
        """
        config_dict = cls.load_yaml(config_path)
        return cls.dict_to_config(config_dict)

    @staticmethod
    def save(config: Config, output_path: str):
        """
        Save configuration to YAML file.

        Args:
            config: Config object
            output_path: Path to save config
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary
        config_dict = {
            'model': config.model.__dict__,
            'lora': config.lora.__dict__,
            'training': config.training.__dict__,
            'data': config.data.__dict__,
            'wandb': config.wandb.__dict__,
            'system_prompt': config.system_prompt,
            'seed': config.seed,
        }

        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved configuration to {output_path}")
