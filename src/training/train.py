"""
Fine-tuning script for Splunk Query LLM.
"""

import os
import sys
from pathlib import Path
import torch
import click
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset
import transformers

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import ConfigLoader
from src.training.data_collator import SplunkQueryDataCollator


def setup_model_and_tokenizer(config):
    """
    Load and configure model and tokenizer.

    Args:
        config: Configuration object

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {config.model.base_model}")

    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model,
        trust_remote_code=True,
        padding_side="right",
        model_max_length=config.model.model_max_length,
    )

    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure quantization if needed
    quantization_config = None
    if config.model.load_in_4bit or config.model.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.model.load_in_4bit,
            load_in_8bit=config.model.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.model.torch_dtype == "bfloat16" else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if config.model.torch_dtype == "bfloat16" else torch.float16,
    )

    # Prepare model for training
    if config.model.load_in_4bit or config.model.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA if enabled
    if config.lora.enabled:
        logger.info("Applying LoRA configuration")
        peft_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=config.lora.target_modules,
            bias=config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def load_and_prepare_data(config, tokenizer):
    """
    Load and prepare training data.

    Args:
        config: Configuration object
        tokenizer: Tokenizer instance

    Returns:
        Dictionary with train and validation datasets
    """
    logger.info("Loading datasets")

    # Load datasets
    data_files = {
        "train": config.data.train_file,
        "validation": config.data.val_file,
    }

    datasets = load_dataset("json", data_files=data_files)

    # Tokenization function
    def tokenize_function(examples):
        """Tokenize examples based on format type."""
        if config.data.format_type == "alpaca":
            # Format: instruction + input + output
            prompts = []
            for i in range(len(examples["instruction"])):
                instruction = examples["instruction"][i]
                input_text = examples.get("input", [""])[i]
                output = examples["output"][i]

                # Build prompt
                if input_text:
                    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

                prompts.append(prompt)

            # Tokenize
            tokenized = tokenizer(
                prompts,
                truncation=True,
                max_length=config.data.max_seq_length,
                padding=False,
            )

            # Set labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        elif config.data.format_type == "chat":
            # Chat format with messages
            # This would need to be implemented based on specific chat format
            raise NotImplementedError("Chat format not yet implemented")

        else:
            raise ValueError(f"Unknown format type: {config.data.format_type}")

    # Tokenize datasets
    logger.info("Tokenizing datasets")
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing",
    )

    return tokenized_datasets


@click.command()
@click.option('--config', default='configs/training_config.yaml', help='Path to training config')
@click.option('--resume-from-checkpoint', default=None, help='Path to checkpoint to resume from')
@click.option('--output-dir', default=None, help='Override output directory')
def train(config: str, resume_from_checkpoint: str, output_dir: str):
    """
    Fine-tune Splunk Query LLM.

    Args:
        config: Path to configuration file
        resume_from_checkpoint: Optional checkpoint to resume from
        output_dir: Optional output directory override
    """
    # Load configuration
    logger.info("Loading configuration")
    cfg = ConfigLoader.load(config)

    # Override output dir if specified
    if output_dir:
        cfg.training.output_dir = output_dir

    # Set random seed
    transformers.set_seed(cfg.seed)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(cfg)

    # Load and prepare data
    tokenized_datasets = load_and_prepare_data(cfg, tokenizer)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        max_grad_norm=cfg.training.max_grad_norm,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        optim=cfg.training.optim,
        evaluation_strategy=cfg.training.evaluation_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        logging_dir=cfg.training.logging_dir,
        logging_strategy=cfg.training.logging_strategy,
        logging_steps=cfg.training.logging_steps,
        report_to=cfg.training.report_to,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        group_by_length=cfg.training.group_by_length,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        remove_unused_columns=False,
    )

    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    logger.info(f"Saving final model to {cfg.training.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(cfg.training.output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("Training complete!")

    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {cfg.training.output_dir}")
    print(f"Training loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Training steps: {metrics.get('train_steps', 'N/A')}")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    train()


if __name__ == "__main__":
    main()
