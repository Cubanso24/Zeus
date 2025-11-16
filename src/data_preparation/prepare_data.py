"""
Main data preparation pipeline for Splunk query training data.
"""

import json
from pathlib import Path
from typing import Optional
import click
from loguru import logger

from src.data_preparation.data_loader import SplunkQueryDataLoader
from src.data_preparation.data_validator import SplunkQueryValidator
from src.data_preparation.data_formatter import DataFormatter


@click.command()
@click.option('--input-dir', default='data/raw', help='Input directory with raw JSONL files')
@click.option('--output-dir', default='data/processed', help='Output directory for processed data')
@click.option('--format-type', default='alpaca', type=click.Choice(['alpaca', 'chat', 'completion']),
              help='Output format type')
@click.option('--train-ratio', default=0.8, help='Training set ratio')
@click.option('--val-ratio', default=0.1, help='Validation set ratio')
@click.option('--test-ratio', default=0.1, help='Test set ratio')
@click.option('--validate-only', is_flag=True, help='Only validate data without processing')
@click.option('--random-seed', default=42, help='Random seed for reproducibility')
def prepare_data(
    input_dir: str,
    output_dir: str,
    format_type: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    validate_only: bool,
    random_seed: int
):
    """
    Prepare Splunk query training data for fine-tuning.

    This script:
    1. Loads raw JSONL training data
    2. Validates data quality
    3. Splits into train/val/test sets
    4. Formats data for specified framework
    5. Saves processed data
    """
    logger.info("Starting data preparation pipeline")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize components
    loader = SplunkQueryDataLoader(input_dir)
    validator = SplunkQueryValidator()
    formatter = DataFormatter()

    # Load data
    logger.info("Loading training data...")
    examples = loader.load_all_data(pattern="train_*.jsonl")

    if not examples:
        logger.error("No training examples found!")
        return

    # Validate data
    logger.info("Validating data quality...")
    validation_results = validator.validate_dataset(examples)

    # Print validation summary
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Total examples: {validation_results['total_examples']}")
    print(f"Valid examples: {validation_results['valid_examples']}")
    print(f"Invalid examples: {validation_results['invalid_examples']}")
    print(f"Validation rate: {validation_results['validation_rate']:.1f}%")
    print(f"\nQuery examples: {validation_results['query_examples']}")
    print(f"Clarification examples: {validation_results['clarification_examples']}")
    print(f"Clarification rate: {validation_results['clarification_rate']:.1f}%")
    print("="*60 + "\n")

    # Get dataset statistics
    logger.info("Calculating dataset statistics...")
    stats = validator.get_dataset_statistics(examples)

    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total examples: {stats['total_examples']}")
    print(f"Unique instructions: {stats['unique_instructions']}")
    print(f"Unique outputs: {stats['unique_outputs']}")
    print(f"\nInstruction length: avg={stats['avg_instruction_length']:.1f}, "
          f"min={stats['min_instruction_length']}, max={stats['max_instruction_length']}")
    print(f"Output length: avg={stats['avg_output_length']:.1f}, "
          f"min={stats['min_output_length']}, max={stats['max_output_length']}")
    print(f"\nTop 10 SPL commands:")
    for cmd, count in stats['common_commands'].items():
        print(f"  {cmd}: {count}")
    print("="*60 + "\n")

    if validate_only:
        logger.info("Validation only mode - exiting")
        return

    # Filter out invalid examples
    valid_examples = []
    for example in examples:
        is_valid, _ = validator.validate_example(example)
        if is_valid:
            valid_examples.append(example)

    logger.info(f"Using {len(valid_examples)} valid examples for training")

    # Split data
    logger.info("Splitting data into train/val/test sets...")
    splits = loader.split_data(
        valid_examples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle=True,
        random_seed=random_seed
    )

    # Format data
    logger.info(f"Formatting data in {format_type} format...")
    formatted_splits = {}
    for split_name, split_data in splits.items():
        formatted_splits[split_name] = formatter.format_dataset(
            split_data,
            format_type=format_type
        )

    # Save processed data
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving processed data...")
    for split_name, split_data in formatted_splits.items():
        output_file = output_path / f"{split_name}_{format_type}.jsonl"
        loader.save_jsonl(split_data, output_file)

    # Save metadata
    metadata = {
        'format_type': format_type,
        'splits': {
            'train': len(formatted_splits['train']),
            'val': len(formatted_splits['val']),
            'test': len(formatted_splits['test']),
        },
        'validation_results': {
            'total_examples': validation_results['total_examples'],
            'valid_examples': validation_results['valid_examples'],
            'validation_rate': validation_results['validation_rate'],
            'clarification_rate': validation_results['clarification_rate'],
        },
        'statistics': stats,
        'random_seed': random_seed,
    }

    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to {metadata_file}")
    logger.info("Data preparation complete!")

    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Processed data saved to: {output_path}")
    print(f"Format: {format_type}")
    print(f"Splits:")
    for split_name, count in metadata['splits'].items():
        print(f"  {split_name}: {count} examples")
    print("="*60 + "\n")


if __name__ == '__main__':
    prepare_data()
