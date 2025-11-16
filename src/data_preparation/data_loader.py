"""
Data loading utilities for Splunk query training data.
"""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Optional, Union
from loguru import logger
import pandas as pd


class SplunkQueryDataLoader:
    """Load and preprocess Splunk query training data."""

    def __init__(self, data_dir: Union[str, Path] = "data/raw"):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing training data files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")

    def load_jsonl(self, file_path: Union[str, Path]) -> List[Dict]:
        """
        Load data from a JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of training examples as dictionaries
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        examples = []
        with jsonlines.open(file_path) as reader:
            for idx, obj in enumerate(reader):
                obj['id'] = f"{file_path.stem}_{idx}"
                examples.append(obj)

        logger.info(f"Loaded {len(examples)} examples from {file_path}")
        return examples

    def load_all_data(self, pattern: str = "*.jsonl") -> List[Dict]:
        """
        Load all JSONL files matching a pattern.

        Args:
            pattern: Glob pattern for files to load

        Returns:
            Combined list of all training examples
        """
        all_examples = []
        files = sorted(self.data_dir.glob(pattern))

        if not files:
            logger.warning(f"No files found matching pattern: {pattern} in {self.data_dir}")
            return all_examples

        for file_path in files:
            try:
                examples = self.load_jsonl(file_path)
                all_examples.extend(examples)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Loaded total of {len(all_examples)} examples from {len(files)} files")
        return all_examples

    def to_dataframe(self, examples: List[Dict]) -> pd.DataFrame:
        """
        Convert examples to pandas DataFrame.

        Args:
            examples: List of training examples

        Returns:
            DataFrame with training data
        """
        return pd.DataFrame(examples)

    def split_data(
        self,
        examples: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Dict[str, List[Dict]]:
        """
        Split data into train, validation, and test sets.

        Args:
            examples: List of training examples
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            shuffle: Whether to shuffle data before splitting
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with 'train', 'val', and 'test' splits
        """
        if not (0.99 <= (train_ratio + val_ratio + test_ratio) <= 1.01):
            raise ValueError("Split ratios must sum to 1.0")

        import random
        if shuffle:
            random.seed(random_seed)
            examples = examples.copy()
            random.shuffle(examples)

        n = len(examples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            'train': examples[:train_end],
            'val': examples[train_end:val_end],
            'test': examples[val_end:]
        }

        logger.info(f"Split data: train={len(splits['train'])}, "
                   f"val={len(splits['val'])}, test={len(splits['test'])}")

        return splits

    def save_jsonl(self, examples: List[Dict], output_path: Union[str, Path]):
        """
        Save examples to JSONL file.

        Args:
            examples: List of training examples
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with jsonlines.open(output_path, mode='w') as writer:
            writer.write_all(examples)

        logger.info(f"Saved {len(examples)} examples to {output_path}")
