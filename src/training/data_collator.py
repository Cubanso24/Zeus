"""
Custom data collators for Splunk Query LLM training.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from transformers import PreTrainedTokenizer


@dataclass
class SplunkQueryDataCollator:
    """
    Data collator for Splunk query instruction tuning.
    Handles padding and label masking for instruction-following tasks.
    """

    tokenizer: PreTrainedTokenizer
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    mask_prompt: bool = True  # Whether to mask prompt in loss calculation

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples.

        Args:
            features: List of examples with input_ids and labels

        Returns:
            Batched tensors
        """
        # Extract input_ids and labels
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature.get("labels", feature["input_ids"]) for feature in features]

        # Determine max length
        max_length = self.max_length
        if max_length is None:
            max_length = max(len(ids) for ids in input_ids)

        # Pad sequences
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for ids, lbls in zip(input_ids, labels):
            # Calculate padding
            padding_length = max_length - len(ids)

            # Pad input_ids
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length

            # Create attention mask
            attention_mask = [1] * len(ids) + [0] * padding_length

            # Pad labels (use -100 for padding to ignore in loss)
            padded_labels = lbls + [-100] * padding_length

            batch["input_ids"].append(padded_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(padded_labels)

        # Convert to tensors
        batch = {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in batch.items()
        }

        return batch
