"""
Evaluation metrics for Splunk Query LLM.
"""

import re
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np
from loguru import logger


class SplunkQueryMetrics:
    """Calculate metrics for Splunk query generation."""

    @staticmethod
    def exact_match(predicted: str, reference: str) -> float:
        """
        Calculate exact match accuracy.

        Args:
            predicted: Predicted query
            reference: Reference query

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return 1.0 if predicted.strip() == reference.strip() else 0.0

    @staticmethod
    def normalized_exact_match(predicted: str, reference: str) -> float:
        """
        Calculate normalized exact match (ignoring whitespace and case).

        Args:
            predicted: Predicted query
            reference: Reference query

        Returns:
            1.0 if normalized match, 0.0 otherwise
        """
        # Normalize: lowercase and collapse whitespace
        pred_norm = " ".join(predicted.lower().split())
        ref_norm = " ".join(reference.lower().split())

        return 1.0 if pred_norm == ref_norm else 0.0

    @staticmethod
    def token_overlap(predicted: str, reference: str) -> float:
        """
        Calculate token-level overlap (similar to BLEU unigrams).

        Args:
            predicted: Predicted query
            reference: Reference query

        Returns:
            F1 score of token overlap
        """
        pred_tokens = set(predicted.lower().split())
        ref_tokens = set(reference.lower().split())

        if not pred_tokens or not ref_tokens:
            return 0.0

        intersection = pred_tokens & ref_tokens

        if not intersection:
            return 0.0

        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(ref_tokens)

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1

    @staticmethod
    def command_accuracy(predicted: str, reference: str) -> float:
        """
        Check if predicted query contains the same SPL commands as reference.

        Args:
            predicted: Predicted query
            reference: Reference query

        Returns:
            F1 score of command overlap
        """
        spl_commands = {
            'search', 'stats', 'eval', 'table', 'where', 'timechart', 'chart',
            'sort', 'head', 'tail', 'dedup', 'rename', 'fields', 'rex', 'transaction',
            'join', 'append', 'lookup', 'inputlookup', 'outputlookup', 'bucket',
            'eventstats', 'streamstats', 'iplocation', 'top', 'rare', 'return',
            'map', 'foreach', 'reverse', 'predict', 'anomalies', 'cluster',
        }

        # Extract commands
        pred_commands = set()
        ref_commands = set()

        for cmd in spl_commands:
            if re.search(rf'\b{cmd}\b', predicted.lower()):
                pred_commands.add(cmd)
            if re.search(rf'\b{cmd}\b', reference.lower()):
                ref_commands.add(cmd)

        if not pred_commands or not ref_commands:
            return 0.0

        intersection = pred_commands & ref_commands

        if not intersection:
            return 0.0

        precision = len(intersection) / len(pred_commands)
        recall = len(intersection) / len(ref_commands)

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1

    @staticmethod
    def is_syntactically_valid(query: str) -> bool:
        """
        Basic syntax validation for Splunk queries.

        Args:
            query: Splunk query to validate

        Returns:
            True if query appears syntactically valid
        """
        query = query.strip().lower()

        # Skip clarification requests
        if query.startswith("clarification:"):
            return True

        # Check if starts with valid search
        valid_starts = ['index=', 'search ', '|', 'eventtype=', 'sourcetype=', 'source=', 'host=']
        if not any(query.startswith(start) for start in valid_starts):
            return False

        # Check for balanced quotes
        single_quotes = query.count("'") - query.count("\\'")
        double_quotes = query.count('"') - query.count('\\"')

        if single_quotes % 2 != 0 or double_quotes % 2 != 0:
            return False

        # Check for balanced parentheses
        paren_balance = query.count('(') - query.count(')')
        if paren_balance != 0:
            return False

        return True

    @staticmethod
    def clarification_detection_accuracy(
        predicted: str,
        reference: str,
        clarification_prefix: str = "CLARIFICATION:"
    ) -> float:
        """
        Check if model correctly identified need for clarification.

        Args:
            predicted: Predicted response
            reference: Reference response
            clarification_prefix: Prefix used for clarification requests

        Returns:
            1.0 if both are clarifications or both are queries, 0.0 otherwise
        """
        pred_is_clarification = predicted.strip().startswith(clarification_prefix)
        ref_is_clarification = reference.strip().startswith(clarification_prefix)

        return 1.0 if pred_is_clarification == ref_is_clarification else 0.0

    def evaluate_predictions(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate a set of predictions against references.

        Args:
            predictions: List of predicted queries
            references: List of reference queries

        Returns:
            Dictionary of metric scores
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions must match number of references")

        metrics = {
            'exact_match': [],
            'normalized_exact_match': [],
            'token_overlap': [],
            'command_accuracy': [],
            'syntax_validity': [],
            'clarification_accuracy': [],
        }

        for pred, ref in zip(predictions, references):
            metrics['exact_match'].append(self.exact_match(pred, ref))
            metrics['normalized_exact_match'].append(self.normalized_exact_match(pred, ref))
            metrics['token_overlap'].append(self.token_overlap(pred, ref))
            metrics['command_accuracy'].append(self.command_accuracy(pred, ref))
            metrics['syntax_validity'].append(1.0 if self.is_syntactically_valid(pred) else 0.0)
            metrics['clarification_accuracy'].append(self.clarification_detection_accuracy(pred, ref))

        # Calculate averages
        results = {
            metric_name: np.mean(scores) * 100  # Convert to percentage
            for metric_name, scores in metrics.items()
        }

        # Add count
        results['num_examples'] = len(predictions)

        return results

    def detailed_evaluation(
        self,
        predictions: List[str],
        references: List[str],
        instructions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Detailed evaluation with per-example analysis.

        Args:
            predictions: List of predicted queries
            references: List of reference queries
            instructions: Optional list of instructions

        Returns:
            Detailed evaluation results
        """
        overall_metrics = self.evaluate_predictions(predictions, references)

        # Per-example results
        examples = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            example = {
                'id': i,
                'prediction': pred,
                'reference': ref,
                'exact_match': self.exact_match(pred, ref),
                'normalized_exact_match': self.normalized_exact_match(pred, ref),
                'token_overlap': self.token_overlap(pred, ref),
                'command_accuracy': self.command_accuracy(pred, ref),
                'syntax_valid': self.is_syntactically_valid(pred),
                'clarification_match': self.clarification_detection_accuracy(pred, ref),
            }

            if instructions:
                example['instruction'] = instructions[i]

            examples.append(example)

        # Category analysis
        num_clarifications_pred = sum(1 for p in predictions if p.strip().startswith("CLARIFICATION:"))
        num_clarifications_ref = sum(1 for r in references if r.strip().startswith("CLARIFICATION:"))
        num_syntax_valid = sum(1 for p in predictions if self.is_syntactically_valid(p))

        return {
            'overall_metrics': overall_metrics,
            'examples': examples,
            'statistics': {
                'total_examples': len(predictions),
                'clarifications_predicted': num_clarifications_pred,
                'clarifications_reference': num_clarifications_ref,
                'syntax_valid': num_syntax_valid,
                'syntax_valid_rate': num_syntax_valid / len(predictions) * 100,
            }
        }
