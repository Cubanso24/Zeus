"""
Data validation utilities for Splunk query training data.
"""

import re
from typing import List, Dict, Set, Tuple
from loguru import logger


class SplunkQueryValidator:
    """Validate Splunk query training data quality."""

    # Common Splunk commands for validation
    SPLUNK_COMMANDS = {
        'search', 'stats', 'eval', 'table', 'where', 'timechart', 'chart',
        'sort', 'head', 'tail', 'dedup', 'rename', 'fields', 'rex', 'transaction',
        'join', 'append', 'lookup', 'inputlookup', 'outputlookup', 'bucket',
        'eventstats', 'streamstats', 'iplocation', 'top', 'rare', 'return',
        'map', 'foreach', 'reverse', 'predict', 'anomalies', 'cluster',
        'fillnull', 'makemv', 'mvexpand', 'replace', 'convert', 'collect',
    }

    REQUIRED_FIELDS = ['instruction', 'output']
    CLARIFICATION_PREFIX = 'CLARIFICATION:'

    def __init__(self):
        """Initialize the validator."""
        self.validation_errors = []
        self.validation_warnings = []

    def validate_example(self, example: Dict, example_id: str = None) -> Tuple[bool, List[str]]:
        """
        Validate a single training example.

        Args:
            example: Training example dictionary
            example_id: Optional identifier for the example

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        example_id = example_id or example.get('id', 'unknown')

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in example or not example[field]:
                errors.append(f"[{example_id}] Missing required field: {field}")

        if errors:
            return False, errors

        instruction = example['instruction']
        output = example['output']

        # Validate instruction
        if len(instruction.strip()) < 5:
            errors.append(f"[{example_id}] Instruction too short: {instruction}")

        # Validate output
        if output.startswith(self.CLARIFICATION_PREFIX):
            # This is a clarification request, different validation rules
            if len(output) < 50:
                errors.append(f"[{example_id}] Clarification message too short")
        else:
            # This should be a Splunk query
            query_errors = self._validate_splunk_query(output, example_id)
            errors.extend(query_errors)

        return len(errors) == 0, errors

    def _validate_splunk_query(self, query: str, example_id: str) -> List[str]:
        """
        Validate that a string looks like a valid Splunk query.

        Args:
            query: Splunk query string
            example_id: Example identifier for error messages

        Returns:
            List of error messages
        """
        errors = []

        # Check minimum length
        if len(query.strip()) < 10:
            errors.append(f"[{example_id}] Query too short: {query}")
            return errors

        # Check if query starts with index= or common search commands
        query_lower = query.lower().strip()

        valid_starts = ['index=', 'search ', '|', 'eventtype=', 'sourcetype=', 'source=', 'host=']
        if not any(query_lower.startswith(start) for start in valid_starts):
            errors.append(f"[{example_id}] Query doesn't start with valid SPL: {query[:50]}...")

        # Check for balanced pipes (basic syntax check)
        # Note: This is a simple check, not comprehensive
        pipe_count = query.count('|')

        # Check for common SPL commands
        found_commands = []
        for cmd in self.SPLUNK_COMMANDS:
            if re.search(rf'\b{cmd}\b', query_lower):
                found_commands.append(cmd)

        # Warn if no recognized SPL commands found (except for simple searches)
        if not found_commands and pipe_count == 0 and not query_lower.startswith('index='):
            errors.append(f"[{example_id}] No recognized SPL commands found in: {query[:50]}...")

        return errors

    def validate_dataset(self, examples: List[Dict]) -> Dict[str, any]:
        """
        Validate entire dataset and return statistics.

        Args:
            examples: List of training examples

        Returns:
            Dictionary with validation results and statistics
        """
        results = {
            'total_examples': len(examples),
            'valid_examples': 0,
            'invalid_examples': 0,
            'clarification_examples': 0,
            'query_examples': 0,
            'errors': [],
            'warnings': [],
        }

        for example in examples:
            example_id = example.get('id', f"example_{examples.index(example)}")

            is_valid, errors = self.validate_example(example, example_id)

            if is_valid:
                results['valid_examples'] += 1
            else:
                results['invalid_examples'] += 1
                results['errors'].extend(errors)

            # Count clarification vs query examples
            output = example.get('output', '')
            if output.startswith(self.CLARIFICATION_PREFIX):
                results['clarification_examples'] += 1
            else:
                results['query_examples'] += 1

        # Calculate statistics
        results['validation_rate'] = (
            results['valid_examples'] / results['total_examples'] * 100
            if results['total_examples'] > 0 else 0
        )

        results['clarification_rate'] = (
            results['clarification_examples'] / results['total_examples'] * 100
            if results['total_examples'] > 0 else 0
        )

        # Log summary
        logger.info(f"Validation complete: {results['valid_examples']}/{results['total_examples']} valid "
                   f"({results['validation_rate']:.1f}%)")
        logger.info(f"Query examples: {results['query_examples']}, "
                   f"Clarification examples: {results['clarification_examples']}")

        if results['errors']:
            logger.warning(f"Found {len(results['errors'])} validation errors")
            for error in results['errors'][:10]:  # Show first 10 errors
                logger.warning(error)
            if len(results['errors']) > 10:
                logger.warning(f"... and {len(results['errors']) - 10} more errors")

        return results

    def get_dataset_statistics(self, examples: List[Dict]) -> Dict[str, any]:
        """
        Get detailed statistics about the dataset.

        Args:
            examples: List of training examples

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_examples': len(examples),
            'avg_instruction_length': 0,
            'avg_output_length': 0,
            'unique_instructions': 0,
            'unique_outputs': 0,
            'common_commands': {},
        }

        if not examples:
            return stats

        # Calculate lengths
        instruction_lengths = [len(ex.get('instruction', '')) for ex in examples]
        output_lengths = [len(ex.get('output', '')) for ex in examples]

        stats['avg_instruction_length'] = sum(instruction_lengths) / len(instruction_lengths)
        stats['avg_output_length'] = sum(output_lengths) / len(output_lengths)
        stats['min_instruction_length'] = min(instruction_lengths)
        stats['max_instruction_length'] = max(instruction_lengths)
        stats['min_output_length'] = min(output_lengths)
        stats['max_output_length'] = max(output_lengths)

        # Count unique values
        unique_instructions = set(ex.get('instruction', '') for ex in examples)
        unique_outputs = set(ex.get('output', '') for ex in examples)

        stats['unique_instructions'] = len(unique_instructions)
        stats['unique_outputs'] = len(unique_outputs)

        # Find common SPL commands
        command_counts = {cmd: 0 for cmd in self.SPLUNK_COMMANDS}

        for example in examples:
            output = example.get('output', '').lower()
            for cmd in self.SPLUNK_COMMANDS:
                if re.search(rf'\b{cmd}\b', output):
                    command_counts[cmd] += 1

        # Get top 10 most common commands
        stats['common_commands'] = dict(
            sorted(command_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        return stats
