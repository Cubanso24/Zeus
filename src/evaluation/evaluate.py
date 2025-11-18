"""
Evaluation script for Splunk Query LLM.
"""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict
import click
from loguru import logger
from tqdm import tqdm
import pandas as pd

from src.inference.model import SplunkQueryGenerator
from src.evaluation.metrics import SplunkQueryMetrics


def load_test_data(test_file: str) -> List[Dict]:
    """
    Load test data from file.

    Args:
        test_file: Path to test data file (JSONL)

    Returns:
        List of test examples
    """
    test_file = Path(test_file)
    examples = []

    with jsonlines.open(test_file) as reader:
        for obj in reader:
            examples.append(obj)

    logger.info(f"Loaded {len(examples)} test examples from {test_file}")
    return examples


def generate_predictions(
    model: SplunkQueryGenerator,
    test_examples: List[Dict],
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> List[str]:
    """
    Generate predictions for test examples.

    Args:
        model: Query generator model
        test_examples: List of test examples
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        List of predicted queries
    """
    predictions = []

    logger.info("Generating predictions...")
    for example in tqdm(test_examples, desc="Generating"):
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')

        prediction = model.generate_query(
            instruction=instruction,
            input_text=input_text,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

        predictions.append(prediction)

    return predictions


@click.command()
@click.option('--model-path', required=True, help='Path to fine-tuned model')
@click.option('--base-model', default=None, help='Base model name (if using PEFT adapter)')
@click.option('--test-file', required=True, help='Path to test data file (JSONL)')
@click.option('--output-dir', default='evaluation_results', help='Directory to save results')
@click.option('--temperature', default=0.1, type=float, help='Sampling temperature')
@click.option('--max-tokens', default=512, type=int, help='Maximum tokens to generate')
@click.option('--save-predictions', is_flag=True, help='Save predictions to file')
def evaluate(
    model_path: str,
    base_model: str,
    test_file: str,
    output_dir: str,
    temperature: float,
    max_tokens: int,
    save_predictions: bool,
):
    """
    Evaluate Splunk Query LLM on test data.
    """
    logger.info("Starting evaluation")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = SplunkQueryGenerator(
        model_path=model_path,
        base_model=base_model,
    )

    # Load test data
    test_examples = load_test_data(test_file)

    # Generate predictions
    predictions = generate_predictions(
        model=model,
        test_examples=test_examples,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Extract references
    references = [example.get('output', '') for example in test_examples]
    instructions = [example.get('instruction', '') for example in test_examples]

    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics_calculator = SplunkQueryMetrics()

    results = metrics_calculator.detailed_evaluation(
        predictions=predictions,
        references=references,
        instructions=instructions,
    )

    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nTotal Examples: {results['statistics']['total_examples']}")
    print(f"Syntax Valid: {results['statistics']['syntax_valid']} "
          f"({results['statistics']['syntax_valid_rate']:.1f}%)")
    print(f"Clarifications Predicted: {results['statistics']['clarifications_predicted']}")
    print(f"Clarifications Reference: {results['statistics']['clarifications_reference']}")

    print("\n" + "-"*70)
    print("METRICS")
    print("-"*70)

    for metric_name, score in results['overall_metrics'].items():
        if metric_name != 'num_examples':
            print(f"{metric_name:.<50} {score:>6.2f}%")

    print("="*70 + "\n")

    # Save results
    results_file = output_path / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved detailed results to {results_file}")

    # Save summary
    summary_file = output_path / 'summary.json'
    summary = {
        'model_path': model_path,
        'test_file': test_file,
        'metrics': results['overall_metrics'],
        'statistics': results['statistics'],
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")

    # Save predictions if requested
    if save_predictions:
        predictions_file = output_path / 'predictions.jsonl'
        with jsonlines.open(predictions_file, mode='w') as writer:
            for i, (example, pred) in enumerate(zip(test_examples, predictions)):
                output_obj = {
                    'id': i,
                    'instruction': example.get('instruction', ''),
                    'input': example.get('input', ''),
                    'reference': example.get('output', ''),
                    'prediction': pred,
                    'metrics': results['examples'][i],
                }
                writer.write(output_obj)
        logger.info(f"Saved predictions to {predictions_file}")

    # Create DataFrame for analysis
    df_data = []
    for ex in results['examples']:
        df_data.append({
            'instruction': ex.get('instruction', ''),
            'exact_match': ex['exact_match'],
            'normalized_match': ex['normalized_exact_match'],
            'token_overlap': ex['token_overlap'],
            'command_accuracy': ex['command_accuracy'],
            'syntax_valid': ex['syntax_valid'],
            'clarification_match': ex['clarification_match'],
        })

    df = pd.DataFrame(df_data)

    # Save CSV
    csv_file = output_path / 'results.csv'
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved results CSV to {csv_file}")

    logger.info("Evaluation complete!")


def main():
    """Main entry point."""
    evaluate()


if __name__ == "__main__":
    main()
