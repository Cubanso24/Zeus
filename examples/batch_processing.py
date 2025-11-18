"""
Batch processing example for Zeus Splunk Query LLM.

This example demonstrates how to process multiple queries efficiently.
"""

import json
import time
from pathlib import Path
from typing import List, Dict
from src.inference.model import SplunkQueryGenerator


def load_requests(file_path: str) -> List[Dict]:
    """Load query requests from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_results(results: List[Dict], output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    """Demonstrate batch processing of Splunk query requests."""

    print("="*70)
    print("Zeus Splunk Query LLM - Batch Processing Example")
    print("="*70)
    print()

    # Initialize the generator
    model_path = "models/splunk-query-llm"
    print(f"Loading model from: {model_path}")

    generator = SplunkQueryGenerator(model_path=model_path)
    print("Model loaded successfully!")
    print()

    # Sample batch requests
    requests = [
        {
            "id": 1,
            "instruction": "Find failed login attempts in the last 24 hours",
            "input": ""
        },
        {
            "id": 2,
            "instruction": "Show me all firewall blocked traffic",
            "input": "Time range: last hour"
        },
        {
            "id": 3,
            "instruction": "Detect potential malware execution",
            "input": "Focus on Windows systems"
        },
        {
            "id": 4,
            "instruction": "Find unusual outbound connections",
            "input": "Exclude internal networks"
        },
        {
            "id": 5,
            "instruction": "Show me the errors",
            "input": ""  # This is ambiguous - should ask for clarification
        },
        {
            "id": 6,
            "instruction": "List top 10 users by event count",
            "input": "Last 7 days"
        },
        {
            "id": 7,
            "instruction": "Find PowerShell encoded commands",
            "input": ""
        },
        {
            "id": 8,
            "instruction": "Detect lateral movement",
            "input": "Windows Event Logs"
        },
    ]

    print(f"Processing {len(requests)} requests...")
    print()

    # Process batch
    results = []
    start_time = time.time()

    for i, request in enumerate(requests, 1):
        print(f"[{i}/{len(requests)}] Processing: {request['instruction'][:60]}...")

        # Generate query
        query = generator.generate_query(
            instruction=request['instruction'],
            input_text=request['input'],
        )

        # Check if clarification
        is_clarification = generator.is_clarification_request(query)

        # Store result
        result = {
            "request_id": request['id'],
            "instruction": request['instruction'],
            "input": request['input'],
            "query": query,
            "is_clarification": is_clarification,
            "timestamp": time.time(),
        }

        if is_clarification:
            result['clarification_questions'] = generator.extract_clarification_questions(query)

        results.append(result)

    end_time = time.time()
    elapsed = end_time - start_time

    print()
    print("-"*70)
    print("Batch Processing Complete!")
    print("-"*70)
    print(f"Total requests: {len(requests)}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average time per query: {elapsed/len(requests):.2f} seconds")
    print()

    # Print summary
    num_queries = sum(1 for r in results if not r['is_clarification'])
    num_clarifications = sum(1 for r in results if r['is_clarification'])

    print("Summary:")
    print(f"  Queries generated: {num_queries}")
    print(f"  Clarifications requested: {num_clarifications}")
    print()

    # Display results
    print("-"*70)
    print("Results:")
    print("-"*70)

    for result in results:
        print(f"\n[Request {result['request_id']}]")
        print(f"Instruction: {result['instruction']}")
        if result['input']:
            print(f"Context: {result['input']}")

        if result['is_clarification']:
            print("Status: CLARIFICATION NEEDED")
            print(f"Response:\n{result['query'][:200]}...")
        else:
            print("Status: QUERY GENERATED")
            print(f"Query: {result['query']}")

    # Save results to file
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "batch_results.json"
    save_results(results, str(output_file))

    print()
    print("="*70)


if __name__ == "__main__":
    main()
