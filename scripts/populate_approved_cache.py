#!/usr/bin/env python3
"""
Populate the approved query cache with validated training queries.

This script:
1. Reads the generated training queries
2. Adds them to the semantic cache as approved queries
3. Reports on cache status
"""

import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.semantic_cache import SemanticCache


def main():
    print("=" * 70)
    print("Populate Approved Query Cache")
    print("=" * 70)

    # Find training data files
    training_files = [
        "data/feedback/generated_training.jsonl",
        "data/feedback/approved_training.jsonl",
    ]

    existing_files = [f for f in training_files if os.path.exists(f)]

    if not existing_files:
        print("ERROR: No training data files found.")
        print("Run scripts/generate_training_queries.py first.")
        return 1

    print(f"Found {len(existing_files)} training data file(s)")
    print()

    # Initialize cache
    print("Initializing semantic cache...")
    cache = SemanticCache()

    if not cache.is_available():
        print("ERROR: Semantic cache not available. Check embedding model.")
        return 1

    print(f"  Embedding model: {cache.model_name}")
    print()

    # Load and add queries
    total_added = 0
    total_skipped = 0

    for filepath in existing_files:
        print(f"Processing {filepath}...")

        with open(filepath, "r") as f:
            queries = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        queries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        print(f"  Found {len(queries)} queries")

        for i, q in enumerate(queries):
            instruction = q.get("instruction", "")
            output = q.get("output", "")

            if not instruction or not output:
                continue

            # Check if already in cache
            match = cache.search(instruction)
            if match.found and match.similarity_score > 0.95:
                total_skipped += 1
                continue

            # Add to cache
            cache.add_approved_query(
                instruction=instruction,
                query=output,
                was_corrected=False
            )
            total_added += 1

            if (i + 1) % 50 == 0:
                print(f"    Added {i + 1}/{len(queries)} queries...")

        print(f"  Added {total_added} new queries from this file")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total queries added: {total_added}")
    print(f"Total skipped (duplicates): {total_skipped}")
    print(f"Cache size: {len(cache.approved_queries)} queries")
    print()

    # Save cache
    print("Saving cache...")
    cache.save_cache()
    print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
