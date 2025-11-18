"""
REST API client example for Zeus Splunk Query LLM.

This demonstrates how to interact with the Zeus API server.
"""

import requests
import json
from typing import Dict, List


class ZeusAPIClient:
    """Client for interacting with Zeus Splunk Query API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')

    def health_check(self) -> Dict:
        """
        Check API health status.

        Returns:
            Health status response
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def generate_query(
        self,
        instruction: str,
        input_text: str = "",
        temperature: float = 0.1,
        max_new_tokens: int = 512,
    ) -> Dict:
        """
        Generate a Splunk query from instruction.

        Args:
            instruction: Natural language instruction
            input_text: Additional context
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            Query generation response
        """
        payload = {
            "instruction": instruction,
            "input": input_text,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }

        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def batch_generate(self, requests: List[Dict]) -> Dict:
        """
        Generate queries for multiple requests.

        Args:
            requests: List of request dictionaries

        Returns:
            Batch generation response
        """
        payload = {"requests": requests}

        response = requests.post(
            f"{self.base_url}/batch_generate",
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def main():
    """Demonstrate API client usage."""

    print("="*70)
    print("Zeus Splunk Query LLM - API Client Example")
    print("="*70)
    print()

    # Initialize client
    client = ZeusAPIClient(base_url="http://localhost:8000")

    # Check health
    print("Checking API health...")
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Model loaded: {health['model_loaded']}")
        print()
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API server!")
        print("Please start the server first:")
        print("  python -m src.inference.server --model-path models/splunk-query-llm")
        return

    # Example 1: Single query generation
    print("-"*70)
    print("Example 1: Single Query Generation")
    print("-"*70)

    instruction = "Find failed login attempts in the last 24 hours"
    print(f"Request: {instruction}")
    print()

    response = client.generate_query(instruction=instruction)

    print(f"Query: {response['query']}")
    print(f"Is clarification: {response['is_clarification']}")
    if response['alternatives']:
        print(f"Alternatives: {len(response['alternatives'])}")
    print()

    # Example 2: Query with context
    print("-"*70)
    print("Example 2: Query with Additional Context")
    print("-"*70)

    instruction = "Show me malware execution events"
    context = "Windows systems, last 6 hours"
    print(f"Request: {instruction}")
    print(f"Context: {context}")
    print()

    response = client.generate_query(
        instruction=instruction,
        input_text=context
    )

    print(f"Query: {response['query']}")
    print()

    # Example 3: Ambiguous request
    print("-"*70)
    print("Example 3: Ambiguous Request (Expect Clarification)")
    print("-"*70)

    instruction = "Show me errors"
    print(f"Request: {instruction}")
    print()

    response = client.generate_query(instruction=instruction)

    if response['is_clarification']:
        print("Clarification requested:")
        print(f"\n{response['query']}\n")

        if response['clarification_questions']:
            print("Questions:")
            for i, q in enumerate(response['clarification_questions'], 1):
                print(f"  {i}. {q}")
    else:
        print(f"Query: {response['query']}")

    print()

    # Example 4: Batch processing
    print("-"*70)
    print("Example 4: Batch Query Generation")
    print("-"*70)

    batch_requests = [
        {
            "instruction": "Find firewall blocked traffic",
            "input": "Last hour",
            "temperature": 0.1,
        },
        {
            "instruction": "Detect brute force attacks",
            "input": "More than 5 failures",
            "temperature": 0.1,
        },
        {
            "instruction": "Show PowerShell execution events",
            "input": "Windows Event Logs",
            "temperature": 0.1,
        },
    ]

    print(f"Sending {len(batch_requests)} requests...")
    print()

    batch_response = client.batch_generate(batch_requests)

    print("Batch results:")
    for i, result in enumerate(batch_response['responses'], 1):
        print(f"\n[{i}] {batch_requests[i-1]['instruction']}")
        print(f"    Query: {result['query'][:80]}...")

    print()

    # Example 5: Multiple alternatives
    print("-"*70)
    print("Example 5: Generate Multiple Query Alternatives")
    print("-"*70)

    instruction = "Find network anomalies"
    print(f"Request: {instruction}")
    print()

    response = client.generate_query(
        instruction=instruction,
        temperature=0.7,  # Higher temp for diversity
    )

    print(f"Primary query:\n{response['query']}")

    if response['alternatives']:
        print(f"\nAlternatives ({len(response['alternatives'])}):")
        for i, alt in enumerate(response['alternatives'], 1):
            print(f"\n{i}. {alt}")

    print()
    print("="*70)
    print("API client examples complete!")
    print("="*70)


if __name__ == "__main__":
    main()
