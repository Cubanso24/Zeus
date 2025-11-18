"""
Basic usage examples for Zeus Splunk Query LLM.
"""

from src.inference.model import SplunkQueryGenerator


def main():
    """Demonstrate basic usage of the Splunk Query Generator."""

    print("="*70)
    print("Zeus Splunk Query LLM - Basic Usage Examples")
    print("="*70)
    print()

    # Initialize the generator
    # NOTE: Replace with your actual model path
    model_path = "models/splunk-query-llm"

    print(f"Loading model from: {model_path}")
    print("(This may take a minute...)")
    print()

    generator = SplunkQueryGenerator(
        model_path=model_path,
        # base_model="mistralai/Mistral-7B-Instruct-v0.2",  # Uncomment if using PEFT adapter
    )

    print("Model loaded successfully!")
    print()

    # Example 1: Simple query generation
    print("-"*70)
    print("Example 1: Simple Query Generation")
    print("-"*70)

    instruction = "Find all failed SSH login attempts in the last 24 hours"
    print(f"Request: {instruction}")
    print()

    query = generator.generate_query(instruction=instruction)
    print(f"Generated Query:\n{query}")
    print()

    # Example 2: Query with additional context
    print("-"*70)
    print("Example 2: Query with Additional Context")
    print("-"*70)

    instruction = "Show me failed login attempts"
    context = "Time range: last hour, Index: linux"
    print(f"Request: {instruction}")
    print(f"Context: {context}")
    print()

    query = generator.generate_query(
        instruction=instruction,
        input_text=context
    )
    print(f"Generated Query:\n{query}")
    print()

    # Example 3: Security use case
    print("-"*70)
    print("Example 3: Security Use Case - Brute Force Detection")
    print("-"*70)

    instruction = "Detect potential brute force attacks with more than 5 failed login attempts from the same IP"
    print(f"Request: {instruction}")
    print()

    query = generator.generate_query(instruction=instruction)
    print(f"Generated Query:\n{query}")
    print()

    # Example 4: Ambiguous request (should ask for clarification)
    print("-"*70)
    print("Example 4: Ambiguous Request (Should Ask for Clarification)")
    print("-"*70)

    instruction = "Show me errors"
    print(f"Request: {instruction}")
    print()

    response = generator.generate_query(instruction=instruction)

    if generator.is_clarification_request(response):
        print("Model is asking for clarification:")
        print(f"\n{response}\n")

        questions = generator.extract_clarification_questions(response)
        if questions:
            print("Extracted questions:")
            for i, question in enumerate(questions, 1):
                print(f"  {i}. {question}")
    else:
        print(f"Generated Query:\n{response}")

    print()

    # Example 5: Multiple query alternatives
    print("-"*70)
    print("Example 5: Generate Multiple Alternatives")
    print("-"*70)

    instruction = "Find network traffic anomalies"
    print(f"Request: {instruction}")
    print()

    queries = generator.generate(
        instruction=instruction,
        num_return_sequences=3,
        temperature=0.7,  # Higher temperature for diversity
    )

    print("Generated alternatives:")
    for i, query in enumerate(queries, 1):
        print(f"\nAlternative {i}:")
        print(query)

    print()
    print("="*70)
    print("Examples complete!")
    print("="*70)


if __name__ == "__main__":
    main()
