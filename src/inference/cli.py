"""
Command-line interface for Splunk Query generation.
"""

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from loguru import logger

from src.inference.model import SplunkQueryGenerator


console = Console()


def print_response(response: str, is_clarification: bool):
    """
    Pretty print the model response.

    Args:
        response: Generated response
        is_clarification: Whether it's a clarification request
    """
    if is_clarification:
        console.print(
            Panel(
                response,
                title="[yellow]Clarification Needed[/yellow]",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                f"[green]{response}[/green]",
                title="[green]Generated Splunk Query[/green]",
                border_style="green",
            )
        )


@click.command()
@click.option('--model-path', required=True, help='Path to fine-tuned model')
@click.option('--base-model', default=None, help='Base model name (if using PEFT adapter)')
@click.option('--interactive', is_flag=True, help='Run in interactive mode')
@click.option('--instruction', default=None, help='Single instruction to process')
@click.option('--input-text', default='', help='Additional context/input')
@click.option('--temperature', default=0.1, type=float, help='Sampling temperature')
@click.option('--max-tokens', default=512, type=int, help='Maximum tokens to generate')
def main(
    model_path: str,
    base_model: str,
    interactive: bool,
    instruction: str,
    input_text: str,
    temperature: float,
    max_tokens: int,
):
    """
    Splunk Query Generator CLI.

    Generate Splunk queries from natural language descriptions.
    """
    console.print("[bold blue]Splunk Query Generator[/bold blue]", style="bold")
    console.print()

    # Load model
    with console.status("[bold green]Loading model...", spinner="dots"):
        generator = SplunkQueryGenerator(
            model_path=model_path,
            base_model=base_model,
        )

    console.print("[green]✓[/green] Model loaded successfully")
    console.print()

    if interactive:
        # Interactive mode
        console.print("[bold]Interactive Mode[/bold]")
        console.print("Enter your Splunk query requests (or 'quit' to exit)")
        console.print()

        while True:
            try:
                # Get user input
                user_instruction = Prompt.ask("\n[bold cyan]Query Request[/bold cyan]")

                if user_instruction.lower() in ['quit', 'exit', 'q']:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break

                if not user_instruction.strip():
                    continue

                # Ask for additional context
                context = Prompt.ask(
                    "[dim]Additional context (press Enter to skip)[/dim]",
                    default=""
                )

                # Generate query
                with console.status("[bold green]Generating query...", spinner="dots"):
                    response = generator.generate_query(
                        instruction=user_instruction,
                        input_text=context,
                        temperature=temperature,
                        max_new_tokens=max_tokens,
                    )

                # Check if clarification
                is_clarification = generator.is_clarification_request(response)

                # Print response
                console.print()
                print_response(response, is_clarification)

                if is_clarification:
                    # Extract questions
                    questions = generator.extract_clarification_questions(response)
                    if questions:
                        console.print("\n[dim]Clarification questions:[/dim]")
                        for i, question in enumerate(questions, 1):
                            console.print(f"  {i}. {question}")

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error:[/red] {e}")

    else:
        # Single query mode
        if instruction is None:
            console.print("[red]Error:[/red] --instruction required in non-interactive mode")
            return

        console.print(f"[bold]Instruction:[/bold] {instruction}")
        if input_text:
            console.print(f"[bold]Context:[/bold] {input_text}")
        console.print()

        # Generate query
        with console.status("[bold green]Generating query...", spinner="dots"):
            response = generator.generate_query(
                instruction=instruction,
                input_text=input_text,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )

        # Check if clarification
        is_clarification = generator.is_clarification_request(response)

        # Print response
        print_response(response, is_clarification)


if __name__ == "__main__":
    main()
