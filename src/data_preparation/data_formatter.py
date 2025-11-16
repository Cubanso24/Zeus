"""
Data formatting utilities for different LLM training frameworks.
"""

from typing import List, Dict, Optional
from loguru import logger


class DataFormatter:
    """Format training data for different LLM frameworks."""

    @staticmethod
    def format_alpaca_style(example: Dict) -> Dict:
        """
        Format example in Alpaca instruction format.

        Args:
            example: Raw training example

        Returns:
            Formatted example in Alpaca style
        """
        return {
            'instruction': example['instruction'],
            'input': example.get('input', ''),
            'output': example['output'],
        }

    @staticmethod
    def format_chat_style(
        example: Dict,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Format example in chat/conversation format (for ChatML, etc.).

        Args:
            example: Raw training example
            system_prompt: Optional system prompt to include

        Returns:
            Formatted example in chat style
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        # User message
        user_content = example['instruction']
        if example.get('input'):
            user_content += f"\n\n{example['input']}"

        messages.append({
            'role': 'user',
            'content': user_content
        })

        # Assistant message
        messages.append({
            'role': 'assistant',
            'content': example['output']
        })

        return {'messages': messages}

    @staticmethod
    def format_completion_style(
        example: Dict,
        prompt_template: Optional[str] = None
    ) -> Dict:
        """
        Format example for completion-style training.

        Args:
            example: Raw training example
            prompt_template: Optional template for formatting the prompt

        Returns:
            Formatted example with prompt and completion
        """
        if prompt_template is None:
            prompt_template = (
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n"
            )

        input_text = example.get('input', '')
        prompt = prompt_template.format(
            instruction=example['instruction'],
            input=input_text
        )

        return {
            'prompt': prompt,
            'completion': example['output']
        }

    @staticmethod
    def get_default_system_prompt() -> str:
        """
        Get the default system prompt for Splunk query generation.

        Returns:
            System prompt string
        """
        return """You are a Splunk Query Expert Assistant specialized in generating accurate SPL (Splunk Processing Language) queries for cybersecurity analysts.

Your responsibilities:
1. Generate syntactically correct and efficient Splunk queries based on user requests
2. If the request is ambiguous or lacks necessary details, ask clarifying questions instead of making assumptions
3. When asking for clarification, prefix your response with "CLARIFICATION:" and ask specific questions about what information you need
4. Provide the most accurate query possible with the information given
5. Focus on cybersecurity use cases including threat detection, incident investigation, and security monitoring

Guidelines:
- Always validate that you have enough information before generating a query
- Use appropriate Splunk commands and best practices
- Consider performance and efficiency in your queries
- Be specific in your clarifying questions
- Only respond with a Splunk query or a clarification request"""

    def format_dataset(
        self,
        examples: List[Dict],
        format_type: str = 'alpaca',
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None
    ) -> List[Dict]:
        """
        Format entire dataset for a specific training framework.

        Args:
            examples: List of raw training examples
            format_type: Type of formatting ('alpaca', 'chat', 'completion')
            system_prompt: Optional system prompt (for chat format)
            prompt_template: Optional prompt template (for completion format)

        Returns:
            List of formatted examples
        """
        if system_prompt is None and format_type == 'chat':
            system_prompt = self.get_default_system_prompt()

        formatted_examples = []
        format_func = {
            'alpaca': self.format_alpaca_style,
            'chat': lambda ex: self.format_chat_style(ex, system_prompt),
            'completion': lambda ex: self.format_completion_style(ex, prompt_template),
        }.get(format_type)

        if not format_func:
            raise ValueError(f"Unknown format type: {format_type}. "
                           f"Choose from: alpaca, chat, completion")

        for example in examples:
            try:
                formatted = format_func(example)
                formatted_examples.append(formatted)
            except Exception as e:
                logger.error(f"Error formatting example {example.get('id', 'unknown')}: {e}")

        logger.info(f"Formatted {len(formatted_examples)} examples in {format_type} format")
        return formatted_examples


class PromptTemplates:
    """Collection of prompt templates for different scenarios."""

    BASIC_TEMPLATE = (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n"
    )

    DETAILED_TEMPLATE = (
        "You are a Splunk Query Expert. Generate an accurate SPL query for the following request.\n\n"
        "Request: {instruction}\n\n"
        "Additional Context: {input}\n\n"
        "Splunk Query:"
    )

    CONVERSATIONAL_TEMPLATE = (
        "User: {instruction}\n"
        "{input}\n\n"
        "Splunk Expert: "
    )

    @classmethod
    def get_template(cls, template_name: str) -> str:
        """
        Get a prompt template by name.

        Args:
            template_name: Name of the template

        Returns:
            Template string
        """
        templates = {
            'basic': cls.BASIC_TEMPLATE,
            'detailed': cls.DETAILED_TEMPLATE,
            'conversational': cls.CONVERSATIONAL_TEMPLATE,
        }

        return templates.get(template_name, cls.BASIC_TEMPLATE)
