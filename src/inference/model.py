"""
Model inference wrapper for Splunk Query generation.
"""

import torch
from typing import Optional, List, Dict
from pathlib import Path
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from peft import PeftModel


class SplunkQueryGenerator:
    """Generate Splunk queries using fine-tuned LLM."""

    def __init__(
        self,
        model_path: str,
        base_model: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        """
        Initialize the query generator.

        Args:
            model_path: Path to fine-tuned model (or adapter)
            base_model: Base model name (if using adapter)
            device: Device to load model on
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
        """
        self.model_path = Path(model_path)
        self.device = device

        logger.info(f"Loading model from {model_path}")

        # Check if this is a PEFT adapter or full model
        adapter_config_path = self.model_path / "adapter_config.json"
        is_peft_adapter = adapter_config_path.exists()

        if is_peft_adapter:
            if base_model is None:
                raise ValueError("base_model must be provided when loading PEFT adapter")

            logger.info(f"Loading PEFT adapter with base model: {base_model}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
            )

            # Load base model (device_map handled after PEFT loading)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            # Load adapter
            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.model_path),
                is_trainable=False,
            )

            # Move to device after PEFT loading to avoid offload issues
            if device == "cpu":
                # For CPU-only environments, explicitly move to CPU (no device_map)
                self.model = self.model.to("cpu")
            elif device != "auto":
                self.model = self.model.to(device)
            else:
                # Use device_map after model is fully assembled (multi-GPU only)
                from accelerate import dispatch_model, infer_auto_device_map
                device_map = infer_auto_device_map(self.model)
                self.model = dispatch_model(self.model, device_map=device_map)

        else:
            logger.info("Loading full fine-tuned model")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                device_map=device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        logger.info("Model loaded successfully")

    def format_prompt(
        self,
        instruction: str,
        input_text: str = "",
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format instruction as prompt for the model.

        Args:
            instruction: User's instruction/request
            input_text: Additional context or input
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        if system_prompt:
            prompt = f"{system_prompt}\n\n"
        else:
            prompt = ""

        prompt += f"### Instruction:\n{instruction}\n\n"

        if input_text:
            prompt += f"### Input:\n{input_text}\n\n"

        prompt += "### Response:\n"

        return prompt

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.2,
    ) -> List[str]:
        """
        Generate Splunk query from instruction.

        Args:
            instruction: User's query request
            input_text: Additional context
            system_prompt: Optional system prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of queries to generate
            do_sample: Whether to use sampling (vs greedy)

        Returns:
            List of generated Splunk queries
        """
        # Format prompt
        prompt = self.format_prompt(instruction, input_text, system_prompt)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode outputs
        generated_texts = []
        for output in outputs:
            # Decode
            text = self.tokenizer.decode(output, skip_special_tokens=True)

            # Extract only the response part
            if "### Response:" in text:
                response = text.split("### Response:")[-1].strip()
            else:
                response = text[len(prompt):].strip()

            # Clean up response to remove self-answering
            response = self.clean_response(response)

            generated_texts.append(response)

        return generated_texts

    def generate_query(
        self,
        instruction: str,
        input_text: str = "",
        **kwargs
    ) -> str:
        """
        Generate a single Splunk query.

        Args:
            instruction: User's query request
            input_text: Additional context
            **kwargs: Additional generation parameters

        Returns:
            Generated Splunk query or clarification request
        """
        results = self.generate(
            instruction=instruction,
            input_text=input_text,
            num_return_sequences=1,
            **kwargs
        )
        return results[0]

    def batch_generate(
        self,
        instructions: List[str],
        input_texts: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate queries for multiple instructions.

        Args:
            instructions: List of user instructions
            input_texts: Optional list of additional context
            **kwargs: Additional generation parameters

        Returns:
            List of generated queries
        """
        if input_texts is None:
            input_texts = [""] * len(instructions)

        results = []
        for instruction, input_text in zip(instructions, input_texts):
            query = self.generate_query(
                instruction=instruction,
                input_text=input_text,
                **kwargs
            )
            results.append(query)

        return results

    def is_clarification_request(self, response: str) -> bool:
        """
        Check if the response is a clarification request.

        Args:
            response: Generated response

        Returns:
            True if response is asking for clarification
        """
        # Check if response starts with or contains CLARIFICATION:
        return "CLARIFICATION:" in response

    def clean_response(self, response: str) -> str:
        """
        Clean up the model response by removing any self-answering or
        continuation after the main response.

        Args:
            response: Raw generated response

        Returns:
            Cleaned response
        """
        # If it contains a clarification, extract just the clarification part
        if "CLARIFICATION:" in response:
            # Find where CLARIFICATION starts
            clarification_start = response.find("CLARIFICATION:")
            # Take everything from CLARIFICATION to the first double newline or end
            clarification_text = response[clarification_start:]

            # Stop at common continuation patterns that indicate the model is self-responding
            stop_patterns = [
                "\n\nAnalyst response:",
                "\n\nQuery update:",
                "\n\nUser:",
                "\n\nResponse:",
                "\n\n###",
                "\n\nHere is",
                "\n\nBased on",
            ]

            for pattern in stop_patterns:
                if pattern.lower() in clarification_text.lower():
                    idx = clarification_text.lower().find(pattern.lower())
                    clarification_text = clarification_text[:idx]
                    break

            return clarification_text.strip()

        # For regular queries, stop at any continuation patterns
        stop_patterns = [
            "\n\nCLARIFICATION:",
            "\n\nAnalyst response:",
            "\n\nQuery update:",
            "\n\nUser:",
            "\n\n###",
            "\n\nExplanation:",
            "\n\nThis query",
        ]

        result = response
        for pattern in stop_patterns:
            if pattern.lower() in result.lower():
                idx = result.lower().find(pattern.lower())
                result = result[:idx]

        return result.strip()

    def extract_clarification_questions(self, response: str) -> List[str]:
        """
        Extract clarification questions from response.

        Args:
            response: Clarification response

        Returns:
            List of questions
        """
        if not self.is_clarification_request(response):
            return []

        # Remove the CLARIFICATION: prefix
        text = response.replace("CLARIFICATION:", "").strip()

        # Split by newlines and filter questions
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Find lines that are questions
        questions = [
            line for line in lines
            if line.endswith("?") or "?" in line
        ]

        return questions

    def generate_explanation(
        self,
        query: str,
        instruction: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate an explanation for a Splunk query using rule-based parsing.

        Args:
            query: The generated Splunk query to explain
            instruction: The original user instruction
            max_new_tokens: Unused, kept for API compatibility
            temperature: Unused, kept for API compatibility

        Returns:
            Explanation of what the query does
        """
        return self._parse_query_explanation(query, instruction)

    def _parse_query_explanation(self, query: str, instruction: str) -> str:
        """
        Parse a Splunk query and generate a human-readable explanation.

        Args:
            query: The Splunk query to explain
            instruction: The original user request

        Returns:
            Structured explanation of the query
        """
        import re

        # Initialize components
        input_parts = []
        output_parts = []

        # Extract index
        index_match = re.search(r'index=(\S+)', query)
        if index_match:
            index_name = index_match.group(1).strip('"\'')
            index_descriptions = {
                'main': 'default index for most data',
                '_internal': 'Splunk internal logs',
                '_audit': 'Splunk audit logs',
                'os': 'operating system logs (Linux/Windows)',
                'wineventlog': 'Windows Event Logs',
                'syslog': 'syslog data',
                # Legacy mappings for backwards compatibility
                'linux': 'Linux system logs',
                'windows': 'Windows event logs',
                'firewall': 'Firewall logs',
                'network': 'Network traffic logs',
                'security': 'Security event logs',
            }
            desc = index_descriptions.get(index_name.lower(), f'{index_name} data')
            input_parts.append(f"index: {index_name} ({desc})")

        # Extract sourcetype
        sourcetype_match = re.search(r'sourcetype=(\S+)', query)
        if sourcetype_match:
            sourcetype = sourcetype_match.group(1).strip('"\'')
            sourcetype_descriptions = {
                'linux_secure': 'Linux authentication/security logs',
                'linux_audit': 'Linux audit logs',
                'WinEventLog:Security': 'Windows Security Event Log',
                'WinEventLog:System': 'Windows System Event Log',
                'access_combined': 'Apache/Nginx access logs',
                'syslog': 'Syslog messages',
                'cisco:asa': 'Cisco ASA firewall logs',
                'pan:traffic': 'Palo Alto traffic logs',
            }
            desc = sourcetype_descriptions.get(sourcetype, f'{sourcetype} log format')
            input_parts.append(f"sourcetype: {sourcetype} ({desc})")

        # Extract action/status filters
        action_match = re.search(r'action=(\S+)', query)
        if action_match:
            action = action_match.group(1).strip('"\'')
            input_parts.append(f"action filter: {action}")

        status_match = re.search(r'status=(\S+)', query)
        if status_match:
            status = status_match.group(1).strip('"\'')
            input_parts.append(f"status filter: {status}")

        # Extract search terms (quoted strings)
        search_terms = re.findall(r'"([^"]+)"', query)
        if search_terms:
            for term in search_terms[:3]:  # Limit to first 3
                input_parts.append(f'search term: "{term}"')

        # Extract time range
        time_patterns = [
            (r'earliest=(-?\d+[hdwm]@?\w*)', 'time range'),
            (r'latest=(\S+)', 'end time'),
            (r'_time\s*[<>=]+\s*(\S+)', 'time filter'),
        ]
        for pattern, label in time_patterns:
            match = re.search(pattern, query)
            if match:
                input_parts.append(f"{label}: {match.group(1)}")
                break

        # Extract IP filters
        ip_patterns = [
            (r'src_ip=[\"\']?([^\s\"\']+)', 'source IP filter'),
            (r'dest_ip=[\"\']?([^\s\"\']+)', 'destination IP filter'),
            (r'src=[\"\']?([^\s\"\']+)', 'source filter'),
            (r'dst=[\"\']?([^\s\"\']+)', 'destination filter'),
        ]
        for pattern, label in ip_patterns:
            match = re.search(pattern, query)
            if match:
                input_parts.append(f"{label}: {match.group(1)}")

        # Extract user filters
        user_match = re.search(r'user=[\"\']?([^\s\"\']+)', query)
        if user_match:
            input_parts.append(f"user filter: {user_match.group(1)}")

        # Analyze pipe commands for output
        pipe_parts = query.split('|')

        for part in pipe_parts[1:]:  # Skip the search part
            part = part.strip()

            # Stats command
            if part.startswith('stats'):
                stats_match = re.search(r'stats\s+(\w+)(?:\([^)]*\))?\s*(?:as\s+\w+)?\s*(?:by\s+(.+))?', part)
                if stats_match:
                    func = stats_match.group(1)
                    by_fields = stats_match.group(2)
                    if by_fields:
                        output_parts.append(f"Calculates {func} grouped by: {by_fields.strip()}")
                    else:
                        output_parts.append(f"Calculates {func} of results")

            # Table command
            elif part.startswith('table'):
                fields_match = re.search(r'table\s+(.+)', part)
                if fields_match:
                    fields = fields_match.group(1).strip()
                    # Limit displayed fields if too many
                    field_list = [f.strip() for f in fields.split(',')]
                    if len(field_list) > 5:
                        output_parts.append(f"Displays fields: {', '.join(field_list[:5])}... and {len(field_list)-5} more")
                    else:
                        output_parts.append(f"Displays fields: {fields}")

            # Sort command
            elif part.startswith('sort'):
                sort_match = re.search(r'sort\s+(-?)(\S+)', part)
                if sort_match:
                    direction = 'descending' if sort_match.group(1) == '-' else 'ascending'
                    field = sort_match.group(2)
                    output_parts.append(f"Sorted by {field} ({direction})")

            # Head/tail command
            elif part.startswith('head'):
                head_match = re.search(r'head\s+(\d+)', part)
                if head_match:
                    output_parts.append(f"Limited to top {head_match.group(1)} results")

            elif part.startswith('tail'):
                tail_match = re.search(r'tail\s+(\d+)', part)
                if tail_match:
                    output_parts.append(f"Shows last {tail_match.group(1)} results")

            # Timechart
            elif part.startswith('timechart'):
                output_parts.append("Creates a time-based chart of results")

            # Dedup
            elif part.startswith('dedup'):
                dedup_match = re.search(r'dedup\s+(.+)', part)
                if dedup_match:
                    output_parts.append(f"Removes duplicates based on: {dedup_match.group(1).strip()}")

            # Where clause
            elif part.startswith('where'):
                output_parts.append("Applies additional filtering conditions")

            # Eval
            elif part.startswith('eval'):
                output_parts.append("Calculates/transforms field values")

            # Rex (regex extraction)
            elif part.startswith('rex'):
                output_parts.append("Extracts fields using regex patterns")

        # Generate summary based on instruction
        summary = f"This query searches for {instruction.lower().rstrip('.')}."

        # Build the explanation
        explanation_parts = [summary, "", "### Input:"]

        if input_parts:
            for part in input_parts:
                explanation_parts.append(f"- {part}")
        else:
            explanation_parts.append("- Searches across available data")

        explanation_parts.append("")
        explanation_parts.append("### Output:")

        if output_parts:
            for part in output_parts:
                explanation_parts.append(f"- {part}")
        else:
            explanation_parts.append("- Returns matching raw events")

        return '\n'.join(explanation_parts)
