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
        return response.strip().startswith("CLARIFICATION:")

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
        Generate an explanation for a Splunk query.

        Args:
            query: The generated Splunk query to explain
            instruction: The original user instruction
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Explanation of what the query does
        """
        explanation_prompt = f"""### Instruction:
Explain this Splunk query for a security analyst using the exact format below. Be concise and specific.

Original request: {instruction}

Splunk Query:
{query}

Provide the explanation in this exact format:
1. First, a one-sentence summary of what the query does
2. Then an "### Input:" section listing each component (index, sourcetype, search terms, time range, etc.)
3. Then an "### Output:" section describing what results will be displayed

Example format:
Show me all failed SSH logins from the last 24 hours, sorted by frequency with the top 10 IPs/users.

### Input:
- index: linux (operating system logs)
- sourcetype: linux_secure (authentication logs)
- search term: "Failed password" (failed login attempts)
- time range: last 24 hours

### Output:
- Count of failed attempts grouped by source IP and username
- Sorted by frequency, showing top 10 results
- Displays timestamp, source IP, username, and count

### Response:
"""

        # Tokenize
        inputs = self.tokenizer(
            explanation_prompt,
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
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part
        if "### Response:" in text:
            explanation = text.split("### Response:")[-1].strip()
        else:
            explanation = text[len(explanation_prompt):].strip()

        return explanation
