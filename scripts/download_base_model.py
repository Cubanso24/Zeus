#!/usr/bin/env python3
"""
Pre-download and cache the base model before starting the server.
This prevents timeouts and connection issues during server startup.
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def download_base_model(
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    cache_dir: str = None,
):
    """
    Download and cache the base model.

    Args:
        base_model: HuggingFace model identifier
        cache_dir: Cache directory (defaults to HF_HOME or ~/.cache/huggingface)
    """
    logger.info("=" * 60)
    logger.info("Zeus Base Model Downloader")
    logger.info("=" * 60)
    logger.info(f"Base Model: {base_model}")
    logger.info(f"Cache Dir: {cache_dir or os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    logger.info("")

    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        logger.info("✓ Tokenizer downloaded successfully")

        # Download model
        logger.info(f"Downloading base model (this may take 5-10 minutes)...")
        logger.info("Model size: ~14GB")

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="cpu",  # Keep on CPU for download only
        )
        logger.info("✓ Base model downloaded successfully")

        # Clean up to free memory
        del model
        del tokenizer

        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ Base model cached successfully!")
        logger.info("  Zeus can now start quickly without downloading.")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Failed to download base model: {e}")
        logger.error("")
        logger.error("Common issues:")
        logger.error("  1. No internet connection")
        logger.error("  2. HuggingFace is down or slow")
        logger.error("  3. Insufficient disk space")
        logger.error("  4. Model requires authentication (need HF token)")
        logger.error("")
        logger.error("Solutions:")
        logger.error("  - Check internet connection")
        logger.error("  - Ensure at least 20GB free disk space")
        logger.error("  - For gated models, set HF_TOKEN environment variable")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Zeus base model")
    parser.add_argument(
        "--base-model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model to download"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory (defaults to HF_HOME)"
    )

    args = parser.parse_args()

    success = download_base_model(
        base_model=args.base_model,
        cache_dir=args.cache_dir,
    )

    sys.exit(0 if success else 1)
