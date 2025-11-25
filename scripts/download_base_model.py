#!/usr/bin/env python3
"""
Pre-download and cache the base model before starting the server.
This prevents timeouts and connection issues during server startup.
Uses file locking to ensure only one instance downloads at a time.
"""

import os
import sys
import time
import fcntl
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
    Download and cache the base model with file locking.
    Only one instance will download; others will wait or skip if already downloaded.

    Args:
        base_model: HuggingFace model identifier
        cache_dir: Cache directory (defaults to HF_HOME or ~/.cache/huggingface)
    """
    # Determine cache directory
    if cache_dir is None:
        cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create lock and marker files
    lock_file_path = cache_path / f".{base_model.replace('/', '_')}_download.lock"
    marker_file_path = cache_path / f".{base_model.replace('/', '_')}_downloaded.marker"

    logger.info("=" * 60)
    logger.info("Zeus Base Model Downloader")
    logger.info("=" * 60)
    logger.info(f"Base Model: {base_model}")
    logger.info(f"Cache Dir: {cache_dir}")
    logger.info("")

    # Check if already downloaded
    if marker_file_path.exists():
        logger.info("✓ Base model already cached (found marker file)")
        logger.info("  Skipping download.")
        logger.info("=" * 60)
        return True

    # Acquire lock
    logger.info("Acquiring download lock...")
    lock_file = None
    try:
        lock_file = open(lock_file_path, 'w')

        # Try to acquire lock (non-blocking first to show message)
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.info("✓ Lock acquired - this instance will download the model")
            logger.info("")
            is_downloader = True
        except IOError:
            logger.info("⏳ Another instance is downloading the model...")
            logger.info("   Waiting for download to complete...")
            logger.info("")
            is_downloader = False

            # Wait for lock (blocking)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            # Check if download completed while we waited
            if marker_file_path.exists():
                logger.info("✓ Model download completed by another instance")
                logger.info("=" * 60)
                return True

            # If no marker, we need to download (previous attempt failed)
            logger.info("⚠ Previous download attempt failed, retrying...")
            is_downloader = True

        if not is_downloader:
            return True

    except Exception as e:
        logger.error(f"Error acquiring lock: {e}")
        if lock_file:
            lock_file.close()
        return False

    # Perform download
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

        # Create marker file to indicate successful download
        marker_file_path.write_text(f"Downloaded: {base_model}\nTimestamp: {time.time()}\n")

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

    finally:
        # Release lock
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except:
                pass


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
