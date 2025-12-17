#!/usr/bin/env python3
"""Command-line interface for GBox CUA Agent.

This allows the package to be run as a standalone agent.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="GBox CUA Agent - Standalone agent for computer use tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    gbox-cua "Open the Settings app"
    gbox-cua --task "Search for weather" --box-type android --verbose
    gbox-cua --task "Navigate to gmail.com" --vllm-api-base http://localhost:8000/v1
        """,
    )
    
    parser.add_argument(
        "task",
        nargs="?",
        help="Task description",
    )
    parser.add_argument(
        "--task", "-t",
        dest="task_arg",
        help="Task description (alternative to positional)",
    )
    parser.add_argument(
        "--box-type", "-b",
        default="android",
        choices=["android", "linux"],
        help="Type of GBox environment (default: android)",
    )
    parser.add_argument(
        "--gbox-api-key",
        help="GBox API key (or set GBOX_API_KEY env var)",
    )
    parser.add_argument(
        "--vllm-api-base",
        help="vLLM server URL (e.g., http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: gpt-4o for OpenAI, or from MODEL_NAME env var)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum turns (default: 20)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--vlm-provider",
        choices=["vllm", "openrouter", "openai"],
        help="VLM provider: 'vllm' (local vLLM server), 'openrouter' (OpenRouter API), or 'openai' (OpenAI API)",
    )
    parser.add_argument(
        "--openrouter-api-key",
        help="OpenRouter API key (required if --vlm-provider=openrouter)",
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (required if --vlm-provider=openai)",
    )
    
    args = parser.parse_args()
    
    # Get task from either positional or named argument
    task = args.task or args.task_arg
    if not task:
        parser.error("Task description is required")
    
    # Use standalone agent implementation
    asyncio.run(run_standalone_agent(args, task))


async def run_standalone_agent(args, task: str):
    """Run agent using standalone implementation."""
    from gbox_cua.agent import StandaloneGBoxCUAAgent
    
    # Get GBox API key
    gbox_api_key = args.gbox_api_key or os.environ.get("GBOX_API_KEY")
    if not gbox_api_key:
        raise ValueError(
            "GBOX_API_KEY not provided. Set it via --gbox-api-key or GBOX_API_KEY env var"
        )
    
    # Determine VLM provider
    vlm_provider = args.vlm_provider or os.getenv("VLM_PROVIDER", "openai").lower()
    if vlm_provider not in ["vllm", "openai", "openrouter"]:
        raise ValueError(f"Invalid VLM_PROVIDER: {vlm_provider}. Must be 'vllm', 'openai' or 'openrouter'")
    
    # Get VLM API configuration
    vlm_api_base = None
    vlm_api_key = None
    
    if vlm_provider == "vllm":
        vlm_api_base = args.vllm_api_base or os.environ.get("VLLM_API_BASE")
        if not vlm_api_base:
            raise ValueError("VLLM_API_BASE is required when VLM_PROVIDER=vllm")
        vlm_api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    elif vlm_provider == "openai":
        vlm_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not vlm_api_key:
            raise ValueError("OPENAI_API_KEY is required when VLM_PROVIDER=openai")
        vlm_api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    elif vlm_provider == "openrouter":
        vlm_api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not vlm_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when VLM_PROVIDER=openrouter")
        vlm_api_base = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    
    # Get model name with provider-specific defaults
    if args.model:
        model_name = args.model
    elif os.getenv("MODEL_NAME"):
        model_name = os.getenv("MODEL_NAME")
    else:
        # Provider-specific defaults
        if vlm_provider == "openai":
            model_name = "gpt-4o"
        elif vlm_provider == "openrouter":
            model_name = "qwen/qwen-2-vl-7b-instruct"  # Common OpenRouter vision model
        else:  # vllm
            model_name = "unsloth/Qwen3-VL-30B-A3B-Instruct"
    
    logger.info(f"Using model: {model_name} with provider: {vlm_provider}")
    
    # Create agent
    agent = StandaloneGBoxCUAAgent(
        gbox_api_key=gbox_api_key,
        vlm_provider=vlm_provider,
        vlm_api_base=vlm_api_base,
        vlm_api_key=vlm_api_key,
        model_name=model_name,
        max_turns=args.max_turns,
        max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        top_p=float(os.getenv("TOP_P", "0.9")),
    )
    
    try:
        # Run task
        result, history = await agent.run_task(
            task_description=task,
            box_type=args.box_type,
            verbose=args.verbose,
        )
        
        # Print results
        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"Task: {task}")
        print(f"Success: {result['task_success']}")
        print(f"Completed: {result['task_completed']}")
        print(f"Turns: {result['num_turns']}/{result['max_turns']}")
        print(f"Message: {result['result_message']}")
        print("="*60)
        
        sys.exit(0 if result['task_success'] else 1)
    finally:
        await agent.close()


if __name__ == "__main__":
    main()

