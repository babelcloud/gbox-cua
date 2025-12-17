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
        default="unsloth/Qwen3-VL-30B-A3B-Instruct",
        help="Model name (default: unsloth/Qwen3-VL-30B-A3B-Instruct)",
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
    
    # Check if rl-cua is available for full agent functionality
    try:
        from cua_agent.config import CUAConfig, GBoxConfig
        from cua_agent.agent import CUAAgent, calculate_reward
        from cua_agent.vlm_inference import VLMInference
        
        # Use rl-cua's full agent implementation
        asyncio.run(run_with_rl_cua(args, task))
    except ImportError:
        logger.warning(
            "rl-cua not available. For full agent functionality, install rl-cua or use as library."
        )
        logger.info("This package is primarily designed to be used as a library.")
        logger.info("For standalone agent, please install rl-cua: pip install -e /path/to/rl-cua")
        sys.exit(1)


async def run_with_rl_cua(args, task: str):
    """Run agent using rl-cua's full implementation."""
    from cua_agent.config import CUAConfig, GBoxConfig
    from cua_agent.agent import CUAAgent, calculate_reward
    
    # Get API key
    gbox_api_key = args.gbox_api_key or os.environ.get("GBOX_API_KEY")
    if not gbox_api_key:
        raise ValueError(
            "GBOX_API_KEY not provided. Set it via --gbox-api-key or GBOX_API_KEY env var"
        )
    
    # Determine VLM provider
    vlm_provider = args.vlm_provider or os.getenv("VLM_PROVIDER", "vllm").lower()
    
    # Get OpenRouter API key if needed
    openrouter_api_key = None
    if vlm_provider == "openrouter":
        openrouter_api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when using OpenRouter provider. "
                "Get your API key from https://openrouter.ai"
            )
    
    # Create config
    gbox_config = GBoxConfig(
        api_key=gbox_api_key,
        box_type=args.box_type,
    )
    
    openrouter_api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    
    config = CUAConfig(
        model_name=args.model,
        max_turns=args.max_turns,
        vlm_provider=vlm_provider,
        vllm_api_base=args.vllm_api_base,
        openrouter_api_key=openrouter_api_key,
        openrouter_api_base=openrouter_api_base,
        gbox=gbox_config,
    )
    
    # Run agent
    async with CUAAgent(config) as agent:
        rubric, history = await agent.run_task(
            task_description=task,
            box_type=args.box_type,
            verbose=args.verbose,
        )
    
    # Calculate reward
    reward = calculate_reward(config, rubric)
    
    result = {
        "task": task,
        "rubric": rubric.to_dict(),
        "reward": reward,
        "num_steps": len(history),
    }
    
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"Task: {result['task']}")
    print(f"Success: {result['rubric']['task_success']}")
    print(f"Completed: {result['rubric']['task_completed']}")
    print(f"Turns: {result['rubric']['num_turns']}/{result['rubric']['max_turns']}")
    print(f"Reward: {result['reward']:.3f}")
    print(f"Message: {result['rubric']['result_message']}")
    print("="*60)
    
    sys.exit(0 if result['rubric']['task_success'] else 1)


if __name__ == "__main__":
    main()

