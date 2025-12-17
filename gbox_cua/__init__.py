"""GBox CUA Agent.

This package can be used as:
1. Standalone agent - run directly or in Docker
2. Library - imported by rl-cua, OSWorld-provider, or other projects
"""

from gbox_cua.tools import (
    get_tools_schema,
    tool_call_to_action_dict,
    PERFORM_ACTION_TOOL,
    SLEEP_TOOL,
    REPORT_TASK_COMPLETE_TOOL,
)
from gbox_cua.prompts import (
    create_system_prompt,
    create_user_message_with_screenshot,
)
from gbox_cua.gbox_coordinate import GBoxCoordinateGenerator

__version__ = "0.1.0"

__all__ = [
    "get_tools_schema",
    "tool_call_to_action_dict",
    "PERFORM_ACTION_TOOL",
    "SLEEP_TOOL",
    "REPORT_TASK_COMPLETE_TOOL",
    "create_system_prompt",
    "create_user_message_with_screenshot",
    "GBoxCoordinateGenerator",
]

