"""Standalone GBox CUA Agent implementation.

This agent can run independently without requiring rl-cua or OSWorld.
It uses GBox SDK for box management and action execution, and supports
vllm, openai, and openrouter VLM providers.

Features:
- Detailed human-readable logging with buffered output
- Complete model output logging (thinking, content, tool calls)
- Per-rollout log buffer for parallel execution support
"""

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from typing import List, Dict, Any, Optional, Tuple

from gbox_sdk import GboxSDK
import httpx

from gbox_cua.tools import get_tools_schema, tool_call_to_action_dict
from gbox_cua.prompts import create_system_prompt, create_user_message_with_screenshot
from gbox_cua.gbox_coordinate import GBoxCoordinateGenerator

logger = logging.getLogger(__name__)


# ============================================================================
# Rollout Logger - Human-Friendly Log Buffer
# ============================================================================

@dataclass
class TurnLog:
    """Log for a single turn in the rollout."""
    turn_number: int
    max_turns: int
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Screenshot info
    screenshot_time: float = 0.0
    screenshot_size_kb: float = 0.0
    
    # VLM info
    vlm_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Model output
    thinking: str = ""
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    # Action execution
    action_type: str = ""
    action_params: Dict[str, Any] = field(default_factory=dict)
    action_time: float = 0.0
    action_success: bool = False
    action_error: str = ""
    action_result: Dict[str, Any] = field(default_factory=dict)
    
    # Task completion (if applicable)
    task_completed: bool = False
    task_success: bool = False
    task_message: str = ""


class RolloutLogger:
    """Human-friendly rollout logger with buffered output.
    
    Collects all log entries during rollout execution and outputs
    a formatted, easy-to-read log at the end. Supports parallel rollouts
    with independent log buffers.
    """
    
    def __init__(self, task_id: str = "", task_description: str = ""):
        self.task_id = task_id or f"task_{int(time.time())}"
        self.task_description = task_description
        self.start_time = time.time()
        self.end_time: float = 0.0
        
        # Box info
        self.box_id: str = ""
        self.box_type: str = ""
        self.box_create_time: float = 0.0
        
        # Model info
        self.model_name: str = ""
        self.provider: str = ""
        
        # Turn logs
        self.turns: List[TurnLog] = []
        self.current_turn: Optional[TurnLog] = None
        
        # Final result
        self.final_success: bool = False
        self.final_message: str = ""
        self.total_turns: int = 0
        self.max_turns: int = 0
        
        # Errors
        self.errors: List[str] = []
    
    def set_box_info(self, box_id: str, box_type: str, create_time: float):
        """Set box information."""
        self.box_id = box_id
        self.box_type = box_type
        self.box_create_time = create_time
    
    def set_model_info(self, model_name: str, provider: str):
        """Set model information."""
        self.model_name = model_name
        self.provider = provider
    
    def start_turn(self, turn_number: int, max_turns: int) -> TurnLog:
        """Start a new turn and return the turn log."""
        turn_log = TurnLog(
            turn_number=turn_number,
            max_turns=max_turns,
            start_time=time.time(),
        )
        self.current_turn = turn_log
        self.turns.append(turn_log)
        return turn_log
    
    def end_turn(self):
        """End the current turn."""
        if self.current_turn:
            self.current_turn.end_time = time.time()
        self.current_turn = None
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(f"[{datetime.now().strftime('%H:%M:%S')}] {error}")
    
    def set_final_result(self, success: bool, message: str, total_turns: int, max_turns: int):
        """Set the final result."""
        self.end_time = time.time()
        self.final_success = success
        self.final_message = message
        self.total_turns = total_turns
        self.max_turns = max_turns
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
    
    def _truncate_text(self, text: str, max_length: int = 500) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def _format_json(self, obj: Any, indent: int = 2) -> str:
        """Format object as indented JSON string."""
        try:
            return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)
        except:
            return str(obj)
    
    def _display_width(self, text: str) -> int:
        """Calculate display width of text, accounting for wide characters (emoji, CJK)."""
        import unicodedata
        width = 0
        for char in text:
            # East Asian Width
            ea_width = unicodedata.east_asian_width(char)
            if ea_width in ('F', 'W'):  # Fullwidth or Wide
                width += 2
            elif ea_width == 'A':  # Ambiguous - treat as wide in terminal
                width += 2
            else:
                width += 1
        return width
    
    def _pad_line(self, text: str, target_width: int) -> str:
        """Pad text to target display width."""
        current_width = self._display_width(text)
        if current_width >= target_width:
            return text
        return text + " " * (target_width - current_width)
    
    def _make_line(self, content: str, width: int = 76) -> str:
        """Create a table line with proper padding."""
        return "│ " + self._pad_line(content, width) + " │\n"
    
    def format_log(self) -> str:
        """Format the complete rollout log as a human-readable string."""
        buf = StringIO()
        w = buf.write
        W = 76  # Content width (total 80 with borders)
        
        total_time = self.end_time - self.start_time if self.end_time else time.time() - self.start_time
        
        # Header
        w("\n")
        w("╔" + "═" * 78 + "╗\n")
        w("║" + " ROLLOUT EXECUTION LOG ".center(78) + "║\n")
        w("╚" + "═" * 78 + "╝\n")
        w("\n")
        
        # Task Info
        w("┌─ TASK INFO " + "─" * 66 + "┐\n")
        w(self._make_line(f"Task ID:     {self.task_id[:60]}", W))
        desc_line = self.task_description[:65] if len(self.task_description) <= 65 else self.task_description[:62] + "..."
        w(self._make_line(f"Description: {desc_line}", W))
        w(self._make_line(f"Started:     {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}", W))
        w(self._make_line(f"Duration:    {self._format_duration(total_time)}", W))
        w("└" + "─" * 78 + "┘\n")
        w("\n")
        
        # Environment Info
        w("┌─ ENVIRONMENT " + "─" * 64 + "┐\n")
        w(self._make_line(f"Box ID:      {self.box_id}", W))
        w(self._make_line(f"Box Type:    {self.box_type}", W))
        w(self._make_line(f"Create Time: {self._format_duration(self.box_create_time)}", W))
        w(self._make_line(f"Model:       {self.model_name[:60]}", W))
        w(self._make_line(f"Provider:    {self.provider}", W))
        w("└" + "─" * 78 + "┘\n")
        w("\n")
        
        # Turn-by-Turn Log
        w("┌─ TURN-BY-TURN EXECUTION " + "─" * 53 + "┐\n")
        w(self._make_line("", W))
        
        for turn in self.turns:
            turn_duration = turn.end_time - turn.start_time if turn.end_time else 0
            
            # Turn header
            w("│ " + "─" * 76 + " │\n")
            w(self._make_line(f"TURN {turn.turn_number}/{turn.max_turns}  [{self._format_duration(turn_duration)}]", W))
            w("│ " + "─" * 76 + " │\n")
            
            # Screenshot (use text instead of emoji for alignment)
            screenshot_info = f"[Screenshot] {turn.screenshot_size_kb:.1f} KB ({self._format_duration(turn.screenshot_time)})"
            w(self._make_line(screenshot_info, W))
            
            # VLM Response
            vlm_info = f"[VLM] {self._format_duration(turn.vlm_time)}, tokens: {turn.total_tokens} (in: {turn.prompt_tokens}, out: {turn.completion_tokens})"
            w(self._make_line(vlm_info, W))
            
            # Thinking (if present)
            if turn.thinking:
                w(self._make_line("", W))
                w(self._make_line("[Thinking]", W))
                thinking_lines = turn.thinking.split('\n')
                for line in thinking_lines[:10]:
                    line_text = self._truncate_text(line.strip(), 72)
                    w(self._make_line(f"  {line_text}", W))
                if len(thinking_lines) > 10:
                    w(self._make_line(f"  ... ({len(thinking_lines) - 10} more lines)", W))
            
            # Content (if present)
            if turn.content:
                w(self._make_line("", W))
                w(self._make_line("[Content]", W))
                content_lines = turn.content.split('\n')
                for line in content_lines[:10]:
                    line_text = self._truncate_text(line.strip(), 72)
                    w(self._make_line(f"  {line_text}", W))
                if len(content_lines) > 10:
                    w(self._make_line(f"  ... ({len(content_lines) - 10} more lines)", W))
            
            # Tool Calls
            if turn.tool_calls:
                w(self._make_line("", W))
                w(self._make_line(f"[Tool Calls] ({len(turn.tool_calls)})", W))
                for i, tc in enumerate(turn.tool_calls):
                    func_name = tc.get("function", {}).get("name", "unknown")
                    w(self._make_line(f"  [{i+1}] {func_name}", W))
                    
                    # Parse and display arguments
                    try:
                        args_str = tc.get("function", {}).get("arguments", "{}")
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        args_formatted = self._format_json(args, indent=2)
                        for arg_line in args_formatted.split('\n')[:15]:
                            line_text = self._truncate_text(arg_line, 68)
                            w(self._make_line(f"      {line_text}", W))
                    except Exception as e:
                        w(self._make_line(f"      (failed to parse: {str(e)[:50]})", W))
            
            # Action Execution Result
            if turn.action_type:
                w(self._make_line("", W))
                status = "OK" if turn.action_success else "FAILED"
                w(self._make_line(f"[Action Result] {turn.action_type} - {status} ({self._format_duration(turn.action_time)})", W))
                
                if turn.action_error:
                    error_text = self._truncate_text(turn.action_error, 68)
                    w(self._make_line(f"  Error: {error_text}", W))
                elif turn.action_result:
                    result_str = self._format_json(turn.action_result)
                    for result_line in result_str.split('\n')[:5]:
                        line_text = self._truncate_text(result_line, 70)
                        w(self._make_line(f"  {line_text}", W))
            
            # Task Completion
            if turn.task_completed:
                w(self._make_line("", W))
                status = "SUCCESS" if turn.task_success else "FAILED"
                w(self._make_line(f">>> TASK COMPLETED: {status}", W))
                if turn.task_message:
                    msg_text = self._truncate_text(turn.task_message, 68)
                    w(self._make_line(f"    Message: {msg_text}", W))
            
            w(self._make_line("", W))
        
        w("└" + "─" * 78 + "┘\n")
        w("\n")
        
        # Errors (if any)
        if self.errors:
            w("┌─ ERRORS " + "─" * 69 + "┐\n")
            for error in self.errors:
                error_text = self._truncate_text(error, 72)
                w(self._make_line(f"! {error_text}", W))
            w("└" + "─" * 78 + "┘\n")
            w("\n")
        
        # Final Summary
        result_text = "SUCCESS" if self.final_success else "FAILED"
        w("╔" + "═" * 78 + "╗\n")
        w("║ " + self._pad_line(f"FINAL RESULT: {result_text}", 76) + " ║\n")
        w("╠" + "═" * 78 + "╣\n")
        w("║ " + self._pad_line(f"Turns Used:   {self.total_turns}/{self.max_turns}", 76) + " ║\n")
        w("║ " + self._pad_line(f"Total Time:   {self._format_duration(total_time)}", 76) + " ║\n")
        msg_line = self.final_message[:65] if len(self.final_message) <= 65 else self.final_message[:62] + "..."
        w("║ " + self._pad_line(f"Message:      {msg_line}", 76) + " ║\n")
        w("╚" + "═" * 78 + "╝\n")
        w("\n")
        
        return buf.getvalue()
    
    def print_log(self):
        """Print the formatted log to stdout."""
        print(self.format_log())


# ============================================================================
# VLM Inference
# ============================================================================

class VLMInference:
    """Vision-Language Model inference for standalone agent.
    
    Supports vllm, openai, and openrouter providers.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        provider: str = "openai",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Initialize VLM inference."""
        self.model_name = model_name
        self.provider = provider.lower()
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        self._client: Optional[httpx.AsyncClient] = None
        
        if self.provider not in ["vllm", "openai", "openrouter"]:
            raise ValueError(f"Invalid provider: {provider}. Must be 'vllm', 'openai' or 'openrouter'")
        
        # Setup API client
        if self.provider == "vllm":
            if not api_base:
                raise ValueError("api_base is required when using vLLM provider")
            self._client = httpx.AsyncClient(
                base_url=api_base,
                headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
                timeout=300.0,
            )
        elif self.provider == "openrouter":
            api_base = api_base or "https://openrouter.ai/api/v1"
            if not api_key:
                raise ValueError("api_key is required when using OpenRouter provider")
            self._client = httpx.AsyncClient(
                base_url=api_base,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/babelcloud/gbox-cua",
                    "X-Title": "GBox CUA Agent",
                },
                timeout=300.0,
            )
        elif self.provider == "openai":
            api_base = api_base or "https://api.openai.com/v1"
            if not api_key:
                raise ValueError("api_key is required when using OpenAI provider")
            self._client = httpx.AsyncClient(
                base_url=api_base,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=300.0,
            )
    
    def _encode_image(self, image_data: bytes) -> str:
        """Encode image bytes to base64 data URI."""
        base64_data = base64.b64encode(image_data).decode("utf-8")
        
        if image_data[:8] == b'\x89PNG\r\n\x1a\n':
            mime_type = "image/png"
        elif image_data[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        else:
            mime_type = "image/png"
        
        return f"data:{mime_type};base64,{base64_data}"
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        image_data: Optional[bytes] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate a response from the VLM."""
        # Prepare messages with image
        api_messages = []
        for msg in messages:
            if msg["role"] == "user" and image_data:
                content = msg.get("content", "")
                image_uri = self._encode_image(image_data)
                api_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": content}
                    ]
                })
                image_data = None  # Only add to first user message
            else:
                api_messages.append(msg)
        
        # Map model name for OpenRouter
        model_name = self.model_name
        if self.provider == "openrouter":
            model_mapping = {
                "unsloth/Qwen3-VL-30B-A3B-Instruct": "qwen/qwen3-vl-30b-a3b-instruct",
                "Qwen3-VL-30B-A3B-Instruct": "qwen/qwen3-vl-30b-a3b-instruct",
                "qwen/Qwen3-VL-30B-A3B-Instruct": "qwen/qwen3-vl-30b-a3b-instruct",
                "Qwen/Qwen3-VL-30B-A3B-Instruct": "qwen/qwen3-vl-30b-a3b-instruct",
                "unsloth/Qwen3-VL-32B-Instruct": "qwen/qwen3-vl-32b-instruct",
                "Qwen3-VL-32B-Instruct": "qwen/qwen3-vl-32b-instruct",
                "unsloth/Qwen3-VL-8B-Instruct": "qwen/qwen3-vl-8b-instruct",
                "Qwen3-VL-8B-Instruct": "qwen/qwen3-vl-8b-instruct",
                "Qwen/Qwen2.5-VL-32B-Instruct": "qwen/qwen2.5-vl-32b-instruct",
            }
            model_name = model_mapping.get(model_name, model_name)
            if "/" in model_name and model_name not in model_mapping.values():
                parts = model_name.split("/")
                if len(parts) == 2:
                    model_name = f"{parts[0].lower()}/{parts[1].lower()}"
        
        payload = {
            "model": model_name,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            provider_name = {"openrouter": "OpenRouter", "vllm": "vLLM", "openai": "OpenAI"}.get(self.provider, self.provider)
            error_detail = ""
            try:
                error_body = e.response.json()
                error_detail = f": {error_body}"
            except:
                error_detail = f": {e.response.text}"
            raise RuntimeError(f"{provider_name} API error: {e.response.status_code}{error_detail}") from e
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            provider_name = {"openrouter": "OpenRouter", "vllm": "vLLM", "openai": "OpenAI"}.get(self.provider, self.provider)
            raise RuntimeError(f"{provider_name} connection error: {e}") from e
        
        return response.json()
    
    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.aclose()


# ============================================================================
# GBox Client (Standalone version for agent.py)
# ============================================================================

class GBoxClient:
    """GBox API client for box management and action execution."""
    
    def __init__(self, api_key: str):
        """Initialize GBox client."""
        self.api_key = api_key
        self._sdk = GboxSDK(api_key=api_key)
        self.box_id: Optional[str] = None
        self._box: Optional[Any] = None
    
    async def create_box(self, box_type: str = "android") -> Dict[str, Any]:
        """Create a new GBox environment."""
        logger.debug(f"Creating {box_type} box...")
        
        box = self._sdk.create(
            type=box_type,
            wait=True,
            timeout="60s",
        )
        
        self._box = box
        self.box_id = box.data.id
        logger.debug(f"Box created: {self.box_id}")
        return {"id": self.box_id}
    
    def _get_box(self) -> Any:
        """Get box operator."""
        if self._box:
            return self._box
        if self.box_id:
            self._box = self._sdk.get(self.box_id)
            return self._box
        raise ValueError("No box available. Call create_box() first.")
    
    async def take_screenshot(self) -> Tuple[bytes, str]:
        """Take a screenshot of the box display."""
        if not self.box_id:
            raise ValueError("No box ID available. Create a box first.")
        
        box = self._get_box()
        result = box.action.screenshot(output_format="base64")
        screenshot_uri = result.uri
        
        if screenshot_uri.startswith("data:"):
            parts = screenshot_uri.split(",", 1)
            image_bytes = base64.b64decode(parts[1])
            return image_bytes, screenshot_uri
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.get(screenshot_uri)
                resp.raise_for_status()
                image_bytes = resp.content
                data_uri = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                return image_bytes, data_uri
    
    async def execute_action(self, action: Dict[str, Any], screenshot_uri: str) -> Dict[str, Any]:
        """Execute an action using GBox API."""
        if not self.box_id:
            raise ValueError("No box ID available")
        
        action_type = action.get("action_type")
        box = self._get_box()
        
        coord_generator = GBoxCoordinateGenerator(
            api_key=self.api_key,
            model="gbox-handy-1"
        )
        
        def target_to_desc(t: Any) -> str:
            """Convert target dict to description string."""
            if isinstance(t, dict):
                parts = [t.get("element", "")]
                if t.get("label"):
                    parts.append(f'labeled "{t["label"]}"')
                if t.get("color"):
                    parts.append(f"{t['color']} colored")
                if t.get("size"):
                    parts.append(f"{t['size']} sized")
                if t.get("location"):
                    parts.append(f"located at {t['location']}")
                if t.get("shape"):
                    parts.append(f"with {t['shape']} shape")
                return " ".join(parts) if parts[0] else "center of the screen"
            return str(t) if t else "center of the screen"
        
        logger.debug(f"Executing action: {action_type}")
        
        if action_type == "click":
            target_desc = target_to_desc(action.get("target", {}))
            result = await coord_generator.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="click",
                target=target_desc,
            )
            coords = result.get("response", {}).get("coordinates", {}) or result.get("coordinates", {})
            x, y = coords.get("x", 0), coords.get("y", 0)
            
            button = action.get("option", "left")
            double_click = button == "double"
            
            result = box.action.click(x=x, y=y, button=button, double=double_click)
            return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
        
        elif action_type == "swipe":
            start_desc = target_to_desc(action.get("start_target", {}))
            end_desc = target_to_desc(action.get("end_target", {}))
            
            drag_result = await coord_generator.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="drag",
                target=start_desc,
                end_target=end_desc,
            )
            
            response_data = drag_result.get("response", {}) or drag_result
            coordinates = response_data.get("coordinates", {})
            
            if "start" in coordinates and "end" in coordinates:
                start_coords = coordinates.get("start", {})
                end_coords = coordinates.get("end", {})
            else:
                start_result = await coord_generator.generate_coordinates(
                    screenshot_uri=screenshot_uri,
                    action_type="click",
                    target=start_desc,
                )
                end_result = await coord_generator.generate_coordinates(
                    screenshot_uri=screenshot_uri,
                    action_type="click",
                    target=end_desc,
                )
                start_coords = (start_result.get("response", {}) or start_result).get("coordinates", {})
                end_coords = (end_result.get("response", {}) or end_result).get("coordinates", {})
            
            result = box.action.swipe(
                start={"x": start_coords.get("x", 0), "y": start_coords.get("y", 0)},
                end={"x": end_coords.get("x", 0), "y": end_coords.get("y", 0)},
                duration="300ms",
            )
            return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
        
        elif action_type == "scroll":
            target_desc = target_to_desc(action.get("target", {}))
            direction = action.get("direction", "down")
            
            result = await coord_generator.generate_coordinates(
                screenshot_uri=screenshot_uri,
                action_type="scroll",
                target=target_desc,
                direction=direction,
            )
            coords = (result.get("response", {}) or result).get("coordinates", {})
            x, y = coords.get("x", 0), coords.get("y", 0)
            
            result = box.action.scroll(
                x=x,
                y=y,
                direction=direction,
                distance=action.get("distance", 300),
            )
            return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
        
        elif action_type == "input":
            text = action.get("text", "")
            target_dict = action.get("target", {})
            
            if target_dict:
                target_desc = target_to_desc(target_dict)
                result = await coord_generator.generate_coordinates(
                    screenshot_uri=screenshot_uri,
                    action_type="click",
                    target=target_desc,
                )
                coords = (result.get("response", {}) or result).get("coordinates", {})
                box.action.click(x=coords.get("x", 0), y=coords.get("y", 0))
            
            result = box.action.type(text=text)
            return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
        
        elif action_type == "key_press":
            keys = action.get("keys", [])
            result = box.action.press_key(keys=keys, combination=len(keys) > 1)
            return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
        
        elif action_type == "button_press":
            button = action.get("button", "home")
            result = box.action.press_button(buttons=[button])
            return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
        
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    async def terminate_box(self):
        """Terminate the box."""
        if self.box_id:
            try:
                box = self._get_box()
                box.terminate()
                logger.debug(f"Box terminated: {self.box_id}")
            except Exception as e:
                error_str = str(e).lower()
                if "404" in error_str or "not found" in error_str:
                    logger.debug(f"Box {self.box_id} already terminated")
                else:
                    logger.debug(f"Failed to terminate box: {e}")
            finally:
                self.box_id = None
                self._box = None


# ============================================================================
# Standalone Agent
# ============================================================================

class StandaloneGBoxCUAAgent:
    """Standalone GBox CUA Agent that can run independently."""
    
    def __init__(
        self,
        gbox_api_key: str,
        vlm_provider: str = "openai",
        vlm_api_base: Optional[str] = None,
        vlm_api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        max_turns: int = 20,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Initialize standalone agent."""
        self.gbox_client = GBoxClient(api_key=gbox_api_key)
        self.vlm = VLMInference(
            model_name=model_name,
            provider=vlm_provider,
            api_base=vlm_api_base,
            api_key=vlm_api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self.tools = get_tools_schema()
        self.max_turns = max_turns
        self.conversation: List[Dict[str, Any]] = []
    
    def _extract_thinking_and_content(self, raw_content: str) -> Tuple[str, str]:
        """Extract thinking and content from model output.
        
        Handles various formats:
        - <think>...</think> tags
        - <thinking>...</thinking> tags
        - Plain content (no thinking)
        """
        thinking = ""
        content = raw_content
        
        # Try <think>...</think>
        import re
        think_match = re.search(r'<think>(.*?)</think>', raw_content, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        
        # Try <thinking>...</thinking>
        if not thinking:
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', raw_content, re.DOTALL)
            if thinking_match:
                thinking = thinking_match.group(1).strip()
                content = re.sub(r'<thinking>.*?</thinking>', '', raw_content, flags=re.DOTALL).strip()
        
        return thinking, content
    
    async def run_task(
        self,
        task_description: str,
        box_type: str = "android",
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run a task with detailed logging."""
        
        # Create rollout logger
        rollout_log = RolloutLogger(
            task_id=f"task_{int(time.time())}",
            task_description=task_description,
        )
        rollout_log.set_model_info(self.vlm.model_name, self.vlm.provider)
        
        action_history = []
        task_completed = False
        task_success = False
        result_message = ""
        
        try:
            # Create box
            box_create_start = time.time()
            box_info = await self.gbox_client.create_box(box_type=box_type)
            box_create_time = time.time() - box_create_start
            box_id = box_info.get("id") or self.gbox_client.box_id
            
            rollout_log.set_box_info(box_id, box_type, box_create_time)
            
            if verbose:
                print(f"📦 Box created: {box_id} ({box_create_time:.2f}s)")
            
            # Initialize conversation
            system_prompt = create_system_prompt(task_description, max_turns=self.max_turns)
            self.conversation = [{"role": "system", "content": system_prompt}]
            
            for turn in range(self.max_turns):
                # Start turn log
                turn_log = rollout_log.start_turn(turn + 1, self.max_turns)
                
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Turn {turn + 1}/{self.max_turns}")
                    print(f"{'='*60}")
                
                # Take screenshot
                screenshot_start = time.time()
                await asyncio.sleep(0.3)
                screenshot_bytes, screenshot_uri = await self.gbox_client.take_screenshot()
                turn_log.screenshot_time = time.time() - screenshot_start
                turn_log.screenshot_size_kb = len(screenshot_bytes) / 1024
                
                if verbose:
                    print(f"📷 Screenshot: {turn_log.screenshot_size_kb:.1f} KB")
                
                # Create user message
                user_message = create_user_message_with_screenshot(
                    turn=turn + 1,
                    screenshot_description="[Screenshot attached - analyze and determine next action]",
                )
                self.conversation.append({"role": "user", "content": user_message})
                
                # Call VLM
                vlm_start = time.time()
                try:
                    vlm_response = await self.vlm.generate(
                        messages=self.conversation,
                        tools=self.tools,
                        image_data=screenshot_bytes,
                    )
                except Exception as e:
                    turn_log.vlm_time = time.time() - vlm_start
                    rollout_log.add_error(f"VLM call failed: {e}")
                    rollout_log.end_turn()
                    raise
                
                turn_log.vlm_time = time.time() - vlm_start
                turn_log.raw_response = vlm_response
                
                # Parse response
                choice = vlm_response.get("choices", [{}])[0]
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls", [])
                raw_content = message.get("content", "") or ""
                
                # Extract usage
                usage = vlm_response.get("usage", {})
                turn_log.prompt_tokens = usage.get("prompt_tokens", 0)
                turn_log.completion_tokens = usage.get("completion_tokens", 0)
                turn_log.total_tokens = usage.get("total_tokens", 0)
                
                # Extract thinking and content
                thinking, content = self._extract_thinking_and_content(raw_content)
                turn_log.thinking = thinking
                turn_log.content = content
                turn_log.tool_calls = tool_calls
                
                if verbose:
                    print(f"🤖 VLM: {turn_log.vlm_time:.2f}s, tokens: {turn_log.total_tokens}")
                    if thinking:
                        print(f"💭 Thinking: {thinking[:200]}...")
                    if content:
                        print(f"💬 Content: {content[:200]}...")
                    if tool_calls:
                        print(f"🔧 Tool calls: {len(tool_calls)}")
                
                # Add assistant message to conversation
                self.conversation.append({
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls,
                })
                
                # Execute tool calls
                for tool_call in tool_calls:
                    tool_name = tool_call.get("function", {}).get("name")
                    try:
                        arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    if tool_name == "report_task_complete":
                        task_completed = True
                        task_success = arguments.get("success", False)
                        result_message = arguments.get("result_message", "")
                        
                        turn_log.task_completed = True
                        turn_log.task_success = task_success
                        turn_log.task_message = result_message
                        
                        if verbose:
                            status = "✅ SUCCESS" if task_success else "❌ FAILED"
                            print(f"🏁 Task Complete: {status}")
                            print(f"   Message: {result_message}")
                        break
                    
                    elif tool_name == "sleep":
                        duration = arguments.get("duration", 1.0)
                        turn_log.action_type = "sleep"
                        turn_log.action_params = {"duration": duration}
                        turn_log.action_success = True
                        
                        if verbose:
                            print(f"😴 Sleep: {duration}s")
                        await asyncio.sleep(duration)
                    
                    elif tool_name == "perform_action":
                        action = tool_call_to_action_dict(tool_name, arguments)
                        action_type = action.get("action_type")
                        
                        turn_log.action_type = action_type
                        turn_log.action_params = action
                        
                        if verbose:
                            print(f"🎯 Action: {action_type}")
                            print(f"   Params: {json.dumps(action, indent=2, ensure_ascii=False)[:300]}")
                        
                        action_start = time.time()
                        try:
                            result = await self.gbox_client.execute_action(action, screenshot_uri)
                            turn_log.action_time = time.time() - action_start
                            turn_log.action_success = True
                            turn_log.action_result = result
                            
                            if verbose:
                                print(f"   ✅ Success ({turn_log.action_time:.2f}s)")
                            
                            action_history.append({
                                "turn": turn + 1,
                                "action": action,
                                "success": True,
                                "time": turn_log.action_time,
                            })
                        except Exception as e:
                            turn_log.action_time = time.time() - action_start
                            turn_log.action_success = False
                            turn_log.action_error = str(e)
                            
                            if verbose:
                                print(f"   ❌ Failed: {e}")
                            
                            rollout_log.add_error(f"Action {action_type} failed: {e}")
                            action_history.append({
                                "turn": turn + 1,
                                "action": action,
                                "success": False,
                                "error": str(e),
                            })
                        
                        await asyncio.sleep(0.5)
                
                rollout_log.end_turn()
                
                if task_completed:
                    break
            
            if not task_completed:
                result_message = f"Task not completed within {self.max_turns} turns"
        
        except Exception as e:
            rollout_log.add_error(f"Task failed: {e}")
            result_message = f"Task failed with error: {str(e)}"
            raise
        finally:
            # Terminate box
            try:
                await self.gbox_client.terminate_box()
            except Exception as cleanup_error:
                rollout_log.add_error(f"Cleanup error: {cleanup_error}")
            
            # Set final result and print log
            rollout_log.set_final_result(
                success=task_success,
                message=result_message,
                total_turns=len(action_history),
                max_turns=self.max_turns,
            )
            
            # Always print the detailed log
            rollout_log.print_log()
        
        return {
            "task_completed": task_completed,
            "task_success": task_success,
            "result_message": result_message,
            "num_turns": len(action_history),
            "max_turns": self.max_turns,
        }, action_history
    
    async def close(self):
        """Close the agent."""
        await self.vlm.close()
        await self.gbox_client.terminate_box()
