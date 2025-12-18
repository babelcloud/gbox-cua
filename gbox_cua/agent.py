"""Standalone GBox CUA Agent implementation.

This agent can run independently without requiring rl-cua or OSWorld.
It uses GBox SDK for box management and action execution, and supports
vllm, openai, and openrouter VLM providers.
"""

import asyncio
import base64
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

try:
    from gbox_sdk import GboxSDK
except ImportError:
    GboxSDK = None

import httpx

from gbox_cua.tools import get_tools_schema, tool_call_to_action_dict
from gbox_cua.prompts import create_system_prompt, create_user_message_with_screenshot
from gbox_cua.gbox_coordinate import GBoxCoordinateGenerator

logger = logging.getLogger(__name__)


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
        """Initialize VLM inference.
        
        Args:
            model_name: Model name
            provider: Provider to use ("vllm", "openai", or "openrouter")
            api_base: API base URL
            api_key: API key
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
        """
        self.model_name = model_name
        self.provider = provider.lower()
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        self._client = None
        
        # Validate provider
        if self.provider not in ["vllm", "openai", "openrouter"]:
            raise ValueError(f"Invalid provider: {provider}. Must be 'vllm', 'openai' or 'openrouter'")
        
        # Setup API client based on provider
        if self.provider == "vllm":
            if not api_base:
                raise ValueError("api_base is required when using vLLM provider")
            self.api_base = api_base
            self._client = httpx.AsyncClient(
                base_url=api_base,
                headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
                timeout=300.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            logger.debug(f"Initialized vLLM client: {api_base}")
        elif self.provider == "openrouter":
            if not api_base:
                api_base = "https://openrouter.ai/api/v1"
            if not api_key:
                raise ValueError("api_key is required when using OpenRouter provider")
            self.api_base = api_base
            self._client = httpx.AsyncClient(
                base_url=api_base,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/babelcloud/gbox-cua",
                    "X-Title": "GBox CUA Agent",
                },
                timeout=300.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            logger.debug(f"Initialized OpenRouter client: {api_base}")
        elif self.provider == "openai":
            if not api_base:
                api_base = "https://api.openai.com/v1"
            if not api_key:
                raise ValueError("api_key is required when using OpenAI provider")
            self.api_base = api_base
            self._client = httpx.AsyncClient(
                base_url=api_base,
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
                timeout=300.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            logger.debug(f"Initialized OpenAI client: {api_base}")
    
    def _encode_image(self, image_data: bytes) -> str:
        """Encode image bytes to base64 data URI.
        
        Args:
            image_data: Image bytes
            
        Returns:
            Base64 data URI
        """
        base64_data = base64.b64encode(image_data).decode("utf-8")
        
        # Detect image format
        if image_data[:8] == b'\x89PNG\r\n\x1a\n':
            mime_type = "image/png"
        elif image_data[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        else:
            mime_type = "image/png"  # Default
        
        return f"data:{mime_type};base64,{base64_data}"
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        image_data: Optional[bytes] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate a response from the VLM.
        
        Args:
            messages: Conversation messages
            tools: Available tools schema
            image_data: Optional image bytes to include
            temperature: Override temperature
            
        Returns:
            VLM response with tool calls
        """
        # Prepare messages with image if provided (like rl-cua)
        api_messages = []
        for msg in messages:
            if msg["role"] == "user" and image_data:
                # Add image to user message (image_url first, then text - like rl-cua)
                content = msg.get("content", "")
                image_uri = self._encode_image(image_data)
                api_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_uri}
                        },
                        {
                            "type": "text",
                            "text": content
                        }
                    ]
                })
                # Only add image to first user message
                image_data = None
            else:
                api_messages.append(msg)
        
        # Map model name for OpenRouter if needed (like rl-cua)
        model_name = self.model_name
        if self.provider == "openrouter":
            # OpenRouter model name mapping (from rl-cua)
            model_mapping = {
                "unsloth/Qwen3-VL-30B-A3B-Instruct": "qwen/qwen3-vl-30b-a3b-instruct",
                "Qwen3-VL-30B-A3B-Instruct": "qwen/qwen3-vl-30b-a3b-instruct",
                "qwen/Qwen3-VL-30B-A3B-Instruct": "qwen/qwen3-vl-30b-a3b-instruct",
                "Qwen/Qwen3-VL-30B-A3B-Instruct": "qwen/qwen3-vl-30b-a3b-instruct",
                "unsloth/Qwen3-VL-32B-Instruct": "qwen/qwen3-vl-32b-instruct",
                "Qwen3-VL-32B-Instruct": "qwen/qwen3-vl-32b-instruct",
                "qwen/Qwen3-VL-32B-Instruct": "qwen/qwen3-vl-32b-instruct",
                "Qwen/Qwen3-VL-32B-Instruct": "qwen/qwen3-vl-32b-instruct",
                "unsloth/Qwen3-VL-8B-Instruct": "qwen/qwen3-vl-8b-instruct",
                "Qwen3-VL-8B-Instruct": "qwen/qwen3-vl-8b-instruct",
                "qwen/Qwen3-VL-8B-Instruct": "qwen/qwen3-vl-8b-instruct",
                "Qwen/Qwen3-VL-8B-Instruct": "qwen/qwen3-vl-8b-instruct",
                "Qwen/Qwen2.5-VL-32B-Instruct": "qwen/qwen2.5-vl-32b-instruct",
                "qwen/Qwen2.5-VL-32B-Instruct": "qwen/qwen2.5-vl-32b-instruct",
            }
            model_name = model_mapping.get(model_name, model_name)
            # If model name still contains "/", assume it's already in OpenRouter format
            if "/" in model_name and model_name not in model_mapping.values():
                # Convert to lowercase for OpenRouter format
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
            provider_name = "OpenRouter" if self.provider == "openrouter" else "vLLM"
            error_detail = ""
            try:
                error_body = e.response.json()
                error_detail = f": {error_body}"
                
                # Provide helpful error message for OpenRouter 404
                if self.provider == "openrouter" and e.response.status_code == 404:
                    error_msg = error_body.get("error", {}).get("message", "")
                    if "No endpoints found" in error_msg:
                        logger.error(
                            f"Model '{model_name}' is not available on OpenRouter. "
                            f"This may mean:\n"
                            f"  1. The model is temporarily unavailable\n"
                            f"  2. The model name is incorrect\n"
                            f"  3. You need to check available models at https://openrouter.ai/models\n"
                            f"  Try using an alternative model like 'qwen/qwen2.5-vl-32b-instruct' or 'qwen/qwen3-vl-8b-instruct'"
                        )
            except:
                error_detail = f": {e.response.text}"
            logger.error(f"{provider_name} API error: {e.response.status_code}{error_detail}")
            raise RuntimeError(
                f"{provider_name} API error: {e.response.status_code}{error_detail}"
            ) from e
        
        result = response.json()
        return result
    
    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.aclose()


class GBoxClient:
    """GBox API client for box management and action execution."""
    
    def __init__(self, api_key: str):
        """Initialize GBox client.
        
        Args:
            api_key: GBox API key
        """
        if GboxSDK is None:
            raise ImportError("gbox_sdk not installed. Install with: pip install gbox-sdk")
        
        self.api_key = api_key
        self._sdk = GboxSDK(api_key=api_key)
        self.box_id: Optional[str] = None
    
    async def create_box(self, box_type: str = "android") -> Dict[str, Any]:
        """Create a new GBox environment.
        
        Args:
            box_type: Type of box ("android" or "linux")
            
        Returns:
            Box creation response
        """
        logger.debug(f"Creating {box_type} box...")
        
        try:
            box_response = self._sdk.create(
                type=box_type,
                wait=True,
                timeout="60s",  # Must be a string with time unit (e.g., "60s", "5m")
            )
            
            # Extract box ID
            if hasattr(box_response, 'data') and hasattr(box_response.data, 'id'):
                self.box_id = box_response.data.id
            elif hasattr(box_response, 'id'):
                self.box_id = box_response.id
            elif isinstance(box_response, dict):
                self.box_id = box_response.get("id") or (box_response.get("data", {}).get("id") if isinstance(box_response.get("data"), dict) else None)
            else:
                raise ValueError(f"Unexpected box response format: {type(box_response)}")
            
            if not self.box_id:
                raise ValueError("Failed to extract box ID from response")
            
            logger.debug(f"Box created: {self.box_id}")
            return {"id": self.box_id}
        except Exception as e:
            logger.error(f"Failed to create {box_type} box: {e}")
            raise
    
    async def take_screenshot(self) -> Tuple[bytes, str]:
        """Take a screenshot of the box display.
        
        Returns:
            Tuple of (image_bytes, base64_data_uri)
        """
        if not self.box_id:
            raise ValueError("No box ID available. Create a box first.")
        
        try:
            screenshot_result = self._sdk.client.post(
                f"/boxes/{self.box_id}/actions/screenshot",
                cast_to=Dict[str, Any],
                body={"format": "png"}
            )
            
            # Extract screenshot data
            if isinstance(screenshot_result, dict):
                screenshot_data = (
                    screenshot_result.get("uri") or
                    screenshot_result.get("screenshot") or
                    screenshot_result.get("url")
                )
            else:
                screenshot_data = None
            
            if not screenshot_data:
                raise ValueError(f"Failed to extract screenshot from response: {screenshot_result}")
            
            # Convert to bytes and data URI
            if screenshot_data.startswith("data:"):
                parts = screenshot_data.split(",", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid data URI format")
                _, data = parts
                image_bytes = base64.b64decode(data)
                return image_bytes, screenshot_data
            else:
                # HTTP URL - fetch the image
                async with httpx.AsyncClient() as client:
                    img_response = await client.get(screenshot_data)
                    img_response.raise_for_status()
                    image_bytes = img_response.content
                    base64_data = base64.b64encode(image_bytes).decode()
                    data_uri = f"data:image/png;base64,{base64_data}"
                    return image_bytes, data_uri
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            raise
    
    async def execute_action(self, action: Dict[str, Any], screenshot_uri: str) -> Dict[str, Any]:
        """Execute an action using GBox API.
        
        Args:
            action: Action dictionary
            screenshot_uri: Screenshot URI for coordinate generation
            
        Returns:
            Action execution result
        """
        if not self.box_id:
            raise ValueError("No box ID available")
        
        action_type = action.get("action_type")
        
        # Generate coordinates using GBox model
        coord_generator = GBoxCoordinateGenerator(
            api_key=self.api_key,
            model="gbox-handy-1"
        )
        
        try:
            logger.debug(f"[GBox Client] Executing action: {action_type}, box_id={self.box_id}")
            
            if action_type == "click":
                target = action.get("target", {})
                # Convert target dict to description string (like rl-cua)
                if isinstance(target, dict):
                    parts = [target.get("element", "")]
                    if target.get("label"):
                        parts.append(f'labeled "{target["label"]}"')
                    if target.get("color"):
                        parts.append(f"{target['color']} colored")
                    if target.get("size"):
                        parts.append(f"{target['size']} sized")
                    if target.get("location"):
                        parts.append(f"located at {target['location']}")
                    if target.get("shape"):
                        parts.append(f"with {target['shape']} shape")
                    target_desc = " ".join(parts) if parts[0] else "center of the screen"
                else:
                    target_desc = str(target) if target else "center of the screen"
                
                logger.debug(f"[GBox Client] Generating coordinates for click: target={target_desc}")
                result = await coord_generator.generate_coordinates(
                    screenshot_uri=screenshot_uri,
                    action_type="click",
                    target=target_desc,
                )
                # Extract coordinates from response (like rl-cua)
                coords = result.get("response", {}).get("coordinates", {}) or result.get("coordinates", {})
                x, y = coords.get("x", 0), coords.get("y", 0)
                logger.debug(f"[GBox Client] Coordinates generated: x={x}, y={y}")
                
                button = action.get("option", "left")
                double_click = button == "double"
                
                payload = {"x": x, "y": y, "button": button}
                if double_click:
                    payload["doubleClick"] = True
                
                logger.debug(f"[GBox Client] Calling GBox API: POST /boxes/{self.box_id}/actions/click")
                result = self._sdk.client.post(
                    f"/boxes/{self.box_id}/actions/click",
                    cast_to=Dict[str, Any],
                    body=payload
                )
                logger.debug(f"[GBox Client] Click action completed")
                return result if isinstance(result, dict) else {}
            
            elif action_type == "swipe":
                # Convert target dicts to description strings
                start_target_dict = action.get("start_target", {})
                end_target_dict = action.get("end_target", {})
                
                def target_to_desc(t):
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
                        return " ".join(parts) if parts[0] else "screen center"
                    return str(t) if t else "screen center"
                
                start_desc = target_to_desc(start_target_dict)
                end_desc = target_to_desc(end_target_dict)
                
                logger.debug(f"[GBox Client] Generating coordinates for swipe: {start_desc} -> {end_desc}")
                # Use drag type to get both coordinates (like rl-cua)
                drag_result = await coord_generator.generate_coordinates(
                    screenshot_uri=screenshot_uri,
                    action_type="drag",
                    target=start_desc,
                    end_target=end_desc,
                )
                
                # Parse drag response (like rl-cua)
                response_data = drag_result.get("response", {}) or drag_result
                coordinates = response_data.get("coordinates", {})
                
                if "start" in coordinates and "end" in coordinates:
                    start_coords = coordinates.get("start", {})
                    end_coords = coordinates.get("end", {})
                else:
                    # Fallback: separate calls
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
                
                result = self._sdk.client.post(
                    f"/boxes/{self.box_id}/actions/swipe",
                    cast_to=Dict[str, Any],
                    body={
                        "start": {"x": start_coords.get("x", 0), "y": start_coords.get("y", 0)},
                        "end": {"x": end_coords.get("x", 0), "y": end_coords.get("y", 0)},
                        "duration": "300ms",
                    }
                )
                return result if isinstance(result, dict) else {}
            
            elif action_type == "scroll":
                target_dict = action.get("target", {})
                direction = action.get("direction", "down")
                
                # Convert target to description
                if isinstance(target_dict, dict):
                    parts = [target_dict.get("element", "")]
                    if target_dict.get("label"):
                        parts.append(f'labeled "{target_dict["label"]}"')
                    if target_dict.get("color"):
                        parts.append(f"{target_dict['color']} colored")
                    if target_dict.get("size"):
                        parts.append(f"{target_dict['size']} sized")
                    if target_dict.get("location"):
                        parts.append(f"located at {target_dict['location']}")
                    if target_dict.get("shape"):
                        parts.append(f"with {target_dict['shape']} shape")
                    target_desc = " ".join(parts) if parts[0] else "center of the screen"
                else:
                    target_desc = str(target_dict) if target_dict else "center of the screen"
                
                logger.debug(f"[GBox Client] Generating coordinates for scroll: target={target_desc}, direction={direction}")
                result = await coord_generator.generate_coordinates(
                    screenshot_uri=screenshot_uri,
                    action_type="scroll",
                    target=target_desc,
                    direction=direction,
                )
                coords = (result.get("response", {}) or result).get("coordinates", {})
                x, y = coords.get("x", 0), coords.get("y", 0)
                logger.debug(f"[GBox Client] Scroll coordinates: x={x}, y={y}")
                
                result = self._sdk.client.post(
                    f"/boxes/{self.box_id}/actions/scroll",
                    cast_to=Dict[str, Any],
                    body={
                        "x": x,
                        "y": y,
                        "direction": direction,
                        "distance": action.get("distance", 300),
                    }
                )
                return result if isinstance(result, dict) else {}
            
            elif action_type == "input":
                text = action.get("text", "")
                target_dict = action.get("target", {})
                
                params = {"text": text}
                if target_dict:
                    # Convert target to description
                    if isinstance(target_dict, dict):
                        parts = [target_dict.get("element", "")]
                        if target_dict.get("label"):
                            parts.append(f'labeled "{target_dict["label"]}"')
                        if target_dict.get("color"):
                            parts.append(f"{target_dict['color']} colored")
                        if target_dict.get("size"):
                            parts.append(f"{target_dict['size']} sized")
                        if target_dict.get("location"):
                            parts.append(f"located at {target_dict['location']}")
                        if target_dict.get("shape"):
                            parts.append(f"with {target_dict['shape']} shape")
                        target_desc = " ".join(parts) if parts[0] else None
                    else:
                        target_desc = str(target_dict) if target_dict else None
                    
                    if target_desc:
                        logger.debug(f"[GBox Client] Generating coordinates for input: target={target_desc}")
                        result = await coord_generator.generate_coordinates(
                            screenshot_uri=screenshot_uri,
                            action_type="click",
                            target=target_desc,
                        )
                        coords = (result.get("response", {}) or result).get("coordinates", {})
                        params["x"] = coords.get("x", 0)
                        params["y"] = coords.get("y", 0)
                        logger.debug(f"[GBox Client] Input coordinates: x={params['x']}, y={params['y']}")
                
                result = self._sdk.client.post(
                    f"/boxes/{self.box_id}/actions/type",
                    cast_to=Dict[str, Any],
                    body=params
                )
                return result if isinstance(result, dict) else {}
            
            elif action_type == "key_press":
                keys = action.get("keys", [])
                result = self._sdk.client.post(
                    f"/boxes/{self.box_id}/actions/key",
                    cast_to=Dict[str, Any],
                    body={"keys": keys}
                )
                return result if isinstance(result, dict) else {}
            
            elif action_type == "button_press":
                button = action.get("button", "home")
                result = self._sdk.client.post(
                    f"/boxes/{self.box_id}/actions/button",
                    cast_to=Dict[str, Any],
                    body={"button": button}
                )
                return result if isinstance(result, dict) else {}
            
            else:
                raise ValueError(f"Unknown action type: {action_type}")
        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            raise
    
    async def terminate_box(self):
        """Terminate the box."""
        if self.box_id:
            box_id = self.box_id
            try:
                # Try SDK terminate method first if available
                if hasattr(self._sdk, 'terminate'):
                    self._sdk.terminate(box_id)
                    logger.debug(f"Box terminated via SDK: {box_id}")
                else:
                    # Fall back to DELETE endpoint
                    self._sdk.client.delete(f"/boxes/{box_id}", cast_to=Dict[str, Any])
                    logger.debug(f"Box terminated via DELETE: {box_id}")
            except Exception as e:
                # Check if it's a 404 error (box already terminated/deleted)
                error_str = str(e)
                error_str_lower = error_str.lower()
                status_code = None
                
                # Try to extract status code from exception attributes
                if hasattr(e, 'status_code'):
                    status_code = e.status_code
                elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    status_code = e.response.status_code
                elif hasattr(e, 'code'):
                    status_code = e.code
                elif hasattr(e, 'statusCode'):
                    status_code = e.statusCode
                
                # Try to extract from error message/dict if it's formatted as "Error code: 404"
                if status_code is None:
                    import re
                    match = re.search(r'Error code:\s*(\d+)', error_str, re.IGNORECASE)
                    if match:
                        try:
                            status_code = int(match.group(1))
                        except ValueError:
                            pass
                
                # Check error message for 404 indicators (more comprehensive)
                is_404 = (
                    status_code == 404 or
                    "404" in error_str or
                    "not found" in error_str_lower or
                    "cannot delete" in error_str_lower or
                    "'statuscode': 404" in error_str_lower or
                    '"statuscode": 404' in error_str_lower
                )
                
                if is_404:
                    logger.debug(f"Box {box_id} already terminated or not found (404) - this is expected if box was auto-terminated")
                else:
                    # Only log as warning if it's not a 404 - don't let this mask other errors
                    logger.debug(f"Failed to terminate box {box_id}: {e} (non-critical)")
            finally:
                self.box_id = None


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
        """Initialize standalone agent.
        
        Args:
            gbox_api_key: GBox API key
            vlm_provider: VLM provider ("vllm", "openai", or "openrouter")
            vlm_api_base: VLM API base URL
            vlm_api_key: VLM API key
            model_name: Model name
            max_turns: Maximum number of turns
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
        """
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
    
    async def run_task(
        self,
        task_description: str,
        box_type: str = "android",
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run a task.
        
        Args:
            task_description: Task description
            box_type: Box type ("android" or "linux")
            verbose: Enable verbose output
            
        Returns:
            Tuple of (result_dict, action_history)
        """
        # Log task information
        logger.info("="*60)
        logger.info(f"Starting Task: {task_description}")
        logger.info(f"Box Type: {box_type}")
        logger.info(f"Max Turns: {self.max_turns}")
        logger.info(f"VLM Provider: {self.vlm.provider}")
        logger.info(f"Model: {self.vlm.model_name}")
        logger.info("="*60)
        
        # Create box
        box_create_start = time.time()
        box_info = await self.gbox_client.create_box(box_type=box_type)
        box_create_time = time.time() - box_create_start
        box_id = (box_info or {}).get("id") or getattr(self.gbox_client, "box_id", None)
        logger.info(f"Box created: id={box_id}, took {box_create_time:.3f}s")
        if verbose:
            print(f"Box created: id={box_id}, took {box_create_time:.3f}s")
        
        # Initialize conversation
        system_prompt = create_system_prompt(task_description, max_turns=self.max_turns)
        self.conversation = [
            {"role": "system", "content": system_prompt},
        ]
        
        action_history = []
        task_completed = False
        task_success = False
        result_message = ""
        
        try:
            for turn in range(self.max_turns):
                turn_start = time.time()
                logger.info(f"\n[Turn {turn + 1}/{self.max_turns}] Starting...")
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Turn {turn + 1}/{self.max_turns}")
                    print(f"{'='*60}")
                
                # Take screenshot
                screenshot_start = time.time()
                await asyncio.sleep(0.3)  # Screenshot delay
                screenshot_bytes, screenshot_uri = await self.gbox_client.take_screenshot()
                screenshot_time = time.time() - screenshot_start
                screenshot_size_kb = len(screenshot_bytes) / 1024
                
                logger.info(f"[Turn {turn + 1}] Screenshot: {screenshot_size_kb:.2f} KB, took {screenshot_time:.3f}s")
                
                # Create user message (like rl-cua: just text, image_data passed separately)
                user_message = create_user_message_with_screenshot(
                    turn=turn + 1,
                    screenshot_description="[Screenshot attached - analyze and determine next action]",
                )
                
                # Add user message to conversation (text only, like rl-cua)
                self.conversation.append({
                    "role": "user",
                    "content": user_message,
                })
                
                logger.debug(f"[Turn {turn + 1}] User message: {user_message[:100]}...")
                
                # Call VLM (pass image_data separately, like rl-cua)
                vlm_start = time.time()
                try:
                    vlm_response = await self.vlm.generate(
                        messages=self.conversation,
                        tools=self.tools,
                        image_data=screenshot_bytes,  # Pass image bytes, not URI
                    )
                except Exception as e:
                    vlm_time = time.time() - vlm_start
                    logger.error(f"[Turn {turn + 1}] VLM call failed after {vlm_time:.3f}s: {e}", exc_info=True)
                    raise
                vlm_time = time.time() - vlm_start
                
                # Parse response (vlm_response is now a dict from response.json())
                choice = vlm_response.get("choices", [{}])[0]
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls", [])
                raw_content = message.get("content", "") or ""
                
                # Extract usage info
                usage_info = vlm_response.get("usage", {})
                prompt_tokens = usage_info.get("prompt_tokens", 0)
                completion_tokens = usage_info.get("completion_tokens", 0)
                total_tokens = usage_info.get("total_tokens", 0)
                
                logger.info(f"[Turn {turn + 1}] VLM Response: {vlm_time:.3f}s, tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
                logger.debug(f"[Turn {turn + 1}] VLM raw content: {raw_content[:200]}...")
                logger.debug(f"[Turn {turn + 1}] Tool calls: {len(tool_calls)}")
                
                # Add assistant message
                self.conversation.append({
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls,
                })
                
                # Execute tool calls
                action_time = 0.0
                for tool_call in tool_calls:
                    tool_name = tool_call.get("function", {}).get("name")
                    arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    
                    if tool_name == "report_task_complete":
                        task_completed = True
                        task_success = arguments.get("success", False)
                        result_message = arguments.get("result_message", "")
                        logger.info(f"[Turn {turn + 1}] Task Complete: success={task_success}, message={result_message}")
                        break
                    
                    elif tool_name == "sleep":
                        duration = arguments.get("duration", 1.0)
                        logger.info(f"[Turn {turn + 1}] Sleep: {duration}s")
                        await asyncio.sleep(duration)
                    
                    elif tool_name == "perform_action":
                        action = tool_call_to_action_dict(tool_name, arguments)
                        action_type = action.get("action_type")
                        
                        logger.info(f"[Turn {turn + 1}] Action: {action_type}")
                        logger.debug(f"[Turn {turn + 1}] Action details: {json.dumps(action, indent=2)}")
                        
                        action_start = time.time()
                        try:
                            result = await self.gbox_client.execute_action(action, screenshot_uri)
                            action_time = time.time() - action_start
                            
                            logger.info(f"[Turn {turn + 1}] Action Success: {action_time:.3f}s")
                            logger.debug(f"[Turn {turn + 1}] Action result: {result}")
                            
                            action_history.append({
                                "turn": turn + 1,
                                "action": action,
                                "success": True,
                                "time": action_time,
                                "result": result,
                            })
                        except Exception as e:
                            action_time = time.time() - action_start
                            logger.error(f"[Turn {turn + 1}] Action Failed: {action_time:.3f}s, error={e}")
                            logger.debug(f"[Turn {turn + 1}] Action error details:", exc_info=True)
                            
                            action_history.append({
                                "turn": turn + 1,
                                "action": action,
                                "success": False,
                                "error": str(e),
                                "time": action_time,
                            })
                        
                        # Wait after action
                        await asyncio.sleep(0.5)
                
                # Log turn summary
                turn_time = time.time() - turn_start
                logger.info(f"[Turn {turn + 1}] Summary: total={turn_time:.3f}s, screenshot={screenshot_time:.3f}s, vlm={vlm_time:.3f}s, action={action_time:.3f}s")
                
                if task_completed:
                    break
            
            if not task_completed:
                result_message = f"Task not completed within {self.max_turns} turns"
                logger.warning(f"Task incomplete: {result_message}")
        
        except Exception as e:
            # Log the exception before cleanup
            logger.error(f"Task execution failed: {e}", exc_info=True)
            result_message = f"Task failed with error: {str(e)}"
            raise
        finally:
            # Terminate box (errors here are non-critical and won't mask original errors)
            try:
                await self.gbox_client.terminate_box()
            except Exception as cleanup_error:
                # Log but don't raise - we don't want cleanup errors to mask the original error
                logger.debug(f"Error during box cleanup (non-critical): {cleanup_error}")
        
        result = {
            "task_completed": task_completed,
            "task_success": task_success,
            "result_message": result_message,
            "num_turns": len(action_history),
            "max_turns": self.max_turns,
        }
        
        logger.info("="*60)
        logger.info(f"Task Completed: {task_completed}")
        logger.info(f"Task Success: {task_success}")
        logger.info(f"Result Message: {result_message}")
        logger.info(f"Total Turns: {len(action_history)}/{self.max_turns}")
        logger.info("="*60)
        
        return result, action_history
    
    async def close(self):
        """Close the agent."""
        await self.vlm.close()
        await self.gbox_client.terminate_box()

