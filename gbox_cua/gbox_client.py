"""GBox API Client for box management and UI actions using official SDK.

Uses the SDK's high-level wrapper API directly:
- sdk.create(type="android") -> box operator
- sdk.get(box_id) -> box operator  
- box.action.click(x=..., y=...)
- box.action.screenshot()

Reference: https://babelcloud.github.io/gbox-sdk-py/
"""

import base64
import logging
from typing import Optional, Dict, Any, List, Tuple

from gbox_sdk import GboxSDK

from gbox_cua.gbox_coordinate import GBoxCoordinateGenerator

logger = logging.getLogger(__name__)


class GBoxClient:
    """Client for interacting with GBox API using official SDK wrapper.
    
    Uses box.action.* methods for all UI operations.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gbox-handy-1",
        box_type: str = "android",
        timeout: str = "60s",
        wait: bool = True,
        expires_in: str = "15m",
        labels: Optional[Dict[str, Any]] = None,
        envs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize GBox client."""
        self.api_key = api_key
        self.model = model
        self.box_type = box_type
        self.timeout = timeout
        self.wait = wait
        self.expires_in = expires_in
        self.labels = labels or {}
        self.envs = envs or {}
        
        self.box_id: Optional[str] = None
        self._sdk = GboxSDK(api_key=api_key)
        self._box: Optional[Any] = None
        
        self._coord_generator = GBoxCoordinateGenerator(
            api_key=api_key,
            model=model
        )
    
    async def create_box(self, box_type: Optional[str] = None) -> Dict[str, Any]:
        """Create a new GBox environment."""
        box_type = box_type or self.box_type
        logger.debug(f"Creating {box_type} box...")
        
        box = self._sdk.create(
            type=box_type,
            wait=self.wait,
            timeout=self.timeout,
            config={
                "expiresIn": self.expires_in,
                **({"labels": self.labels} if self.labels else {}),
                **({"envs": self.envs} if self.envs else {}),
            }
        )
        
        self._box = box
        self.box_id = box.data.id
        logger.debug(f"Box created: {self.box_id}")
        return {"id": self.box_id}
    
    def _get_box(self, box_id: Optional[str] = None) -> Any:
        """Get box operator."""
        if box_id and box_id != self.box_id:
            return self._sdk.get(box_id)
        if self._box:
            return self._box
        if self.box_id:
            self._box = self._sdk.get(self.box_id)
            return self._box
        raise ValueError("No box available. Call create_box() first.")
    
    async def terminate_box(self, box_id: Optional[str] = None) -> Dict[str, Any]:
        """Terminate a GBox environment."""
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        logger.debug(f"Terminating box: {box_id}")
        box = self._get_box(box_id)
        box.terminate()
        
        if box_id == self.box_id:
            self.box_id = None
            self._box = None
        
        return {"id": box_id, "status": "terminated"}
    
    async def get_box(self, box_id: Optional[str] = None) -> Dict[str, Any]:
        """Get box information."""
        box_id = box_id or self.box_id
        if not box_id:
            raise ValueError("No box ID provided")
        
        box = self._sdk.get(box_id)
        return box.data.model_dump() if hasattr(box.data, 'model_dump') else dict(box.data)
    
    async def take_screenshot(
        self,
        box_id: Optional[str] = None,
        format: str = "png",
    ) -> Tuple[bytes, str]:
        """Take a screenshot of the box display."""
        box = self._get_box(box_id)
        result = box.action.screenshot(output_format="base64")
        
        screenshot_uri = result.uri
        
        if screenshot_uri.startswith("data:"):
            parts = screenshot_uri.split(",", 1)
            image_bytes = base64.b64decode(parts[1])
            return image_bytes, screenshot_uri
        else:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(screenshot_uri)
                resp.raise_for_status()
                image_bytes = resp.content
                data_uri = f"data:image/{format};base64,{base64.b64encode(image_bytes).decode()}"
                return image_bytes, data_uri
    
    async def generate_coordinates(
        self,
        screenshot_uri: str,
        action_type: str,
        target: str,
        end_target: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate coordinates using gbox-handy-1 model."""
        result = await self._coord_generator.generate_coordinates(
            screenshot_uri=screenshot_uri,
            action_type=action_type,
            target=target,
            end_target=end_target,
            direction=direction,
        )
        
        if "response" not in result:
            return {"response": result}
        return result
    
    async def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        double_click: bool = False,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a click action."""
        box = self._get_box(box_id)
        result = box.action.click(x=x, y=y, button=button, double=double_click)
        return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
    
    async def tap(
        self,
        x: int,
        y: int,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a tap action (for touch screens)."""
        box = self._get_box(box_id)
        result = box.action.tap(x=x, y=y)
        return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
    
    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: int = 300,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a swipe action."""
        box = self._get_box(box_id)
        result = box.action.swipe(
            start={"x": start_x, "y": start_y},
            end={"x": end_x, "y": end_y},
            duration=f"{duration}ms",
        )
        return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
    
    async def scroll(
        self,
        x: int,
        y: int,
        direction: str = "down",
        distance: int = 300,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a scroll action."""
        box = self._get_box(box_id)
        result = box.action.scroll(x=x, y=y, direction=direction, distance=distance)
        return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
    
    async def type_text(
        self,
        text: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Type text into the focused element."""
        box = self._get_box(box_id)
        if x is not None and y is not None:
            box.action.click(x=x, y=y)
        result = box.action.type(text=text)
        return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
    
    async def press_key(
        self,
        keys: List[str],
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Press one or more keys."""
        box = self._get_box(box_id)
        result = box.action.press_key(keys=keys, combination=len(keys) > 1)
        return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
    
    async def press_button(
        self,
        button: str,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Press a device button (Android)."""
        box = self._get_box(box_id)
        result = box.action.press_button(buttons=[button])
        return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
    
    async def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: int = 500,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a drag action."""
        box = self._get_box(box_id)
        result = box.action.drag(
            start={"x": start_x, "y": start_y},
            end={"x": end_x, "y": end_y},
            duration=f"{duration}ms",
        )
        return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
    
    async def long_press(
        self,
        x: int,
        y: int,
        duration: int = 1000,
        box_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a long press action."""
        box = self._get_box(box_id)
        result = box.action.long_press(x=x, y=y, duration=f"{duration}ms")
        return result.model_dump() if hasattr(result, 'model_dump') else dict(result)
    
    async def close(self):
        """Close and terminate the box if active."""
        if self._box:
            try:
                await self.terminate_box()
            except Exception as e:
                logger.warning(f"Failed to terminate box on close: {e}")
        self._box = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
