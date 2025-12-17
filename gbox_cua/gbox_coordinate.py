"""Shared GBox coordinate generation logic.

This module provides the core coordinate generation functionality
that is shared between rl-cua and OSWorld-provider implementations.
"""

import json
import logging
from typing import Optional, Dict, Any

try:
    from gbox_sdk import GboxSDK
except ImportError:
    GboxSDK = None
    logging.warning("gbox_sdk not installed. Install with: pip install gbox-sdk")

logger = logging.getLogger(__name__)


class GBoxCoordinateGenerator:
    """Shared coordinate generation using GBox model.
    
    This class provides the core coordinate generation functionality
    that can be used by both rl-cua and OSWorld-provider implementations.
    """
    
    def __init__(self, api_key: str, model: str = "gbox-handy-1"):
        """Initialize GBox coordinate generator.
        
        Args:
            api_key: GBox API key
            model: Model name for coordinate generation (default: gbox-handy-1)
        """
        if GboxSDK is None:
            raise ImportError(
                "gbox_sdk not installed. Install with: pip install gbox-sdk"
            )
        
        self.api_key = api_key
        self.model = model
        self._sdk = GboxSDK(api_key=api_key)
    
    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse SDK response to dictionary."""
        if hasattr(response, 'json'):
            return response.json()
        elif hasattr(response, 'data'):
            if hasattr(response.data, 'model_dump'):
                return response.data.model_dump()
            elif hasattr(response.data, 'dict'):
                return response.data.dict()
            else:
                return dict(response.data) if hasattr(response.data, '__dict__') else response.data
        elif hasattr(response, 'model_dump'):
            return response.model_dump()
        elif hasattr(response, 'dict'):
            return response.dict()
        else:
            return dict(response) if isinstance(response, dict) else {"response": response}
    
    async def generate_coordinates(
        self,
        screenshot_uri: str,
        action_type: str,
        target: str,
        end_target: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate coordinates using gbox-handy-1 model.
        
        Args:
            screenshot_uri: Screenshot URI (base64 data URI or URL)
            action_type: Type of action ("click", "drag", "scroll")
            target: Target element description (natural language)
            end_target: End target for drag actions
            direction: Direction for scroll actions
            
        Returns:
            Coordinate generation response with coordinates
        """
        # Build action object based on type
        # According to GBox API docs: https://docs.gbox.ai/api-reference/model/generate-coordinates-for-a-model
        if action_type == "click":
            action = {
                "type": "click",
                "target": target,
            }
        elif action_type == "drag":
            action = {
                "type": "drag",
                "target": target,
                "destination": end_target or target,
            }
        elif action_type == "scroll":
            action = {
                "type": "scroll",
                "location": target,
                "direction": direction or "down",
            }
        else:
            raise ValueError(f"Unknown action type: {action_type}")
        
        try:
            logger.info(f"[GBox Coordinate] Generating coordinates: action_type={action_type}, target={target}")
            logger.debug(f"[GBox Coordinate] Action object: {action}")
            logger.debug(f"[GBox Coordinate] Screenshot URI length: {len(screenshot_uri)} chars")
            
            # Use SDK client to call model API
            # POST /model
            logger.info(f"[GBox Coordinate] Calling GBox model API: POST /model with model={self.model}")
            result = self._sdk.client.post(
                "/model",
                cast_to=Dict[str, Any],
                body={
                    "model": self.model,
                    "screenshot": screenshot_uri,
                    "action": action,
                }
            )
            
            parsed_result = self._parse_response(result)
            logger.info(f"[GBox Coordinate] Coordinate generation successful")
            logger.debug(f"[GBox Coordinate] Response: {json.dumps(parsed_result, indent=2, default=str)}")
            
            return parsed_result
        except Exception as e:
            logger.error(f"[GBox Coordinate] Failed to generate coordinates: {e}", exc_info=True)
            raise

