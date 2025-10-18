import json
import re
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from PIL import Image

from .vlm.base import BaseVLM
from .actions import Action, ActionType, ActionExecutor
from .grounding import GroundingParser, GroundingResult
from .prompting import (
    PromptingStrategy,
    SetOfMarkPrompting,
    AccessibilityTreePrompting,
    DirectPrompting
)


class MobileGUIAgent:
    def __init__(
        self,
        vlm: BaseVLM,
        prompting_strategy: Union[str, PromptingStrategy] = "direct",
        device_controller=None,
        verbose: bool = True
    ):
        self.vlm = vlm
        self.device_controller = device_controller
        self.verbose = verbose
        self.action_executor = ActionExecutor(device_controller)
        self.action_history: List[Dict[str, Any]] = []
        
        if isinstance(prompting_strategy, str):
            if prompting_strategy == "set_of_mark" or prompting_strategy == "som":
                self.prompting_strategy = SetOfMarkPrompting()
            elif prompting_strategy == "accessibility_tree" or prompting_strategy == "a11y":
                self.prompting_strategy = AccessibilityTreePrompting()
            elif prompting_strategy == "direct":
                self.prompting_strategy = DirectPrompting()
            else:
                raise ValueError(f"Unknown prompting strategy: {prompting_strategy}")
        else:
            self.prompting_strategy = prompting_strategy
    
    def execute_task(
        self,
        screenshot: Union[str, Path, Image.Image],
        task_instruction: str,
        step_instruction: Optional[str] = None,
        accessibility_tree: Optional[Dict[str, Any]] = None,
        max_steps: int = 10
    ) -> Dict[str, Any]:
        if isinstance(screenshot, (str, Path)):
            screenshot = Image.open(screenshot)
        
        self.action_history = []
        
        for step in range(max_steps):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Step {step + 1}/{max_steps}")
                print(f"{'='*60}")
            
            result = self.execute_step(
                screenshot=screenshot,
                task_instruction=task_instruction,
                step_instruction=step_instruction,
                accessibility_tree=accessibility_tree
            )
            
            if result['action'].action_type == ActionType.STATUS:
                if self.verbose:
                    print(f"\nTask completed with status: {result['action'].goal_status}")
                return result
            
            self.action_history.append(result['action'].to_dict())
            
            if self.device_controller:
                screenshot = self.device_controller.get_screenshot()
        
        if self.verbose:
            print(f"\nReached maximum steps ({max_steps})")
        
        return {
            'action': Action(action_type=ActionType.STATUS, goal_status="incomplete"),
            'reasoning': "Maximum steps reached",
            'success': False
        }
    
    def execute_step(
        self,
        screenshot: Union[str, Path, Image.Image],
        task_instruction: str,
        step_instruction: Optional[str] = None,
        accessibility_tree: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if isinstance(screenshot, (str, Path)):
            screenshot = Image.open(screenshot)
        
        prompt_data = self.prompting_strategy.prepare_prompt(
            task_instruction=task_instruction,
            screenshot=screenshot,
            previous_actions=self.action_history,
            step_instruction=step_instruction,
            accessibility_tree=accessibility_tree
        )
        
        if self.verbose:
            print(f"\nTask: {task_instruction}")
            if step_instruction:
                print(f"Step: {step_instruction}")
            if self.action_history:
                print(f"Previous actions: {len(self.action_history)}")
        
        vlm_response = self.vlm.query(
            prompt=prompt_data['prompt'],
            image=prompt_data['image']
        )
        
        if self.verbose:
            print(f"\nVLM Response ({self.vlm.get_model_name()}):")
            if vlm_response.reasoning and self.vlm.supports_reasoning():
                print(f"Reasoning: {vlm_response.reasoning[:500]}...")
            print(f"Content: {vlm_response.content[:500]}...")
            print(f"Tokens: {vlm_response.input_tokens} in, {vlm_response.output_tokens} out")
        
        action = self._parse_action(vlm_response.content, screenshot.size)
        
        if self.verbose:
            print(f"\nParsed Action: {action.to_dict()}")
        
        success = self.action_executor.execute(action)
        
        return {
            'action': action,
            'reasoning': vlm_response.reasoning,
            'vlm_response': vlm_response,
            'success': success,
            'screenshot_size': screenshot.size
        }
    
    def _parse_action(self, response_text: str, screenshot_size: tuple) -> Action:
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                action_dict = json.loads(json_match.group())
                
                if 'x' in action_dict and 'y' in action_dict:
                    x, y = action_dict['x'], action_dict['y']
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            action_dict['x'] = int(x * screenshot_size[0])
                            action_dict['y'] = int(y * screenshot_size[1])
                        else:
                            action_dict['x'] = int(x)
                            action_dict['y'] = int(y)
                
                return Action.from_dict(action_dict)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if self.verbose:
                print(f"Warning: Failed to parse JSON action: {e}")
        
        coords = GroundingParser.parse_coordinates(response_text)
        if coords:
            x, y = coords
            if 0 <= x <= 1 and 0 <= y <= 1:
                x = int(x * screenshot_size[0])
                y = int(y * screenshot_size[1])
            else:
                x, y = int(x), int(y)
            
            return Action(action_type=ActionType.CLICK, x=x, y=y)
        
        response_lower = response_text.lower()
        
        if "scroll" in response_lower:
            direction = "down" if "down" in response_lower else "up"
            return Action(action_type=ActionType.SCROLL, direction=direction)
        
        if "swipe" in response_lower:
            direction = "down" if "down" in response_lower else "up"
            return Action(action_type=ActionType.SWIPE, direction=direction)
        
        if "type" in response_lower or "input" in response_lower:
            text_match = re.search(r'["\']([^"\']+)["\']', response_text)
            if text_match:
                return Action(action_type=ActionType.TYPE, text=text_match.group(1))
        
        if "successful" in response_lower or "complete" in response_lower:
            return Action(action_type=ActionType.STATUS, goal_status="successful")
        
        if "fail" in response_lower:
            return Action(action_type=ActionType.STATUS, goal_status="failed")
        
        return Action(
            action_type=ActionType.CLICK,
            x=screenshot_size[0] // 2,
            y=screenshot_size[1] // 2
        )
    
    def reset(self):
        self.action_history = []
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        return self.action_history.copy()
