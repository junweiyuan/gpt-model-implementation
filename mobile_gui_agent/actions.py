from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel


class ActionType(str, Enum):
    CLICK = "click"
    SCROLL = "scroll"
    TYPE = "type"
    SWIPE = "swipe"
    OPEN_APP = "open_app"
    NAVIGATE_HOME = "navigate_home"
    STATUS = "status"


class Action(BaseModel):
    action_type: ActionType
    x: Optional[int] = None
    y: Optional[int] = None
    direction: Optional[str] = None
    text: Optional[str] = None
    app_name: Optional[str] = None
    goal_status: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"action_type": self.action_type.value}
        if self.x is not None:
            result["x"] = self.x
        if self.y is not None:
            result["y"] = self.y
        if self.direction is not None:
            result["direction"] = self.direction
        if self.text is not None:
            result["text"] = self.text
        if self.app_name is not None:
            result["app_name"] = self.app_name
        if self.goal_status is not None:
            result["goal_status"] = self.goal_status
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        return cls(**data)


class ActionExecutor:
    def __init__(self, device_controller=None):
        self.device_controller = device_controller
    
    def execute(self, action: Action) -> bool:
        if self.device_controller is None:
            print(f"[Simulated] Executing action: {action.to_dict()}")
            return True
        
        if action.action_type == ActionType.CLICK:
            return self._execute_click(action.x, action.y)
        elif action.action_type == ActionType.SCROLL:
            return self._execute_scroll(action.direction)
        elif action.action_type == ActionType.TYPE:
            return self._execute_type(action.text)
        elif action.action_type == ActionType.SWIPE:
            return self._execute_swipe(action.direction)
        elif action.action_type == ActionType.OPEN_APP:
            return self._execute_open_app(action.app_name)
        elif action.action_type == ActionType.NAVIGATE_HOME:
            return self._execute_navigate_home()
        elif action.action_type == ActionType.STATUS:
            return self._execute_status(action.goal_status)
        
        return False
    
    def _execute_click(self, x: int, y: int) -> bool:
        if self.device_controller:
            self.device_controller.click(x, y)
        return True
    
    def _execute_scroll(self, direction: str) -> bool:
        if self.device_controller:
            self.device_controller.scroll(direction)
        return True
    
    def _execute_type(self, text: str) -> bool:
        if self.device_controller:
            self.device_controller.type_text(text)
        return True
    
    def _execute_swipe(self, direction: str) -> bool:
        if self.device_controller:
            self.device_controller.swipe(direction)
        return True
    
    def _execute_open_app(self, app_name: str) -> bool:
        if self.device_controller:
            self.device_controller.open_app(app_name)
        return True
    
    def _execute_navigate_home(self) -> bool:
        if self.device_controller:
            self.device_controller.navigate_home()
        return True
    
    def _execute_status(self, goal_status: str) -> bool:
        print(f"Task status: {goal_status}")
        return goal_status == "successful"
