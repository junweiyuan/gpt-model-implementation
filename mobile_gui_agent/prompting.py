from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import io
import base64


class PromptingStrategy:
    def prepare_prompt(
        self,
        task_instruction: str,
        screenshot: Image.Image,
        previous_actions: Optional[List[Dict[str, Any]]] = None,
        step_instruction: Optional[str] = None,
        accessibility_tree: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError


class SetOfMarkPrompting(PromptingStrategy):
    def __init__(self, mark_size: int = 30):
        self.mark_size = mark_size
        self.next_mark_id = 0
    
    def prepare_prompt(
        self,
        task_instruction: str,
        screenshot: Image.Image,
        previous_actions: Optional[List[Dict[str, Any]]] = None,
        step_instruction: Optional[str] = None,
        accessibility_tree: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        marked_image, element_map = self._add_marks(screenshot, accessibility_tree)
        
        prompt = self._build_prompt(
            task_instruction,
            step_instruction,
            previous_actions,
            element_map
        )
        
        return {
            "prompt": prompt,
            "image": marked_image,
            "element_map": element_map
        }
    
    def _add_marks(
        self,
        screenshot: Image.Image,
        accessibility_tree: Optional[Dict[str, Any]] = None
    ) -> tuple[Image.Image, Dict[int, Dict[str, Any]]]:
        marked_image = screenshot.copy()
        draw = ImageDraw.Draw(marked_image)
        element_map = {}
        
        if accessibility_tree:
            elements = self._extract_interactive_elements(accessibility_tree)
        else:
            elements = self._detect_grid_points(screenshot)
        
        for idx, element in enumerate(elements):
            mark_id = idx
            x, y = element['center']
            
            draw.ellipse(
                [(x - self.mark_size // 2, y - self.mark_size // 2),
                 (x + self.mark_size // 2, y + self.mark_size // 2)],
                fill=(255, 0, 0, 128),
                outline=(255, 255, 255)
            )
            
            draw.text(
                (x, y),
                str(mark_id),
                fill=(255, 255, 255),
                anchor="mm"
            )
            
            element_map[mark_id] = {
                'center': (x, y),
                'bounds': element.get('bounds'),
                'description': element.get('description', f"Element {mark_id}")
            }
        
        return marked_image, element_map
    
    def _extract_interactive_elements(
        self,
        accessibility_tree: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        elements = []
        
        def traverse(node):
            if node.get('clickable') or node.get('focusable'):
                bounds = node.get('bounds')
                if bounds:
                    x1, y1, x2, y2 = bounds
                    elements.append({
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'bounds': bounds,
                        'description': node.get('text') or node.get('content-desc') or 'Interactive element'
                    })
            
            for child in node.get('children', []):
                traverse(child)
        
        traverse(accessibility_tree)
        return elements
    
    def _detect_grid_points(self, screenshot: Image.Image) -> List[Dict[str, Any]]:
        width, height = screenshot.size
        grid_size = 5
        elements = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = int((i + 0.5) * width / grid_size)
                y = int((j + 0.5) * height / grid_size)
                elements.append({
                    'center': (x, y),
                    'bounds': None,
                    'description': f'Grid point ({i}, {j})'
                })
        
        return elements
    
    def _build_prompt(
        self,
        task_instruction: str,
        step_instruction: Optional[str],
        previous_actions: Optional[List[Dict[str, Any]]],
        element_map: Dict[int, Dict[str, Any]]
    ) -> str:
        prompt_parts = []
        
        prompt_parts.append("You are a mobile GUI agent that controls a smartphone to complete user tasks.")
        prompt_parts.append("\nThe screenshot has been marked with numbered red circles indicating interactive elements.")
        
        prompt_parts.append(f"\nTask instruction: {task_instruction}")
        
        if step_instruction:
            prompt_parts.append(f"Step instruction: {step_instruction}")
        
        if previous_actions:
            prompt_parts.append("\nPrevious actions:")
            for action in previous_actions:
                prompt_parts.append(f"  - {action}")
        
        prompt_parts.append("\nAvailable elements:")
        for mark_id, element in element_map.items():
            prompt_parts.append(f"  [{mark_id}] {element['description']} at {element['center']}")
        
        prompt_parts.append("\nPlease analyze the current screen and determine the next action to complete the task.")
        prompt_parts.append("Think step by step about:")
        prompt_parts.append("1. What is the current state of the GUI?")
        prompt_parts.append("2. What elements are visible and relevant to the task?")
        prompt_parts.append("3. What action should be taken next?")
        prompt_parts.append("4. Why is this action the best choice?")
        
        prompt_parts.append("\nProvide your response in JSON format:")
        prompt_parts.append('{"action_type": "click|scroll|type|...", "x": <x_coord>, "y": <y_coord>, ...}')
        
        return "\n".join(prompt_parts)


class AccessibilityTreePrompting(PromptingStrategy):
    def prepare_prompt(
        self,
        task_instruction: str,
        screenshot: Image.Image,
        previous_actions: Optional[List[Dict[str, Any]]] = None,
        step_instruction: Optional[str] = None,
        accessibility_tree: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(
            task_instruction,
            step_instruction,
            previous_actions,
            accessibility_tree
        )
        
        return {
            "prompt": prompt,
            "image": screenshot,
            "accessibility_tree": accessibility_tree
        }
    
    def _build_prompt(
        self,
        task_instruction: str,
        step_instruction: Optional[str],
        previous_actions: Optional[List[Dict[str, Any]]],
        accessibility_tree: Optional[Dict[str, Any]]
    ) -> str:
        prompt_parts = []
        
        prompt_parts.append("You are a mobile GUI agent that controls a smartphone to complete user tasks.")
        
        prompt_parts.append(f"\nTask instruction: {task_instruction}")
        
        if step_instruction:
            prompt_parts.append(f"Step instruction: {step_instruction}")
        
        if previous_actions:
            prompt_parts.append("\nPrevious actions:")
            for action in previous_actions:
                prompt_parts.append(f"  - {action}")
        
        if accessibility_tree:
            prompt_parts.append("\nAccessibility tree:")
            prompt_parts.append(self._format_tree(accessibility_tree))
        
        prompt_parts.append("\nPlease analyze the current screen and determine the next action to complete the task.")
        prompt_parts.append("Think step by step about:")
        prompt_parts.append("1. What is the current state of the GUI?")
        prompt_parts.append("2. What elements are visible and relevant to the task?")
        prompt_parts.append("3. What action should be taken next?")
        prompt_parts.append("4. Why is this action the best choice?")
        
        prompt_parts.append("\nProvide your response in JSON format:")
        prompt_parts.append('{"action_type": "click|scroll|type|...", "x": <x_coord>, "y": <y_coord>, ...}')
        
        return "\n".join(prompt_parts)
    
    def _format_tree(self, node: Dict[str, Any], indent: int = 0) -> str:
        lines = []
        prefix = "  " * indent
        
        node_type = node.get('class', 'Unknown')
        text = node.get('text', '')
        content_desc = node.get('content-desc', '')
        bounds = node.get('bounds', '')
        clickable = node.get('clickable', False)
        
        info = f"{prefix}{node_type}"
        if text:
            info += f" text='{text}'"
        if content_desc:
            info += f" desc='{content_desc}'"
        if bounds:
            info += f" bounds={bounds}"
        if clickable:
            info += " [clickable]"
        
        lines.append(info)
        
        for child in node.get('children', []):
            lines.append(self._format_tree(child, indent + 1))
        
        return "\n".join(lines)


class DirectPrompting(PromptingStrategy):
    def prepare_prompt(
        self,
        task_instruction: str,
        screenshot: Image.Image,
        previous_actions: Optional[List[Dict[str, Any]]] = None,
        step_instruction: Optional[str] = None,
        accessibility_tree: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(
            task_instruction,
            step_instruction,
            previous_actions
        )
        
        return {
            "prompt": prompt,
            "image": screenshot
        }
    
    def _build_prompt(
        self,
        task_instruction: str,
        step_instruction: Optional[str],
        previous_actions: Optional[List[Dict[str, Any]]]
    ) -> str:
        prompt_parts = []
        
        prompt_parts.append("You are a mobile GUI agent that controls a smartphone to complete user tasks.")
        
        prompt_parts.append(f"\nTask instruction: {task_instruction}")
        
        if step_instruction:
            prompt_parts.append(f"Step instruction: {step_instruction}")
        
        if previous_actions:
            prompt_parts.append("\nPrevious actions:")
            for action in previous_actions:
                prompt_parts.append(f"  - {action}")
        
        prompt_parts.append("\nPlease analyze the current screen and determine the next action to complete the task.")
        prompt_parts.append("Think step by step about:")
        prompt_parts.append("1. What is the current state of the GUI?")
        prompt_parts.append("2. What elements are visible and relevant to the task?")
        prompt_parts.append("3. What action should be taken next?")
        prompt_parts.append("4. Why is this action the best choice?")
        
        prompt_parts.append("\nProvide your response in JSON format:")
        prompt_parts.append('{"action_type": "click|scroll|type|...", "x": <x_coord>, "y": <y_coord>, ...}')
        
        return "\n".join(prompt_parts)
