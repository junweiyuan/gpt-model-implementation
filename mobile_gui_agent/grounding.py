from typing import Tuple, Optional, List
from pydantic import BaseModel
import re


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    
    def to_center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_normalized(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            x1=self.x1 / width,
            y1=self.y1 / height,
            x2=self.x2 / width,
            y2=self.y2 / height
        )
    
    def to_pixel(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            x1=int(self.x1 * width),
            y1=int(self.y1 * height),
            x2=int(self.x2 * width),
            y2=int(self.y2 * height)
        )


class GroundingResult(BaseModel):
    element_description: str
    bounding_box: Optional[BoundingBox] = None
    center_point: Optional[Tuple[float, float]] = None
    confidence: float = 1.0
    
    def get_click_coordinates(self, width: int, height: int) -> Tuple[int, int]:
        if self.center_point:
            if 0 <= self.center_point[0] <= 1 and 0 <= self.center_point[1] <= 1:
                return (int(self.center_point[0] * width), int(self.center_point[1] * height))
            else:
                return (int(self.center_point[0]), int(self.center_point[1]))
        
        if self.bounding_box:
            center = self.bounding_box.to_center()
            if 0 <= center[0] <= 1 and 0 <= center[1] <= 1:
                return (int(center[0] * width), int(center[1] * height))
            else:
                return (int(center[0]), int(center[1]))
        
        return (width // 2, height // 2)


class GroundingParser:
    @staticmethod
    def parse_bounding_box(text: str) -> Optional[BoundingBox]:
        patterns = [
            r'\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]',
            r'\((\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\)',
            r'x1[=:](\d+\.?\d*).*?y1[=:](\d+\.?\d*).*?x2[=:](\d+\.?\d*).*?y2[=:](\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                coords = [float(x) for x in match.groups()]
                return BoundingBox(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])
        
        return None
    
    @staticmethod
    def parse_center_point(text: str) -> Optional[Tuple[float, float]]:
        patterns = [
            r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)',
            r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]',
            r'x[=:](\d+\.?\d*).*?y[=:](\d+\.?\d*)',
            r'"x"\s*:\s*(\d+\.?\d*).*?"y"\s*:\s*(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                coords = [float(x) for x in match.groups()]
                return (coords[0], coords[1])
        
        return None
    
    @staticmethod
    def parse_coordinates(text: str) -> Optional[Tuple[float, float]]:
        bbox = GroundingParser.parse_bounding_box(text)
        if bbox:
            return bbox.to_center()
        
        center = GroundingParser.parse_center_point(text)
        if center:
            return center
        
        return None
