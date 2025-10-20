from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel
from PIL import Image


class VLMResponse(BaseModel):
    content: str
    reasoning: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    input_tokens: int = 0
    output_tokens: int = 0
    
    def get_total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseVLM(ABC):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        use_reasoning: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ):
        self.model = model
        self.api_key = api_key
        self.use_reasoning = use_reasoning
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def query(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        system_prompt: Optional[str] = None
    ) -> VLMResponse:
        pass
    
    def supports_reasoning(self) -> bool:
        return self.use_reasoning
    
    def get_model_name(self) -> str:
        return self.model
