import os
import base64
import io
from typing import Optional
from PIL import Image
from .base import BaseVLM, VLMResponse

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ClaudeVLM(BaseVLM):
    def __init__(
        self,
        model: str = "claude-3-7-sonnet-20250219",
        api_key: Optional[str] = None,
        use_reasoning: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ):
        super().__init__(model, api_key, use_reasoning, temperature, max_tokens)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is not installed. Install it with: pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        if use_reasoning and "sonnet" not in model.lower():
            print(f"Warning: Reasoning mode requested but model {model} may not support extended thinking")
    
    def query(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        system_prompt: Optional[str] = None
    ) -> VLMResponse:
        messages = []
        
        content_parts = []
        
        if image:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
            
            content_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_base64
                }
            })
        
        content_parts.append({
            "type": "text",
            "text": prompt
        })
        
        messages.append({
            "role": "user",
            "content": content_parts
        })
        
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        if self.use_reasoning:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 2000
            }
        
        try:
            response = self.client.messages.create(**kwargs)
            
            content = ""
            reasoning = None
            
            for block in response.content:
                if block.type == "thinking":
                    reasoning = block.thinking
                elif block.type == "text":
                    content = block.text
            
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            return VLMResponse(
                content=content,
                reasoning=reasoning,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        
        except Exception as e:
            raise RuntimeError(f"Error querying Claude API: {str(e)}")
