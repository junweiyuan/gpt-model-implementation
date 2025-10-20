import os
import base64
import io
from typing import Optional
from PIL import Image
from .base import BaseVLM, VLMResponse

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIVLM(BaseVLM):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        use_reasoning: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ):
        super().__init__(model, api_key, use_reasoning, temperature, max_tokens)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is not installed. Install it with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=self.api_key)
        
        if use_reasoning:
            if "o1" not in model and "o3" not in model:
                print(f"Warning: Reasoning mode requested but model {model} may not support extended thinking. Consider using 'o1-preview' or 'o1-mini'")
    
    def query(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        system_prompt: Optional[str] = None
    ) -> VLMResponse:
        messages = []
        
        if system_prompt and not self.use_reasoning:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        content_parts = []
        
        if image:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
            
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
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
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        
        if not self.use_reasoning:
            kwargs["temperature"] = self.temperature
        
        try:
            response = self.client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content
            reasoning = None
            
            if hasattr(response.choices[0].message, 'reasoning_content'):
                reasoning = response.choices[0].message.reasoning_content
            
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            return VLMResponse(
                content=content,
                reasoning=reasoning,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        
        except Exception as e:
            raise RuntimeError(f"Error querying OpenAI API: {str(e)}")
