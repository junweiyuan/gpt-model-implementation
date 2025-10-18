import os
from typing import Optional
from PIL import Image
from .base import BaseVLM, VLMResponse

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class GeminiVLM(BaseVLM):
    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        use_reasoning: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ):
        super().__init__(model, api_key, use_reasoning, temperature, max_tokens)
        
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai package is not installed. Install it with: pip install google-generativeai")
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        
        if use_reasoning:
            if "thinking" not in model.lower():
                print(f"Warning: Reasoning mode requested. Consider using 'gemini-2.0-flash-thinking-exp' model")
            self.model = model
        
        self.client = genai.GenerativeModel(self.model)
    
    def query(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        system_prompt: Optional[str] = None
    ) -> VLMResponse:
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        
        content_parts = []
        
        if system_prompt:
            content_parts.append(system_prompt + "\n\n")
        
        if image:
            content_parts.append(image)
        
        content_parts.append(prompt)
        
        try:
            response = self.client.generate_content(
                content_parts,
                generation_config=generation_config
            )
            
            content = response.text
            reasoning = None
            
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    reasoning = str(candidate.grounding_metadata)
            
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage_metadata'):
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            return VLMResponse(
                content=content,
                reasoning=reasoning,
                raw_response=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        
        except Exception as e:
            raise RuntimeError(f"Error querying Gemini API: {str(e)}")
