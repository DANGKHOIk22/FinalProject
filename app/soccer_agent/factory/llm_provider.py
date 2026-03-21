from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.config import DEFAULT_MODEL, MODEL_TEMPERATURE, MODEL_TOP_P, MAX_COMPLETION_TOKENS
from app.config.settings import settings

def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = MODEL_TEMPERATURE,
    top_p: float = MODEL_TOP_P,
    max_tokens: int = MAX_COMPLETION_TOKENS,
    api_key: Optional[str] = None,
    **kwargs
) -> ChatGoogleGenerativeAI:
    """
    Factory function to create a ChatGoogleGenerativeAI instance.
    
    Args:
        model: The Google GenAI model to use (default from config)
        temperature: Temperature for generation
        top_p: Top P sampling parameter
        max_tokens: Maximum output tokens
        api_key: Optional explicit API key. If not provided, it uses settings.GOOGLE_API_KEY.
        **kwargs: Additional parameters passed to ChatGoogleGenerativeAI
        
    Returns:
        ChatGoogleGenerativeAI instance
    """
    key = api_key or settings.GOOGLE_API_KEY
    if not key:
        raise ValueError("GOOGLE_API_KEY must be provided or configured in settings.")
        
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
        api_key=key,
        **kwargs
    )
