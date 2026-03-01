import os
from litellm import completion
from typing import List,Dict,Any

def fast_llm(
    messages:List[Dict[str,str]],
    temperature: float = 0.3,
    max_tokens: int = 512,
    **kwargs: Any
) -> str:
    response = completion(
        model= 'openai/gpt-4o-mini',
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    return response.choices[0].message.content.strip()

def strong_llm(
    messages:List[Dict[str,str]],
    temperature:float=0.2,
    max_tokens:int = 2048,
    **kwargs:Any
) -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        model_name = "anthropic/claude-sonnet-4-20250514"
    else:
        model_name = "openai/gpt-4o"
    response=completion(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    return response.choices[0].message.content.strip()

