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
    return response.choices[0].messages.content.strip()

def strong_llm(
    messages:List[Dict[str,str]],
    temperature:float=0.2,
    max_tokens:int = 2048,
    **kwargs:Any
) -> str:
    response=completion(
        model='anthropic/claude-3-5-sonnet-20241022',
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    return response.choices[0].messages.content.strip()

