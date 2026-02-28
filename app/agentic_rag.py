from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import operator

# Import for saving into memory
from langgraph.checkpoint.memory import MemorySaver

# Import your real retrieval + rerank logic
from app.rag import retrieve, rerank_request

#import multi model support
from app.llm_utils import fast_llm,strong_llm

llm = ChatOpenAI(model="gpt-4o-mini")

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    query: str
    retrieved_context: str | None
    critique: str
    critique_score: int
    final_answer: str
    loop_count: Annotated[int, operator.add]


# Node 1: Retrieve + Rerank
def retrieve_node(state: AgentState):
    if not state["query"]:
        return {"retrieved_context":None}
    chunks = retrieve(state["query"], top_k=30)
    top_chunks = rerank_request(state["query"], chunks, top_n=6)
    context = "\n\n---\n\n".join(top_chunks)
    return {"retrieved_context": context}


# Node 2: Critique
def critique_node(state: AgentState):
    if not state.get("retrieved_context"):
        return {
            "critique": "No retrieval was performed",
            "critique_score": 10
        }

    critique_prompt = f"""
You are a strict critic. Given the query and retrieved context, 
is the context sufficient and relevant? Rate 1-10. Suggest improvements.

Query: {state['query']}
Context: {state['retrieved_context']}
"""

    raw=fast_llm([{"role": "user", "content":critique_prompt}])

    lines = raw.strip().split("\n", 1)
    try:
        score=int(lines[0].strip())
    except:
        score=5
        
    critique_text= lines[1] if len(lines)>1 else raw

    return {
        "critique": critique,
        "critique_score": score
    }


# Node 3: Generate
def generate_node(state: AgentState):
    messages=[]
    
    if state.get("retrieved_context"):
        messages.append({
            "role": "user",
            "content": f"""Context:\n{state['retrieved_context']}
Critique: {state.get('critique', 'No critique available')}
Query: {state['query']}

Answer accurately using ONLY the provided context when available.
Be concise, factual and technical where appropriate."""
        })
    else:
        messages.append({
            "role": "user",
            "content": f"""Query: {state['query']}

You are an expert assistant. Prioritize current (2026) understanding in AI/ML/LLM context if relevant.
If the query is ambiguous (acronyms etc.), start with most common tech meaning, then briefly mention others.
Answer helpfully, accurately and concisely."""
        })

    # This is the quality-critical step → use strong_llm
    answer = strong_llm(messages)

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)]
    }


# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("critique", critique_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "critique")
workflow.add_edge("critique", "generate")
workflow.add_edge("generate", END)

#Save everything in memory to make is more than just retrieval
memory=MemorySaver()

# 🔥 THIS is what exports the agent
agent = workflow.compile(checkpointer=memory)