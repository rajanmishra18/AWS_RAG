from fastapi import FastAPI
from uuid import uuid4
from pydantic import BaseModel
from app.rag import answer_question
from app.agentic_rag import agent
from langchain_core.messages import HumanMessage

app = FastAPI(
    title='AWS RAG API',
    description='A simple RAG application',
    version='1.0'
)

class QuestionRequest(BaseModel):
    question:str
    thread_Id:str | None=None # Optional parameter, Client may send

@app.get('/')
def health_check():
    return {'status':'RAG api is running.'}

@app.post('/ask')
def ask_question(request:QuestionRequest):
    answer= answer_question(request.question)
    return {
        'question': request.question,
        'answer':answer
    }

@app.post("/agent_query")
async def agent_query(request: QuestionRequest):

    #create or reuse thread id
    thread_Id=request.thread_Id or uuid4.thread_Id

    #pass the thread id to langgraph
    config={"configurable":{"thread_id": thread_Id}}

    input ={
        "query": request.question,
        "messages": [HumanMessage(content=request.question)],
        "retrieved_context":None,
        "critique":"",
        "critique_score":0,
        "final_answer":"",
        "loop_count":0
    }

    result = agent.invoke(input=input,config=config)

    return {
        "answer": result.get("final_answer"),
        "critique": result.get("critique"),
        "thread_id": thread_Id
    }