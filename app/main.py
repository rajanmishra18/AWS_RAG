from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import answer_question

app = FastAPI(
    title='AWS RAG API',
    description='A simple RAG application',
    version='1.0'
)

class QuestionRequest(BaseModel):
    question:str

@app.get('/')
def health_check():
    return {'status':'RAG api is running.'}

@app.post('/')
def ask_question(request:QuestionRequest):
    answer= answer_question(request.question)
    return {
        'question': request.question,
        'answer':answer
    }