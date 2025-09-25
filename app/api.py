from fastapi import FastAPI, Query
from pydantic import BaseModel
from test import RAGPipeline
import uvicorn

app = FastAPI(title="RAG QA API")
rag = RAGPipeline()

class QuestionRequest(BaseModel):
    question: str

@app.get("/ask")
def ask(question: str = Query(..., description="Question to ask the RAG pipeline")):
    answer = rag.ask(question)
    return {"question": question, "answer": answer}

@app.post("/ask")
def ask_post(req: QuestionRequest):
    answer = rag.ask(req.question)
    return {"question": req.question, "answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
