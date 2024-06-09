from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv("google_api_key")
os.environ["google_api_key"] = google_api_key

app=FastAPI(
    title="Langchain server",
    version="1.0",
    description="A simple API Server"
)

add_routes(
    app,
    ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=google_api_key),
    path="/gemini"
    
)
model=ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=google_api_key)
llm=Ollama(model="llama2")
prompt1=ChatPromptTemplate.from_template("Write me an essay about some {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about some {topic} with 100 words")

add_routes(
    app,
    prompt1|model,
    path="/essay"
    
)
add_routes(
    app,
    prompt2|llm,
    path="/poem"
    
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)