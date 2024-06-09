from langchain_community.vectorstores import FAISS


from langchain_groq import ChatGroq
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)
groq_api_key=os.environ['groq_api_key']
# os.environ['groq_api_key']=groq_api_key
gemini_api_key=os.environ['google_api_key']
# Create Google Palm LLM model
llm=ChatGroq(groq_api_key=groq_api_key,
                model_name="Gemma-7b-It")
# llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
# # Initialize instructor embeddings using the Hugging Face model
embeddings=GoogleGenerativeAIEmbeddings(gemini_api_key=gemini_api_key,model="models/embedding-001")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='query_respons.csv', source_column="Query")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # prompt_template = """Given the following context and a question, generate an answer based on this context only.
    # In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    # If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    # CONTEXT: {context}

    # QUESTION: {question}"""
    prompt_template = """Given the following context and a question, generate an answer based solely on this context.

    In the answer, strive to provide as much relevant text as possible from the "response" section of the source documents, minimizing modifications.

    If the answer cannot be located within the context, kindly state "I don't know" instead of fabricating a response.

    CONTEXT: {context}

    QUESTION: {question}"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))