from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Minimal RAG API")

class QueryRequest(BaseModel):
    question: str

# 1. Connect to ChromaDB (Keep the same embedding model)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 2. Setup the LLM (Updated to gemini-3.8-flash)
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

# 3. Create the modern RAG prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

# Helper function to combine retrieved chunks
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Create the RAG Chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.post("/chat")
async def chat_with_pdf(request: QueryRequest):
    # Run the modern RAG pipeline
    answer = rag_chain.invoke(request.question)
    return {"answer": answer}

@app.get("/")
def health_check():
    return {"status": "RAG API is running"}