"""
graph_rag.py — Classic RAG implemented with LangGraph + LangChain components + Ollama

What it does (end-to-end):
1) Load PDF with PyMuPDFLoader
2) Split into chunks
3) Embed chunks with HuggingFace embeddings
4) Store in Qdrant (in-memory by default)
5) LangGraph workflow:
   START → retrieve → format_context → generate → END

Prereqs:
- Ollama running on the same machine (or reachable via base_url)
  e.g. on server:  ollama serve
  and pull a model: ollama pull qwen2:7b

Install (rag_env):
pip install langgraph langchain langchain-community langchain-text-splitters \
    langchain-huggingface langchain-ollama qdrant-client pymupdf sentence-transformers
"""
from __future__ import annotations
import os

from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

#from langchain_community.vectorstores import Qdrant
#from qdrant_client import QdrantClient
from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph, START, END


# -----------------------------
# 1) CONFIG
# -----------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = BASE_DIR / "pdf_references"


# For older GPUs / limited RAM
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embeddings
EMBED_MODEL = "BAAI/bge-small-en"  

# Retriever
TOP_K = 10

# Ollama model (e.g. `ollama pull qwen2:7b`)
OLLAMA_MODEL = "qwen2:7b"
OLLAMA_BASE_URL = None  # "http://localhost:11434"

# Prompt for RAG 
PROMPT_TMPL = """You are a scientific assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say: "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""


# -----------------------------
# 2) BUILD RAG COMPONENTS
# -----------------------------
def build_vectorstore(pdf_folder: str):

    all_docs = []

    print("Loading PDFs...")

    for file in sorted(os.listdir(str(pdf_folder))):

        if not file.endswith(".pdf"):
            continue

        path = os.path.join(pdf_folder, file)

        print("Loading:", file)

        try:
            loader = PyMuPDFLoader(path)
            docs = loader.load()
        except Exception as e:
            print("Skipping", file, ":", e)
            continue

        for d in docs:
            d.metadata["paper"] = file

        all_docs.extend(docs)

    print("Total pages loaded:", len(all_docs))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(all_docs)

    print("Total chunks created:", len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        # model_kwargs={"device": "cpu"},
        model_kwargs={"device": "cuda"}
    )

    '''
    vectorstore = Qdrant.from_documents(
    documents=chunks,
    embedding=embeddings,
    location=":memory:",   # in-memory DB
    collection_name="papers",
)
'''

    vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

    return vectorstore


def build_llm() -> ChatOllama:
    # If Ollama is remote or forwarded, set OLLAMA_BASE_URL
    kwargs = {"model": OLLAMA_MODEL, "temperature": 0.2}
    if OLLAMA_BASE_URL:
        kwargs["base_url"] = OLLAMA_BASE_URL
    return ChatOllama(**kwargs)


# Build globally once (so we don't re-embed every question)
vectorstore = build_vectorstore(PDF_FOLDER)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
llm = build_llm()

prompt = PromptTemplate(
    template=PROMPT_TMPL,
    input_variables=["context", "question"],
)


# -----------------------------
# 3) LANGGRAPH STATE
# -----------------------------
class RAGState(TypedDict, total=False):
    question: str
    documents: List[Document]
    context: str
    answer: str


# -----------------------------
# 4) LANGGRAPH NODES
# -----------------------------
def retrieve_node(state: RAGState) -> dict:
    """Vector DB lookup.
    Reads:  state["question"]
    Writes: state["documents"]
    """
    question = state["question"]
    docs = retriever.invoke(question)  # returns List[Document]
    return {"documents": docs}


def format_context_node(state: RAGState) -> dict:
    """Format retrieved docs into a single string context.
    Reads:  state["documents"]
    Writes: state["context"]
    """
    docs = state.get("documents", [])
    # Include metadata (page/source) so you can cite later
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "NA")
        parts.append(f"[Doc {i} | {src} p.{page}]\n{d.page_content}")
    context = "\n\n---\n\n".join(parts)
    return {"context": context}


def generate_node(state: RAGState) -> dict:
    """LLM generates answer from context.
    Reads:  state["question"], state["context"]
    Writes: state["answer"]
    """
    question = state["question"]
    context = state.get("context", "")

    prompt_text = prompt.format(context=context, question=question)
    resp = llm.invoke(prompt_text)

    # ChatOllama returns an AIMessage; get text via .content
    return {"answer": resp.content}


# -----------------------------
# 5) BUILD + COMPILE GRAPH
# -----------------------------
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("format_context", format_context_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "format_context")
graph.add_edge("format_context", "generate")
graph.add_edge("generate", END)

app = graph.compile()


# -----------------------------
# 6) RUN
# -----------------------------
if __name__ == "__main__":
    question = "What is Monte Carlo simulation in diffusion MRI?"
    result = app.invoke({"question": question})

    print("\n==================== QUESTION ====================")
    print(question)
    print("\n===================== ANSWER =====================")
    print(result["answer"])