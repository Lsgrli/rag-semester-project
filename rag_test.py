from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List, TypedDict
from collections import defaultdict, Counter

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = BASE_DIR / "pdf_references"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# retrieve broad, then rerank/filter
TOP_K_RETRIEVE = 25
TOP_K_FINAL = 8

EMBED_MODEL = "BAAI/bge-small-en"
OLLAMA_MODEL = "qwen2:7b"

# safer if needed:
DEVICE = "cuda"   # change to "cpu" if necessary


# ============================================================
# PROMPTS
# ============================================================
PROMPT_D1 = """You are a scientific assistant.

This question may require combining information across multiple papers.
Use ONLY the provided context.
Be precise and concise.
When comparing methods, explicitly state similarities and differences.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT_D4 = """You are a scientific assistant.

Use ONLY the provided context.
Answer precisely.
If the answer is not supported by the context, say: "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT_D5 = """You are a scientific teaching assistant.

Use ONLY the provided context.

Answer clearly and pedagogically:
- define concepts if needed
- explain mechanisms for "why" questions
- explain steps for "how" questions
- compare methods explicitly when relevant
- mention limitations when relevant
- include numerical values if present

Context:
{context}

Question:
{question}

Answer:
"""


# ============================================================
# TYPES
# ============================================================
class RAGState(TypedDict, total=False):
    question: str
    dataset: str
    documents: List[Document]
    context: str
    answer: str


# ============================================================
# BUILD LLM / EMBEDDINGS
# ============================================================
def build_llm() -> ChatOllama:
    return ChatOllama(model=OLLAMA_MODEL, temperature=0.2)


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": DEVICE},
    )


llm = build_llm()
embeddings = build_embeddings()


# ============================================================
# PDF INGESTION + CHUNKING
# ============================================================
def load_pdf_documents(pdf_folder: Path) -> List[Document]:
    all_docs: List[Document] = []

    print("Loading PDFs...")

    for file in sorted(os.listdir(str(pdf_folder))):
        if not file.endswith(".pdf"):
            continue

        path = pdf_folder / file
        print("Loading:", file)

        try:
            loader = PyMuPDFLoader(str(path))
            docs = loader.load()
        except Exception as e:
            print("Skipping", file, ":", e)
            continue

        for d in docs:
            d.metadata["paper"] = file
            # helpful for retrieval
            d.page_content = f"Paper: {file}\n\n{d.page_content}"

        all_docs.extend(docs)

    print("Total pages loaded:", len(all_docs))
    return all_docs


def chunk_documents(all_docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    print("Total chunks created:", len(chunks))
    return chunks


# ============================================================
# VECTORSTORE
# ============================================================
def build_vectorstore(chunks: List[Document]) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)


# ============================================================
# LIGHT KNOWLEDGE GRAPH
# ============================================================
# In-memory graph:
# - concept_to_chunks: concept -> relevant chunks
# - concept_to_papers: concept -> papers mentioning it
# - relation_graph: concept -> related concepts (co-occurrence graph)

STOP_CONCEPTS = {
    "the", "and", "for", "with", "from", "this", "that", "using", "used",
    "paper", "model", "models", "method", "methods", "results", "data"
}


def extract_concepts_fast(text: str) -> List[str]:
    # heuristic scientific concept extraction
    raw = re.findall(r"[A-Z][a-zA-Z0-9\-]{2,}", text)
    concepts = []
    for w in raw:
        wl = w.lower()
        if wl not in STOP_CONCEPTS:
            concepts.append(w)
    # keep order, remove duplicates
    seen = set()
    out = []
    for c in concepts:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            out.append(c)
    return out[:12]


def build_knowledge_graph(chunks: List[Document]):
    concept_to_chunks = defaultdict(list)
    concept_to_papers = defaultdict(set)
    relation_graph = defaultdict(Counter)

    print("Building lightweight knowledge graph...")

    # you can raise this limit later; keeping it moderate helps runtime
    for chunk in chunks:
        concepts = extract_concepts_fast(chunk.page_content)
        paper = chunk.metadata.get("paper", "unknown")

        lowered = [c.lower() for c in concepts]

        for c in lowered:
            concept_to_chunks[c].append(chunk)
            concept_to_papers[c].add(paper)

        # co-occurrence graph inside the chunk
        for i, c1 in enumerate(lowered):
            for j, c2 in enumerate(lowered):
                if i != j:
                    relation_graph[c1][c2] += 1

    return concept_to_chunks, concept_to_papers, relation_graph


# ============================================================
# RETRIEVAL HELPERS
# ============================================================
def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9\-]+\b", text.lower())


def lexical_overlap_score(question: str, text: str) -> int:
    q_words = set(tokenize(question))
    t_words = tokenize(text)
    return sum(1 for w in t_words if w in q_words)


def simple_rerank(question: str, docs: List[Document]) -> List[Document]:
    scored = []
    for d in docs:
        score = lexical_overlap_score(question, d.page_content)
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored]


def dedup_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        key = (
            d.metadata.get("paper", ""),
            d.metadata.get("page", ""),
            d.page_content[:200],
        )
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def group_by_paper(docs: List[Document]):
    papers = defaultdict(list)
    for d in docs:
        p = d.metadata.get("paper", "unknown")
        papers[p].append(d)
    return papers


def paper_level_rerank(question: str, docs: List[Document], top_papers: int = 3, max_chunks_per_paper: int = 2) -> List[Document]:
    papers = group_by_paper(docs)

    paper_scores = []
    for paper, p_docs in papers.items():
        # score paper by frequency + lexical support
        freq = len(p_docs)
        lex = max(lexical_overlap_score(question, d.page_content) for d in p_docs)
        score = 2 * freq + lex
        paper_scores.append((score, paper, p_docs))

    paper_scores.sort(key=lambda x: x[0], reverse=True)

    selected = []
    for _, paper, p_docs in paper_scores[:top_papers]:
        reranked = simple_rerank(question, p_docs)
        selected.extend(reranked[:max_chunks_per_paper])

    return selected


def expand_with_graph(question: str, concept_to_chunks, relation_graph, hop_limit: int = 2) -> List[Document]:
    q_concepts = [c.lower() for c in extract_concepts_fast(question)]
    graph_docs = []

    for qc in q_concepts:
        graph_docs.extend(concept_to_chunks.get(qc, []))

        # 1-hop neighbors: related concepts
        neighbors = relation_graph.get(qc, {})
        top_neighbors = [c for c, _ in neighbors.most_common(hop_limit)]
        for nb in top_neighbors:
            graph_docs.extend(concept_to_chunks.get(nb, []))

    return graph_docs


def retrieve_for_d1(question: str, retriever) -> List[Document]:
    # broad retrieve, then paper-aware rerank for cross-paper reasoning
    docs = retriever.invoke(question)
    docs = dedup_docs(docs)
    docs = paper_level_rerank(question, docs, top_papers=4, max_chunks_per_paper=2)
    return docs[:TOP_K_FINAL]


def retrieve_for_d4(question: str, retriever) -> List[Document]:
    # keep retrieval cleaner for evaluation
    docs = retriever.invoke(question)
    docs = dedup_docs(docs)
    docs = simple_rerank(question, docs)
    return docs[:TOP_K_FINAL]


def retrieve_for_d5(question: str, retriever, concept_to_chunks, relation_graph) -> List[Document]:
    vector_docs = retriever.invoke(question)
    graph_docs = expand_with_graph(question, concept_to_chunks, relation_graph, hop_limit=2)

    docs = vector_docs + graph_docs
    docs = dedup_docs(docs)

    # for D5, use paper-aware rerank after graph expansion
    docs = paper_level_rerank(question, docs, top_papers=4, max_chunks_per_paper=2)
    docs = simple_rerank(question, docs)

    return docs[:TOP_K_FINAL]


# ============================================================
# BUILD EVERYTHING
# ============================================================
all_docs = load_pdf_documents(PDF_FOLDER)
chunks = chunk_documents(all_docs)
vectorstore = build_vectorstore(chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})

concept_to_chunks, concept_to_papers, relation_graph = build_knowledge_graph(chunks)


# ============================================================
# NODES
# ============================================================
def retrieve_node(state: RAGState) -> dict:
    question = state["question"]
    dataset = state.get("dataset", "D4")

    if dataset == "D1":
        docs = retrieve_for_d1(question, retriever)
    elif dataset == "D5":
        docs = retrieve_for_d5(question, retriever, concept_to_chunks, relation_graph)
    else:
        docs = retrieve_for_d4(question, retriever)

    return {"documents": docs}


def format_context_node(state: RAGState) -> dict:
    docs = state.get("documents", [])

    parts = []
    for i, d in enumerate(docs, start=1):
        paper = d.metadata.get("paper", "unknown")
        page = d.metadata.get("page", "NA")
        parts.append(f"[Doc {i} | {paper} p.{page}]\n{d.page_content}")

    return {"context": "\n\n---\n\n".join(parts)}


def generate_node(state: RAGState) -> dict:
    dataset = state.get("dataset", "D4")

    if dataset == "D1":
        prompt_tmpl = PROMPT_D1
    elif dataset == "D5":
        prompt_tmpl = PROMPT_D5
    else:
        prompt_tmpl = PROMPT_D4

    prompt_text = prompt_tmpl.format(
        context=state.get("context", ""),
        question=state["question"],
    )

    resp = llm.invoke(prompt_text)

    return {
        "answer": resp.content,
        "documents": state.get("documents", []),
    }


# ============================================================
# LANGGRAPH
# ============================================================
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("format_context", format_context_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "format_context")
graph.add_edge("format_context", "generate")
graph.add_edge("generate", END)

app = graph.compile()


# ============================================================
# API
# ============================================================
def graphrag_query(question: str, dataset: str = "D4"):
    result = app.invoke({
        "question": question,
        "dataset": dataset,
    })

    docs = result.get("documents", [])

    return {
        "answer": result["answer"],
        "documents": docs,
        "papers": [d.metadata.get("paper", "") for d in docs],
    }


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    q1 = "Compare AxCaliber and ActiveAx"
    q2 = "Why does NODDI fail in grey matter?"

    print("\n===== D1 =====")
    r1 = graphrag_query(q1, "D1")
    print(r1["answer"])
    print(r1["papers"])

    print("\n===== D5 =====")
    r2 = graphrag_query(q2, "D5")
    print(r2["answer"])
    print(r2["papers"])