from __future__ import annotations

import os
import re
import pandas as pd
import networkx as nx
import csv

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, TypedDict
from collections import defaultdict, Counter

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
#from langchain_huggingface import HuggingFaceEmbeddings

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = BASE_DIR / "pdf_references"

@dataclass
class ExperimentConfig:
    llm_backend: str = "ollama"
    llm_model: str = "qwen2:7b"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    embed_model: str = "BAAI/bge-small-en"
    chunk_size: int = 800
    chunk_overlap: int = 150
    chunk_strategy: str = "fixed"   # fixed or paragraph

    top_k_retrieve: int = 25
    top_k_final: int = 8

    retrieval_mode: str = "dense"   # "dense", "bm25", "hybrid"
    bm25_k: int = 25
    hybrid_mode: str = "rrf"        # "concat" or "rrf"
    rrf_k: int = 60

    use_graph: bool = True
    graph_hops: int = 1
    edge_min_shared_keywords: int = 1

    use_rerank: bool = True
    top_papers_d1: int = 4
    top_papers_d5: int = 4
    max_chunks_per_paper: int = 2
    use_chunk_edges: bool = True
    use_sequential_edges: bool = True
    max_graph_seed_chunks: int = 5
    max_edge_neighbors_per_chunk: int = 5

    # ----------------------------
    # Graph parameters
    # ----------------------------
    graph_edge_mode: str = "keyword"
    # options: "keyword", "similarity", "hybrid"

    edge_similarity_threshold: float = 0.76
    max_similarity_neighbors: int = 5
    max_similarity_chunks: int = 5000

    # ----------------------------
    # Reranking
    # ----------------------------
    use_cross_encoder_rerank: bool = False
    cross_encoder_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 50
    
    device: str = "cuda"   # or cpu

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
# COMMON UTILITIES
# ============================================================

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9\-]+\b", text.lower())

def doc_key(d: Document):
    return (
        d.metadata.get("paper", ""),
        d.metadata.get("page", ""),
        d.page_content[:200],
    )

def dedup_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []

    for d in docs:
        key = doc_key(d)
        if key not in seen:
            seen.add(key)
            out.append(d)

    return out

def normalize_vectors(vectors):
    import numpy as np

    X = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def group_by_paper(docs: List[Document]):
    papers = defaultdict(list)
    for d in docs:
        p = d.metadata.get("paper", "unknown")
        papers[p].append(d)
    return papers

# ============================================================
# MODEL BUILDERS
# ============================================================

def build_llm(cfg):
    if cfg.llm_backend == "ollama":
        return ChatOllama(
            model=cfg.llm_model,
            temperature=0.2,
        )

    if cfg.llm_backend == "vllm":
        return ChatOpenAI(
            model=cfg.llm_model,
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
            temperature=0.2,
        )

    raise ValueError(f"Unknown llm_backend: {cfg.llm_backend}")

def build_embeddings(cfg: ExperimentConfig) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=cfg.embed_model,
        model_kwargs={"device": cfg.device},
    )

# ============================================================
# PDF INGESTION
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
            d.page_content = f"Paper: {file}\n\n{d.page_content}"

        all_docs.extend(docs)

    print("Total pages loaded:", len(all_docs))
    return all_docs

# ============================================================
# CHUNKING
# ============================================================

def chunk_documents(all_docs: List[Document], cfg: ExperimentConfig) -> List[Document]:
    if cfg.chunk_strategy == "fixed":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        chunks = splitter.split_documents(all_docs)

    elif cfg.chunk_strategy == "paragraph":
        chunks = []
        for d in all_docs:
            paragraphs = [p.strip() for p in d.page_content.split("\n\n") if p.strip()]
            current = ""

            for p in paragraphs:
                if len(current) + len(p) <= cfg.chunk_size:
                    current = f"{current}\n\n{p}" if current else p
                else:
                    if current:
                        chunks.append(
                            Document(
                                page_content=current,
                                metadata=d.metadata.copy(),
                            )
                        )
                    current = p

            if current:
                chunks.append(
                    Document(
                        page_content=current,
                        metadata=d.metadata.copy(),
                    )
                )
    else:
        raise ValueError(f"Unknown chunk strategy: {cfg.chunk_strategy}")

    print("Total chunks created:", len(chunks))
    return chunks

# ============================================================
# VECTORSTORE
# ============================================================

def build_vectorstore(chunks: List[Document], embeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)

# ============================================================
# RETRIEVAL
# ============================================================

def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    rrf_k: int = 60,
) -> List[Document]:
    """
    Fuse several ranked document lists using Reciprocal Rank Fusion.

    Score(doc) = sum over rankings 1 / (rrf_k + rank)
    """
    scores = {}
    doc_map = {}

    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = doc_key(doc)

            if key not in doc_map:
                doc_map[key] = doc

            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    return [doc_map[k] for k in ranked_keys]

def interleave_ranked_lists(
    ranked_lists: List[List[Document]],
) -> List[Document]:
    out = []
    seen = set()

    max_len = max(len(docs) for docs in ranked_lists) if ranked_lists else 0

    for i in range(max_len):
        for docs in ranked_lists:
            if i < len(docs):
                d = docs[i]
                key = doc_key(d)
                if key not in seen:
                    seen.add(key)
                    out.append(d)

    return out

def hybrid_concat_retrieve(
    question: str,
    dense_retriever,
    bm25_retriever,
) -> List[Document]:
    dense_docs = dense_retriever.invoke(question)
    bm25_docs = bm25_retriever.invoke(question)

    return interleave_ranked_lists([dense_docs, bm25_docs])


def hybrid_rrf_retrieve(
    question: str,
    dense_retriever,
    bm25_retriever,
    cfg: ExperimentConfig,
) -> List[Document]:
    """
    Better hybrid retrieval:
    fuse dense and BM25 rankings with RRF.
    """
    dense_docs = dense_retriever.invoke(question)
    bm25_docs = bm25_retriever.invoke(question)

    return reciprocal_rank_fusion(
        [dense_docs, bm25_docs],
        rrf_k=cfg.rrf_k,
    )

class ConfigurableRetriever:
    """
    Wrapper with the same interface as LangChain retrievers.

    It allows the rest of the code to keep calling:
        retriever.invoke(question)

    while switching between:
        - dense
        - bm25
        - hybrid
    """

    def __init__(self, dense_retriever, bm25_retriever, cfg: ExperimentConfig):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.cfg = cfg

    def invoke(self, question: str) -> List[Document]:
        if self.cfg.retrieval_mode == "dense":
            return self.dense_retriever.invoke(question)

        if self.cfg.retrieval_mode == "bm25":
            return self.bm25_retriever.invoke(question)

        if self.cfg.retrieval_mode == "hybrid":
            if self.cfg.hybrid_mode == "concat":
                return hybrid_concat_retrieve(
                    question,
                    self.dense_retriever,
                    self.bm25_retriever,
                )

            if self.cfg.hybrid_mode == "rrf":
                return hybrid_rrf_retrieve(
                    question,
                    self.dense_retriever,
                    self.bm25_retriever,
                    self.cfg,
                )

            raise ValueError(f"Unknown hybrid_mode: {self.cfg.hybrid_mode}")

        raise ValueError(f"Unknown retrieval_mode: {self.cfg.retrieval_mode}")
    
# ============================================================
# GRAPH: CONCEPT EXTRACTION
# ============================================================

def extract_domain_concepts(text: str) -> List[str]:
    """
    Domain-aware concept extraction for the final graph.

    Captures:
    - DW-MRI domain phrases
    - acronyms
    - filtered scientific noun-like phrases
    """

    text_low = text.lower()

    domain_phrases = [
        "axon diameter",
        "axon radius",
        "cylinder radius",
        "cylinder diameter",
        "fiber volume fraction",
        "intra-cellular volume fraction",
        "intra cellular volume fraction",
        "extra-cellular volume fraction",
        "extra cellular volume fraction",
        "orientation dispersion",
        "neurite density",
        "restricted diffusion",
        "hindered diffusion",
        "grey matter",
        "gray matter",
        "white matter",
        "diffusion mri",
        "diffusion weighted imaging",
        "diffusion signal",
        "monte carlo simulation",
        "microstructure model",
        "compartment model",
        "soma size",
        "cell body",
        "dendrite radius",
        "permeability",
        "exchange time",
        "activeax",
        "axcaliber",
        "noddi",
        "charmed",
        "dti",
        "dki",
        "mc-dc",
        "cactus",
    ]

    concepts = []

    for phrase in domain_phrases:
        if phrase in text_low:
            concepts.append(phrase)

    acronyms = re.findall(r"\b[A-Z][A-Z0-9\-]{2,}\b", text)
    for a in acronyms:
        concepts.append(a.lower())

    phrase_candidates = re.findall(
        r"\b[a-z][a-z\-]+(?:\s+[a-z][a-z\-]+){1,3}\b",
        text_low,
    )

    stop_words = {
        "the", "and", "for", "with", "from", "this", "that",
        "using", "used", "paper", "model", "models", "method",
        "methods", "results", "data", "figure", "table",
        "study", "studies", "section", "equation", "approach",
        "analysis", "experiment", "experiments", "introduction",
        "discussion", "conclusion", "abstract"
    }

    for cand in phrase_candidates:
        words = cand.split()

        if any(w in stop_words for w in words):
            continue

        if len(cand) < 8:
            continue

        concepts.append(cand)

    seen = set()
    out = []

    for c in concepts:
        c = c.strip().lower()
        if c and c not in seen:
            seen.add(c)
            out.append(c)

    return out[:30]

# ============================================================
# GRAPH BUILDING
# ============================================================

def build_chunk_lookup(chunks: List[Document]):
    return {
        doc_key(ch): i
        for i, ch in enumerate(chunks)
    }

def build_graph(
    chunks: List[Document],
    cfg: ExperimentConfig,
    embeddings=None,
):
    """
    Final graph builder.

    Nodes:
    - chunks

    Edges:
    - keyword/concept overlap edges
    - sequential edges inside the same paper
    - optional semantic similarity edges
    """

    concept_to_chunks = defaultdict(list)
    concept_to_papers = defaultdict(set)
    relation_graph = defaultdict(Counter)

    chunk_edges = defaultdict(list)
    sequential_graph = defaultdict(list)
    similarity_edges = defaultdict(list)

    graph_edge_mode = getattr(cfg, "graph_edge_mode", "keyword")
    edge_similarity_threshold = getattr(cfg, "edge_similarity_threshold", 0.76)
    max_similarity_neighbors = getattr(cfg, "max_similarity_neighbors", 5)
    max_similarity_chunks = getattr(cfg, "max_similarity_chunks", 5000)

    print("Building FINAL graph...")
    print("Graph edge mode:", graph_edge_mode)

    concept_to_chunk_ids = defaultdict(list)

    # --------------------------------------------------
    # 1. Domain-aware concept extraction
    # --------------------------------------------------
    for idx, chunk in enumerate(chunks):
        concepts = extract_domain_concepts(chunk.page_content)
        paper = chunk.metadata.get("paper", "unknown")

        for c in concepts:
            concept_to_chunks[c].append(chunk)
            concept_to_papers[c].add(paper)
            concept_to_chunk_ids[c].append(idx)

        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                if i != j:
                    relation_graph[c1][c2] += 1

    # --------------------------------------------------
    # 2. Keyword / concept-overlap edges
    # --------------------------------------------------
    if graph_edge_mode in {"keyword", "hybrid"}:
        print("Building concept-overlap edges...")

        pair_counts = defaultdict(int)

        for concept, ids in concept_to_chunk_ids.items():
            ids = list(set(ids))

            for pos_i in range(len(ids)):
                for pos_j in range(pos_i + 1, len(ids)):
                    i = ids[pos_i]
                    j = ids[pos_j]
                    pair_counts[(i, j)] += 1

        edge_sets = defaultdict(set)

        for (i, j), count in pair_counts.items():
            if count >= cfg.edge_min_shared_keywords:
                edge_sets[i].add(j)
                edge_sets[j].add(i)

        chunk_edges = defaultdict(list, {
            k: sorted(v)
            for k, v in edge_sets.items()
        })

    # --------------------------------------------------
    # 3. Sequential edges inside same paper
    # --------------------------------------------------
    seq_sets = defaultdict(set)

    for i in range(len(chunks) - 1):
        p1 = chunks[i].metadata.get("paper", "")
        p2 = chunks[i + 1].metadata.get("paper", "")

        if p1 == p2:
            seq_sets[i].add(i + 1)
            seq_sets[i + 1].add(i)

    sequential_graph = defaultdict(list, {
        k: sorted(v)
        for k, v in seq_sets.items()
    })

    # --------------------------------------------------
    # 4. Optional semantic similarity edges
    # --------------------------------------------------
    if graph_edge_mode in {"similarity", "hybrid"}:
        if embeddings is None:
            print("Skipping similarity edges: embeddings object is None.")
        elif len(chunks) > max_similarity_chunks:
            print(
                f"Skipping similarity edges: {len(chunks)} chunks > "
                f"max_similarity_chunks={max_similarity_chunks}."
            )
        else:
            print("Building semantic similarity edges...")

            import numpy as np

            texts = [c.page_content for c in chunks]
            vectors = embeddings.embed_documents(texts)
            X = normalize_vectors(vectors)

            sim_matrix = X @ X.T

            sim_sets = defaultdict(set)

            for i in range(len(chunks)):
                sims = sim_matrix[i]
                candidate_ids = np.argsort(-sims)

                added = 0

                for j in candidate_ids:
                    j = int(j)

                    if i == j:
                        continue

                    sim = float(sims[j])

                    if sim < edge_similarity_threshold:
                        break

                    sim_sets[i].add(j)
                    sim_sets[j].add(i)

                    added += 1

                    if added >= max_similarity_neighbors:
                        break

            similarity_edges = defaultdict(list, {
                k: sorted(v)
                for k, v in sim_sets.items()
            })

    print("Final graph built.")
    print("Concept edges:", sum(len(v) for v in chunk_edges.values()))
    print("Sequential edges:", sum(len(v) for v in sequential_graph.values()))
    print("Similarity edges:", sum(len(v) for v in similarity_edges.values()))

    return (
        concept_to_chunks,
        concept_to_papers,
        relation_graph,
        chunk_edges,
        sequential_graph,
        similarity_edges,
    )

# ============================================================
# GRAPH: EXPANSION
# ============================================================

def expand_with_final_graph(
    question: str,
    seed_docs: List[Document],
    concept_to_chunks,
    relation_graph,
    chunk_edges,
    sequential_graph,
    similarity_edges,
    chunk_id_to_doc,
    chunks,
    cfg: ExperimentConfig,
) -> List[Document]:
    """
    Final graph expansion.

    Main idea:
    - start from initially retrieved seed chunks
    - optionally add concept-based seed chunks
    - traverse selected graph edges
    - limit branching
    - locally sort neighbors by lexical overlap with the query
    """

    graph_edge_mode = getattr(cfg, "graph_edge_mode", "keyword")
    use_similarity_edges = graph_edge_mode in {"similarity", "hybrid"}

    chunk_lookup = build_chunk_lookup(chunks)

    # --------------------------------------------------
    # 1. Start from retriever seed chunks
    # --------------------------------------------------
    candidate_seed_docs = list(seed_docs[: cfg.max_graph_seed_chunks])

    # --------------------------------------------------
    # 2. Add a few query-concept seeds as fallback/enrichment
    # --------------------------------------------------
    q_concepts = extract_domain_concepts(question)

    for qc in q_concepts:
        candidate_seed_docs.extend(concept_to_chunks.get(qc, [])[:2])

        neighbors = relation_graph.get(qc, Counter())
        top_neighbors = [c for c, _ in neighbors.most_common(3)]

        for nb in top_neighbors:
            candidate_seed_docs.extend(concept_to_chunks.get(nb, [])[:1])

    candidate_seed_docs = dedup_docs(candidate_seed_docs)
    candidate_seed_docs = candidate_seed_docs[: cfg.max_graph_seed_chunks]

    # --------------------------------------------------
    # 3. Convert seed docs to chunk ids
    # --------------------------------------------------
    seed_ids = []

    for d in candidate_seed_docs:
        key = doc_key(d)
        if key in chunk_lookup:
            seed_ids.append(chunk_lookup[key])

    seed_ids = list(dict.fromkeys(seed_ids))

    # --------------------------------------------------
    # 4. Traverse graph
    # --------------------------------------------------
    visited_ids = set(seed_ids)
    frontier = seed_ids[:]

    for _ in range(cfg.graph_hops):
        next_frontier = []

        for cid in frontier:
            neighbors = []

            if cfg.use_chunk_edges:
                neighbors.extend(chunk_edges.get(cid, []))

            if cfg.use_sequential_edges:
                neighbors.extend(sequential_graph.get(cid, []))

            if use_similarity_edges:
                neighbors.extend(similarity_edges.get(cid, []))

            # local dedup
            local_neighbors = []
            seen_local = set()

            for nb in neighbors:
                if nb not in seen_local:
                    seen_local.add(nb)
                    local_neighbors.append(nb)

            # prefer neighbors lexically closer to the question
            local_neighbors.sort(
                key=lambda nb: lexical_overlap_score(
                    question,
                    chunk_id_to_doc[nb].page_content,
                ),
                reverse=True,
            )

            local_neighbors = local_neighbors[: cfg.max_edge_neighbors_per_chunk]

            for nb in local_neighbors:
                if nb not in visited_ids:
                    visited_ids.add(nb)
                    next_frontier.append(nb)

        frontier = next_frontier

    graph_docs = [
        chunk_id_to_doc[cid]
        for cid in visited_ids
        if cid in chunk_id_to_doc
    ]

    return dedup_docs(graph_docs)


# ============================================================
# RERANK AND PAPER SELECTION
# ============================================================

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

_CROSS_ENCODER_CACHE = {}

def cross_encoder_rerank(
    question: str,
    docs: List[Document],
    cfg: ExperimentConfig,
) -> List[Document]:
    """
    True cross-encoder reranker.

    Scores (question, chunk) pairs directly.
    """

    if not docs:
        return docs

    model_name = cfg.cross_encoder_model

    if model_name not in _CROSS_ENCODER_CACHE:
        print("Loading cross-encoder reranker:", model_name)
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(
            model_name,
            device=cfg.device,
        )

    model = _CROSS_ENCODER_CACHE[model_name]

    docs_to_rerank = docs[: cfg.rerank_top_n]
    remaining_docs = docs[cfg.rerank_top_n :]

    pairs = [
        (question, d.page_content)
        for d in docs_to_rerank
    ]

    scores = model.predict(pairs)

    scored = list(zip(scores, docs_to_rerank))
    scored.sort(key=lambda x: float(x[0]), reverse=True)

    reranked = [d for _, d in scored]

    return reranked + remaining_docs

def apply_rerank_if_needed(
    question: str,
    docs: List[Document],
    cfg: ExperimentConfig,
) -> List[Document]:
    """
    Rerank dispatcher.

    Priority:
    1. cross-encoder rerank
    2. lexical overlap rerank
    3. no rerank
    """

    if cfg.use_cross_encoder_rerank:
        return cross_encoder_rerank(question, docs, cfg)

    if cfg.use_rerank:
        return simple_rerank(question, docs)

    return docs

def paper_level_rerank(
    question: str,
    docs: List[Document],
    top_papers: int,
    max_chunks_per_paper: int,
) -> List[Document]:
    papers = group_by_paper(docs)
    paper_scores = []

    for paper, p_docs in papers.items():
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

# ============================================================
# RETRIEVAL ORCHESTRATION
# ============================================================

def retrieve_documents(
    question: str,
    dataset: str,
    pipeline: dict,
) -> List[Document]:
    cfg = pipeline["cfg"]
    retriever = pipeline["retriever"]

    # --------------------------------------------------
    # 1. Initial retrieval: dense / BM25 / hybrid
    # --------------------------------------------------
    seed_docs = retriever.invoke(question)
    seed_docs = dedup_docs(seed_docs)

    docs = seed_docs

    # --------------------------------------------------
    # 2. Optional graph expansion
    # --------------------------------------------------
    if cfg.use_graph:
        graph_docs = expand_with_final_graph(
            question=question,
            seed_docs=seed_docs,
            concept_to_chunks=pipeline["concept_to_chunks"],
            relation_graph=pipeline["relation_graph"],
            chunk_edges=pipeline["chunk_edges"],
            sequential_graph=pipeline["sequential_graph"],
            similarity_edges=pipeline["similarity_edges"],
            chunk_id_to_doc=pipeline["chunk_id_to_doc"],
            chunks=pipeline["chunks"],
            cfg=cfg,
        )

        docs = dedup_docs(seed_docs + graph_docs)

    # --------------------------------------------------
    # 3. Dataset-specific paper-level rerank
    # --------------------------------------------------
    if dataset == "D1":
        docs = paper_level_rerank(
            question,
            docs,
            top_papers=cfg.top_papers_d1,
            max_chunks_per_paper=cfg.max_chunks_per_paper,
        )

    elif dataset == "D5":
        docs = paper_level_rerank(
            question,
            docs,
            top_papers=cfg.top_papers_d5,
            max_chunks_per_paper=cfg.max_chunks_per_paper,
        )

    # D4: no paper-level rerank by default

    # --------------------------------------------------
    # 4. Final rerank
    # --------------------------------------------------
    docs = apply_rerank_if_needed(question, docs, cfg)

    return docs[: cfg.top_k_final]

# ============================================================
# CONTEXT AND GENERATION
# ============================================================

def format_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        paper = d.metadata.get("paper", "unknown")
        page = d.metadata.get("page", "NA")
        parts.append(f"[Doc {i} | {paper} p.{page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def choose_prompt(dataset: str) -> str:
    if dataset == "D1":
        return PROMPT_D1
    if dataset == "D5":
        return PROMPT_D5
    return PROMPT_D4


# ============================================================
# PIPELINE BUILD
# ============================================================

def build_pipeline(cfg: ExperimentConfig):
    print("\n========================================")
    print("Building pipeline with config:")
    print(asdict(cfg))
    print("========================================\n")

    llm = build_llm(cfg)
    embeddings = build_embeddings(cfg)

    all_docs = load_pdf_documents(PDF_FOLDER)
    chunks = chunk_documents(all_docs, cfg)
    chunk_id_to_doc = {i: chunk for i, chunk in enumerate(chunks)}

    vectorstore = build_vectorstore(chunks, embeddings)

    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": cfg.top_k_retrieve}
    )

    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=tokenize,
    )
    bm25_retriever.k = cfg.bm25_k

    retriever = ConfigurableRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        cfg=cfg,
    )

    # Do not build graph for no-graph experiments
    if cfg.use_graph:
        (
            concept_to_chunks,
            concept_to_papers,
            relation_graph,
            chunk_edges,
            sequential_graph,
            similarity_edges,
        ) = build_graph(
            chunks=chunks,
            cfg=cfg,
            embeddings=embeddings,
        )
    else:
        concept_to_chunks = defaultdict(list)
        concept_to_papers = defaultdict(set)
        relation_graph = defaultdict(Counter)
        chunk_edges = defaultdict(list)
        sequential_graph = defaultdict(list)
        similarity_edges = defaultdict(list)

    return {
        "cfg": cfg,
        "llm": llm,
        "embeddings": embeddings,
        "all_docs": all_docs,
        "chunks": chunks,
        "chunk_id_to_doc": chunk_id_to_doc,
        "vectorstore": vectorstore,
        "dense_retriever": dense_retriever,
        "bm25_retriever": bm25_retriever,
        "retriever": retriever,
        "concept_to_chunks": concept_to_chunks,
        "concept_to_papers": concept_to_papers,
        "relation_graph": relation_graph,
        "chunk_edges": chunk_edges,
        "sequential_graph": sequential_graph,
        "similarity_edges": similarity_edges,
    }

# ============================================================
# QUERY API
# ============================================================

def graphrag_query(question: str, dataset: str, pipeline: dict):
    cfg = pipeline["cfg"]
    retriever = pipeline["retriever"]

    docs = retrieve_documents(
        question=question,
        dataset=dataset,
        pipeline=pipeline,
    )

    context = format_context(docs)

    prompt_tmpl = choose_prompt(dataset)
    prompt_text = prompt_tmpl.format(
        context=context,
        question=question,
    )

    resp = pipeline["llm"].invoke(prompt_text)

    return {
        "answer": resp.content,
        "documents": docs,
        "papers": [d.metadata.get("paper", "") for d in docs],
        "context": context,
    }

# ============================================================
# EXPORT
# ============================================================

def export_graph(
    chunks,
    chunk_edges,
    sequential_graph,
    similarity_edges,
    cfg: ExperimentConfig,
    output_prefix,
):
    G = nx.Graph()

    for i, chunk in enumerate(chunks):
        paper = chunk.metadata.get("paper", "unknown")
        page = chunk.metadata.get("page", "NA")

        G.add_node(
            str(i),
            chunk_id=i,
            paper=str(paper),
            page=str(page),
        )

    def add_typed_edge(src, dst, edge_type):
        src_s = str(src)
        dst_s = str(dst)

        if G.has_edge(src_s, dst_s):
            old_type = G[src_s][dst_s].get("edge_type", "")
            types = set(old_type.split("+")) if old_type else set()
            types.add(edge_type)
            G[src_s][dst_s]["edge_type"] = "+".join(sorted(types))
        else:
            G.add_edge(src_s, dst_s, edge_type=edge_type)

    if cfg.use_chunk_edges and cfg.graph_edge_mode in {"keyword", "hybrid"}:
        for src, neighbors in chunk_edges.items():
            for dst in neighbors:
                add_typed_edge(src, dst, "concept")

    if cfg.use_sequential_edges:
        for src, neighbors in sequential_graph.items():
            for dst in neighbors:
                add_typed_edge(src, dst, "sequential")

    if cfg.graph_edge_mode in {"similarity", "hybrid"}:
        for src, neighbors in similarity_edges.items():
            for dst in neighbors:
                add_typed_edge(src, dst, "similarity")

    for node in G.nodes():
        G.nodes[node]["degree"] = G.degree(node)

    Path("graphs").mkdir(exist_ok=True)

    graphml_path = f"graphs/{output_prefix}.graphml"
    nodes_path = f"graphs/{output_prefix}_nodes.csv"
    edges_path = f"graphs/{output_prefix}_edges.csv"

    nx.write_graphml(G, graphml_path)

    nodes_df = pd.DataFrame([
        {
            "node": n,
            "chunk_id": data.get("chunk_id"),
            "paper": data.get("paper"),
            "page": data.get("page"),
            "degree": G.degree(n),
            "text_preview": chunks[int(n)].page_content[:200].replace("\n", " "),
        }
        for n, data in G.nodes(data=True)
    ])

    edges_df = pd.DataFrame([
        {
            "source": u,
            "target": v,
            "edge_type": data.get("edge_type"),
        }
        for u, v, data in G.edges(data=True)
    ])

    nodes_df.to_csv(
        nodes_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
    )

    edges_df.to_csv(
        edges_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    cfg = ExperimentConfig(
        embed_model="BAAI/bge-small-en",
        chunk_size=800,
        chunk_overlap=150,
        chunk_strategy="fixed",
        top_k_retrieve=25,
        top_k_final=8,
        use_graph=True,
        graph_hops=1,
        edge_min_shared_keywords=1,
        use_rerank=True,
        device="cuda",
    )

    pipeline = build_pipeline(cfg)

    q1 = "Compare AxCaliber and ActiveAx"
    q2 = "Why does NODDI fail in grey matter?"
    q3 = "What imaging modality does Abdollahzadeh2019 use?"

    print("\n===== D1 =====")
    r1 = graphrag_query(q1, "D1", pipeline)
    print(r1["answer"])
    print(r1["papers"])

    print("\n===== D4 =====")
    r3 = graphrag_query(q3, "D4", pipeline)
    print(r3["answer"])
    print(r3["papers"])

    print("\n===== D5 =====")
    r2 = graphrag_query(q2, "D5", pipeline)
    print(r2["answer"])
    print(r2["papers"])