import re
import random
import textwrap
import os
import csv
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Import from your refactored RAG file
from rag_test import (
    ExperimentConfig,
    build_pipeline,
    graphrag_query,
    export_graph,
)

# ============================================================
# LLM JUDGE
# ============================================================

JUDGE_BACKEND = "ollama"   # or "vllm"
JUDGE_MODEL = "qwen2:7b"
JUDGE_BASE_URL = "http://localhost:8000/v1"
JUDGE_API_KEY = "token-abc123"

def build_judge_llm():
    if JUDGE_BACKEND == "ollama":
        return ChatOllama(
            model=JUDGE_MODEL,
            temperature=0
        )
    elif JUDGE_BACKEND == "vllm":
        return ChatOpenAI(
            model=JUDGE_MODEL,
            base_url=JUDGE_BASE_URL,
            api_key=JUDGE_API_KEY,
            temperature=0
        )
    else:
        raise ValueError(f"Unknown JUDGE_BACKEND: {JUDGE_BACKEND}")
    
judge_llm = build_judge_llm()

def llm_judge(question, reference, prediction):
    prompt = f"""
You are evaluating a scientific question answering system.

Question:
{question}

Reference Answer:
{reference}

System Prediction:
{prediction}

Score the prediction from 0 to 3:

0 = Incorrect or unrelated.
1 = Partially correct but missing key scientific elements.
2 = Mostly correct with minor omissions.
3 = Fully correct, specific, and scientifically accurate.

Respond with ONLY a single number (0, 1, 2, or 3).
"""
    response = judge_llm.invoke(prompt)

    try:
        return int(response.content.strip())
    except:
        return 0

# ============================================================
# METRICS
# ============================================================

def normalize_paper(name):
    return str(name).lower().replace(".pdf", "").strip()


def paper_recall_score(item, retrieved_docs):
    target = normalize_paper(item.get("paper", ""))

    retrieved = {
        normalize_paper(d.metadata.get("paper", ""))
        for d in retrieved_docs
    }

    return int(target in retrieved)


def chunk_precision(item, retrieved_docs):
    target = normalize_paper(item.get("paper", ""))
    
    retrieved = [
        normalize_paper(d.metadata.get("paper", ""))
        for d in retrieved_docs
    ]

    if not retrieved:
        return 0.0

    correct_chunks = sum(1 for p in retrieved if p == target)
    return correct_chunks / len(retrieved)


def top_chunk_score(item, retrieved_docs):
    if not retrieved_docs:
        return 0

    target = normalize_paper(item.get("paper", ""))
    top_paper = normalize_paper(retrieved_docs[0].metadata.get("paper", ""))

    return int(target == top_paper)


def inspect_chunks(retrieved_docs, k=3):
    for i, d in enumerate(retrieved_docs[:k]):
        paper = d.metadata.get("paper", "unknown")
        print("\n--- Chunk", i + 1, "---")
        print("Paper:", paper)
        print(d.page_content[:300])

# ============================================================
# DATASET HELPERS
# ============================================================

def entry(question, answer, paper="", tag=""):
    return {"question": question, "answer": answer, "paper": paper, "tag": tag}

_D1_TAGS = {
    "A": "synthesis",
    "B": "multi-hop",
    "C": "contradiction",
    "D": "comparison",
    "E": "factual"
}

def load_dataset1(base="dataset1"):
    prefix = re.compile(r'^\[Q\d+\]\[([A-Z])\]\s+')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, base)

    def parse_lines(path):
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = prefix.match(line)
                tag = _D1_TAGS.get(m.group(1), m.group(1)) if m else ""
                text = prefix.sub("", line) if m else line
                out.append((text, tag))
        return out

    qs = parse_lines(os.path.join(base_path, "questions.txt"))
    as_ = parse_lines(os.path.join(base_path, "answers.txt"))

    n = min(len(qs), len(as_))
    return [entry(q, a, tag=t) for (q, t), (a, _) in zip(qs[:n], as_[:n])]

def _parse_d4_answers(path):
    entries = []
    current_paper = None
    current_q = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("## "):
                current_paper = line[3:].strip()
                current_q = None
            elif current_paper:
                m = re.match(r'^Q\d+\.\s+\[([^\]]+)\]\s+(.*)', line)
                if m:
                    current_q = entry("", "", paper=current_paper, tag=m.group(1))
                    current_q["question"] = m.group(2)
                    entries.append(current_q)
                elif current_q:
                    m2 = re.match(r'^A\d+\.\s+(.*)', line)
                    if m2:
                        current_q["answer"] = m2.group(1)
    return entries

def load_dataset4(base="dataset4"):
    script_dir = Path(__file__).resolve().parent
    base_path = script_dir / base

    answer_index = _parse_d4_answers(base_path / "per_paper_answers.txt")

    with open(base_path / "per_paper_questions.txt", encoding="utf-8") as f:
        q_lines = [l.strip() for l in f if re.match(r'^Q\d{3}\.', l.strip())]

    n = min(len(q_lines), len(answer_index))
    return answer_index[:n]

_tag_prefix = re.compile(r'^\[(\w+)\]\s+')

def _parse_d5_answers(path):
    entries = []
    current_paper = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("## "):
                current_paper = line[3:].strip()
            elif re.match(r'^Q\d+\.', line):
                entries.append(entry(re.sub(r'^Q\d+\.\s*', '', line), "", paper=current_paper))
            elif entries and re.match(r'^A\d+\.', line):
                raw = re.sub(r'^A\d+\.\s*', '', line)
                m = _tag_prefix.match(raw)
                entries[-1]["tag"] = m.group(1) if m else ""
                entries[-1]["answer"] = _tag_prefix.sub("", raw) if m else raw

    return entries

def load_dataset5(base="dataset5"):
    script_dir = Path(__file__).resolve().parent
    base_path = script_dir / base

    refs = _parse_d5_answers(base_path / "answers.txt")

    with open(base_path / "questions.txt", encoding="utf-8") as f:
        questions = [
            re.sub(r'^Q\d+\.\s*', '', l.strip())
            for l in f if re.match(r'^Q\d', l.strip())
        ]

    n = min(len(questions), len(refs))

    for q, ref in zip(questions[:n], refs[:n]):
        ref["question"] = q

    return refs[:n]

# ============================================================
# PRINT
# ============================================================

def print_sample(label, item, width=100):
    q_str = textwrap.fill(item["question"], width=width - 5, subsequent_indent="     ")
    a_str = textwrap.fill(item["answer"][:400], width=width - 5, subsequent_indent="     ")
    paper = item["paper"] or "—"
    tag = f"  Question tag: [{item['tag']}]" if item["tag"] else ""

    print(f"\n{'=' * width}")
    print(f"  [{label}]  {paper}{tag}")
    print(f"{'─' * width}")
    print(f"  Q: {q_str}")
    print(f"{'─' * width}")
    print(f"  A: {a_str}")
    print(f"{'=' * width}")

# ============================================================
# EVALUATION
# ============================================================

def evaluate_dataset(dataset, label, dataset_code, pipeline, limit=10):
    llm_scores = []
    recalls = []
    precisions = []
    top_scores = []

    for item in dataset[:limit]:
        question = item["question"]
        reference = item["answer"]
        target_paper = item.get("paper", "")

        result = graphrag_query(question, dataset_code, pipeline)

        prediction = result["answer"]
        retrieved_docs = result.get("documents", [])

        recall = paper_recall_score(item, retrieved_docs)
        precision = chunk_precision(item, retrieved_docs)
        top_score = top_chunk_score(item, retrieved_docs)

        recalls.append(recall)
        precisions.append(precision)
        top_scores.append(top_score)

        score = llm_judge(question, reference, prediction)
        llm_scores.append(score)

        retrieved_papers = [
            d.metadata.get("paper", "")
            for d in retrieved_docs
        ]

        print("\n========================")
        print("Dataset:", label)
        print("Question:", question)

        print("\nTarget paper:", target_paper)
        print("Retrieved papers:", retrieved_papers)

        print("\nPrediction:", prediction)
        print("Reference:", reference)

        print("\nPaper Recall:", recall)
        print("Chunk Precision:", round(precision, 3))
        print("Top Chunk Accuracy:", top_score)
        print("LLM Score:", score)

    avg_score = sum(llm_scores) / len(llm_scores)
    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    avg_top = sum(top_scores) / len(top_scores)

    print("\n========================")
    print("FINAL RESULTS FOR", label)
    print("========================")
    print("Average LLM score      =", round(avg_score, 3))
    print("Paper Recall           =", round(avg_recall, 3))
    print("Chunk Precision        =", round(avg_precision, 3))
    print("Top Chunk Accuracy     =", round(avg_top, 3))

def evaluate_dataset_csv(dataset, label, dataset_code, output_file, pipeline, limit=None):
    results = []

    if limit is not None:
        #dataset = dataset[:limit]
        dataset = random.sample(dataset, k=min(limit, len(dataset)))
        

    for i, item in enumerate(dataset):
        question = item["question"]
        reference = item["answer"]
        target_paper = item.get("paper", "")
        tag = item.get("tag", "")

        result = graphrag_query(question, dataset_code, pipeline)

        prediction = result["answer"]
        retrieved_docs = result.get("documents", [])
        retrieved_papers = [d.metadata.get("paper", "") for d in retrieved_docs]

        recall = paper_recall_score(item, retrieved_docs)
        precision = chunk_precision(item, retrieved_docs)
        top_score = top_chunk_score(item, retrieved_docs)
        nb_chunks = len(retrieved_docs)
        nb_unique_papers = len(set(retrieved_papers))
        score = llm_judge(question, reference, prediction)

        results.append({
            "dataset": label,
            "question_id": i,
            "tag": tag,
            "question": question,
            "target_paper": target_paper,
            "retrieved_papers": ";".join(retrieved_papers),
            "nb_chunks_retrieved": nb_chunks,
            "nb_unique_papers": nb_unique_papers,
            "paper_recall": recall,
            "chunk_precision": precision,
            "top_chunk_accuracy": top_score,
            "llm_score": score
        })

        print(f"Processed question {i+1}/{len(dataset)}")

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nCSV written to:", output_file)

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# ----------------------------
# BASELINE EXAMPLE
# ----------------------------
BASELINE = [
    ExperimentConfig(
        llm_backend="ollama",
        llm_model="qwen2:7b",
        embed_model="BAAI/bge-small-en",
        chunk_size=512,
        chunk_overlap=100,
        chunk_strategy="fixed",
        top_k_retrieve=25,
        top_k_final=5,
        use_graph=True,
        graph_hops=1,
        edge_min_shared_keywords=1,
        use_rerank=True,
        top_papers_d1=4,
        top_papers_d5=4,
        max_chunks_per_paper=2,
        device="cuda",
    ),
]

def make_cfg(
    embed_model="BAAI/bge-small-en",
    chunk_size=512,
    chunk_overlap=100,
    chunk_strategy="fixed",
    top_k_retrieve=25,
    top_k_final=5,
    retrieval_mode="dense",
    bm25_k=25,
    hybrid_mode="rrf",
    rrf_k=60,
    use_graph=False,
    graph_hops=0,
    edge_min_shared_keywords=999,
    use_rerank=False,
    use_chunk_edges=False,
    use_sequential_edges=False,
    max_graph_seed_chunks=0,
    max_edge_neighbors_per_chunk=0,
    top_papers_d1=4,
    top_papers_d5=4,
    max_chunks_per_paper=2,

    graph_edge_mode="keyword",
    edge_similarity_threshold=0.76,
    max_similarity_neighbors=5,
    max_similarity_chunks=5000,

    use_cross_encoder_rerank=False,
    cross_encoder_model="BAAI/bge-reranker-base",
    rerank_top_n=50,
):
    return ExperimentConfig(
        llm_backend="ollama",
        llm_model="qwen2:7b",

        embed_model=embed_model,

        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_strategy,

        top_k_retrieve=top_k_retrieve,
        top_k_final=top_k_final,

        retrieval_mode=retrieval_mode,
        bm25_k=bm25_k,
        hybrid_mode=hybrid_mode,
        rrf_k=rrf_k,

        use_graph=use_graph,
        graph_hops=graph_hops,
        edge_min_shared_keywords=edge_min_shared_keywords,
        use_chunk_edges=use_chunk_edges,
        use_sequential_edges=use_sequential_edges,
        max_graph_seed_chunks=max_graph_seed_chunks,
        max_edge_neighbors_per_chunk=max_edge_neighbors_per_chunk,

        use_rerank=use_rerank,

        top_papers_d1=top_papers_d1,
        top_papers_d5=top_papers_d5,
        max_chunks_per_paper=max_chunks_per_paper,

        graph_edge_mode=graph_edge_mode,
        edge_similarity_threshold=edge_similarity_threshold,
        max_similarity_neighbors=max_similarity_neighbors,
        max_similarity_chunks=max_similarity_chunks,

        # Reranker
        use_cross_encoder_rerank=use_cross_encoder_rerank,
        cross_encoder_model=cross_encoder_model,
        rerank_top_n=rerank_top_n,

        device="cuda",
    )

# ----------------------------
# 1) EMBEDDING MODEL
# Run on D4 first
# ----------------------------

BEST_EMBED = 'intfloat/e5-large-v2'
BEST_CHUNK_SIZE = 1024     
BEST_OVERLAP = 200         
BEST_STRATEGY = "fixed" 
BEST_TOP_K_RETRIEVE = 25
BEST_TOP_K_FINAL = 12
BEST_RETRIEVAL_MODE = "hybrid" 
BEST_BM25_K = 50
BEST_HYBRID_MODE = "rrf"
BEST_RRF_K = 60

RERANK_CONFIGS = [

    (
        "I1_best_graph_lexical_rerank",
        make_cfg(
            embed_model=BEST_EMBED,
            chunk_size=BEST_CHUNK_SIZE,
            chunk_overlap=BEST_OVERLAP,
            chunk_strategy=BEST_STRATEGY,

            top_k_retrieve=BEST_TOP_K_RETRIEVE,
            top_k_final=BEST_TOP_K_FINAL,

            retrieval_mode=BEST_RETRIEVAL_MODE,
            bm25_k=BEST_BM25_K,
            hybrid_mode=BEST_HYBRID_MODE,
            rrf_k=BEST_RRF_K,

            use_graph=True,
            graph_edge_mode="keyword",
            graph_hops=1,
            edge_min_shared_keywords=999,
            use_chunk_edges=False,
            use_sequential_edges=True,
            max_graph_seed_chunks=5,
            max_edge_neighbors_per_chunk=3,

            use_rerank=True,
            use_cross_encoder_rerank=False,
        )
    ),

    (
        "I2_best_graph_cross_encoder_rerank",
        make_cfg(
            embed_model=BEST_EMBED,
            chunk_size=BEST_CHUNK_SIZE,
            chunk_overlap=BEST_OVERLAP,
            chunk_strategy=BEST_STRATEGY,

            top_k_retrieve=BEST_TOP_K_RETRIEVE,
            top_k_final=BEST_TOP_K_FINAL,

            retrieval_mode=BEST_RETRIEVAL_MODE,
            bm25_k=BEST_BM25_K,
            hybrid_mode=BEST_HYBRID_MODE,
            rrf_k=BEST_RRF_K,

            use_graph=True,
            graph_edge_mode="keyword",
            graph_hops=1,
            edge_min_shared_keywords=999,
            use_chunk_edges=True,
            use_sequential_edges=False,
            max_graph_seed_chunks=5,
            max_edge_neighbors_per_chunk=3,

            use_rerank=False,
            use_cross_encoder_rerank=True,
            cross_encoder_model="BAAI/bge-reranker-base",
            rerank_top_n=50,
        )
    ),
]

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    random.seed(42)

    print("Loading datasets...\n")

    d4 = load_dataset4()
    #d5 = load_dataset5()

    # Choose one phase at a time
    #configs = GRAPH_CONFIGS
    configs = RERANK_CONFIGS
    dataset = d4
    dataset_code = "D4"
    dataset_label = "Dataset4"

    # For graph experiments:
    #configs = GRAPH_CONFIGS
    #configs = RERANK_CONFIGS
    #dataset = d5
    #dataset_code = "D5"
    #dataset_label = "Dataset5"

    for exp_name, cfg in configs:
        print(f"\n===== EXPERIMENT {exp_name} =====")
        pipeline = build_pipeline(cfg)

        if cfg.use_graph:
            export_graph(
                chunks=pipeline["chunks"],
                chunk_edges=pipeline["chunk_edges"],
                sequential_graph=pipeline["sequential_graph"],
                similarity_edges=pipeline["similarity_edges"],
                cfg=cfg,
                output_prefix=exp_name,
            )

        evaluate_dataset_csv(
            dataset,
            label=f"{dataset_label}_{exp_name}",
            dataset_code=dataset_code,
            output_file=f"{dataset_label}_{exp_name}.csv",
            pipeline=pipeline,
            limit = None,
        )

