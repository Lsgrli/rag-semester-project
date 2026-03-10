import re
import random
import textwrap

# Add
import os
from pathlib import Path
from rag_test import app

from langchain_ollama import ChatOllama

# ── Stubs ─────────────────────────────────────────────────────────────────────

# Update
def graphrag_query(question):
    result = app.invoke({"question": question})
    return result 

def llm_judge(question, reference, prediction):
    judge_llm = ChatOllama(
    model="qwen2:7b",   # ou un modèle plus petit si tu veux
    temperature=0
)
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

def llm_judge_citation(statement, prediction):
    return random.randint(0, 2)

# Right paper ?
def paper_recall_score(item, retrieved_docs):
    target = item["paper"].lower().replace(".pdf", "")

    retrieved = [
        d.metadata.get("paper", "").lower().replace(".pdf", "")
        for d in retrieved_docs
    ]

    print("Target paper:", target)
    print("Retrieved papers:", list(retrieved))

    return int(target_paper in retrieved_papers)

# ── Unified entry constructor ──────────────────────────────────────────────────
# Every loader returns a list of dicts with exactly these four keys.

def entry(question, answer, paper="", tag=""):
    return {"question": question, "answer": answer, "paper": paper, "tag": tag}

# ── Dataset 1 loader ──────────────────────────────────────────────────────────
# Format: non-blank lines prefixed [Q#][TAG]; questions.txt[i] ↔ answers.txt[i].

_D1_TAGS = {"A": "synthesis", "B": "multi-hop", "C": "contradiction",
            "D": "comparison", "E": "factual"}

def load_dataset1(base="dataset1"):
    prefix = re.compile(r'^\[Q\d+\]\[([A-Z])\]\s+')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, base)

    def parse_lines(path):
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = prefix.match(line)
                tag  = _D1_TAGS.get(m.group(1), m.group(1)) if m else ""
                text = prefix.sub("", line) if m else line
                out.append((text, tag))
        return out
    
    # Update
    qs  = parse_lines(os.path.join(base_path, "questions.txt"))
    as_ = parse_lines(os.path.join(base_path, "answers.txt"))

    n = min(len(qs), len(as_))
    return [entry(q, a, tag=t) for (q, t), (a, _) in zip(qs[:n], as_[:n])]

# ── Dataset 2 & 3 loader ──────────────────────────────────────────────────────
# Statements: [LINE N] text [???]   →  question
# Answers:    [LINE N] CiteKey | passage  →  paper + answer
# Skip [MISSING_REFERENCE] lines.

def load_citation_dataset(base):
    script_dir = Path(__file__).resolve().parent
    base_path = script_dir / base

    def parse_file(path, skip_missing=False):
        rows = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("[LINE "):
                    continue
                if skip_missing and "[MISSING_REFERENCE]" in line:
                    continue
                m = re.match(r'\[LINE (\d+)\]\s+(.*)', line)
                if m:
                    rows[int(m.group(1))] = m.group(2)
        return rows

    stmts   = parse_file(base_path / "citation_statements.txt", skip_missing=True)
    raw_ans = parse_file(base_path / "citation_answers.txt",    skip_missing=True)

    result = []
    for n in sorted(stmts):
        if n not in raw_ans or " | " not in raw_ans[n]:
            continue
        cite_key, passage = raw_ans[n].split(" | ", 1)
        result.append(entry(stmts[n], passage.strip(),
                            paper=cite_key.strip(), tag="citation"))
    return result

# ── Dataset 4 loader ──────────────────────────────────────────────────────────
# Questions: Q###. [Tag] text  (anonymous, one per line)
# Answers:   ## PaperName / Q#. [Tag] question / A#. answer
# Tag and clean question text come from the answers index; match by position.

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
    if len(q_lines) != len(answer_index):
        print(f"  [D4 note] questions={len(q_lines)}, answers={len(answer_index)}; "
              f"using first {n} pairs")

    return answer_index[:n]

# ── Dataset 5 loader ──────────────────────────────────────────────────────────
# Questions: Q###. text  (globally numbered, no paper names)
# Answers:   ## PaperName / Q#. question / A#. [Tag] answer text
# 610 answer pairs vs 604 questions: 6 extras are intentional shared answers.

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
                entries.append(entry(re.sub(r'^Q\d+\.\s*', '', line),
                                     "", paper=current_paper))
            elif entries and re.match(r'^A\d+\.', line):
                raw = re.sub(r'^A\d+\.\s*', '', line)
                m = _tag_prefix.match(raw)
                entries[-1]["tag"]    = m.group(1) if m else ""
                entries[-1]["answer"] = _tag_prefix.sub("", raw) if m else raw
    return entries


def load_dataset5(base="dataset5"):
    script_dir = Path(__file__).resolve().parent
    base_path = script_dir / base

    refs = _parse_d5_answers(base_path / "answers.txt")

    with open(base_path / "questions.txt", encoding="utf-8") as f:
        questions = [re.sub(r'^Q\d+\.\s*', '', l.strip())
                     for l in f if re.match(r'^Q\d', l.strip())]

    n = min(len(questions), len(refs))
    if len(questions) != len(refs):
        print(f"  [D5 note] questions={len(questions)}, answers={len(refs)}; "
              f"using first {n} pairs")

    for q, ref in zip(questions[:n], refs[:n]):
        ref["question"] = q

    return refs[:n]

# ── Pretty-print ──────────────────────────────────────────────────────────────

def print_sample(label, item, width=100):
    q_str = textwrap.fill(item["question"], width=width - 5, subsequent_indent="     ")
    a_str = textwrap.fill(item["answer"][:400], width=width - 5, subsequent_indent="     ")
    paper = item["paper"] or "—"
    tag   = f"  Question tag: [{item['tag']}]" if item["tag"] else ""
    print(f"\n{'=' * width}")
    print(f"  [{label}]  {paper}{tag}")
    print(f"{'─' * width}")
    print(f"  Q: {q_str}")
    print(f"{'─' * width}")
    print(f"  A: {a_str}")
    print(f"{'=' * width}")

# ── Evaluation-function ──────────────────────────────────────────────────────────────
def evaluate_dataset(dataset, label, limit=10):

    scores = []
    recalls = []

    for item in dataset[:limit]:

        question = item["question"]
        reference = item["answer"]
        target_paper = item.get("paper", "")

        result = graphrag_query(question)

        prediction = result["answer"]
        retrieved_docs = result.get("documents", [])

        # ---- Paper Recall ----
        retrieved_papers = [
            d.metadata.get("paper", "")
            for d in retrieved_docs
        ]

        recall = int(target_paper in retrieved_papers)
        recalls.append(recall)

        # ---- LLM Judge ----
        score = llm_judge(question, reference, prediction)
        scores.append(score)

        print("\n========================")
        print("Dataset:", label)
        print("Question:", question)
        print("Target paper:", target_paper)
        print("Retrieved papers:", retrieved_papers)
        print("Prediction:", prediction)
        print("Reference:", reference)
        print("Paper Recall:", recall)
        print("LLM Score:", score)

    avg_score = sum(scores) / len(scores)
    avg_recall = sum(recalls) / len(recalls)

    print("\nAverage LLM score for", label, "=", round(avg_score, 3))
    print("Paper Recall for", label, "=", round(avg_recall, 3))

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading datasets...\n")
    '''
    d1 = load_dataset1()
    print(f"Dataset 1 : {len(d1)} Q&A pairs")
    for item in random.sample(d1, 2):
        print_sample("D1", item)

    d2 = load_citation_dataset("dataset2")
    print(f"\nDataset 2 : {len(d2)} citation pairs")
    for item in random.sample(d2, 2):
        print_sample("D2", item)

    d3 = load_citation_dataset("dataset3")
    print(f"\nDataset 3 : {len(d3)} citation pairs")
    for item in random.sample(d3, 2):
        print_sample("D3", item)
    '''
    d4 = load_dataset4()
    print(f"\nDataset 4 : {len(d4)} Q&A pairs")
    for item in random.sample(d4, 2):
        print_sample("D4", item)

    '''
    d5 = load_dataset5()
    print(f"\nDataset 5 : {len(d5)} Q&A pairs")
    for item in random.sample(d5, 2):
        print_sample("D5", item)

    print("\nDone. All datasets loaded successfully.")
    '''

    # Evaluation

    print("\nRunning RAG evaluation...\n")
    #evaluate_dataset(d1, "Dataset 1")
    #evaluate_dataset(d2, "Dataset 2")
    #evaluate_dataset(d3, "Dataset 3")
    evaluate_dataset(d4, "Dataset 4")
    #evaluate_dataset(d5, "Dataset 5")
