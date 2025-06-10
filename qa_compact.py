"""
qa_compact.py –– load FAISS + CompAct + local-Llama
---------------------------------------------------
python qa_compact.py \
    --question "What is ... ?" \
    --index-dir index \
    --model-path models/llama-3-8b-instruct.Q4_K_M.gguf \
    --compact-model models/compact7b/CompAct-7b.Q4_K_S.gguf
"""
import argparse, pickle, json, os, numpy as np, faiss, sys, time, re
from llama_cpp import Llama as _GGUF
from sentence_transformers import SentenceTransformer
sys.path.append(os.path.join(os.path.dirname(__file__), "CompAct"))

TOP_K = 8
LLAMA_CTX = 4096
EMBED_MODEL = "all-MiniLM-L6-v2"

def load_index(idx_dir):
    idx = faiss.read_index(f"{idx_dir}/faiss.idx")
    meta= pickle.load(open(f"{idx_dir}/meta.pkl","rb"))
    return idx, meta

def get_compactor(path_to_gguf, device):
    llm = _GGUF(
        model_path=path_to_gguf,
        n_ctx=2048,
        n_threads=os.cpu_count() or 4,
        n_gpu_layers=24,
        temperature=0.0,
    )

    def compress(question: str, passages: list[str]) -> str:
        ctx = "\n".join(passages)
        prompt = (
            "### Instruction:\n"
            "Summarise <documents> in under 160 words so that the summary "
            "contains **all** information required to answer the question. "
            "Do NOT answer the question itself.\n\n"
            f"<question>\n{question}\n</question>\n\n"
            f"<documents>\n{ctx}\n</documents>\n\n"
            "### Summary:"
        )
        res = llm(prompt, max_tokens=120, stop=["###","</s>"])
        return res["choices"][0]["text"].strip()

    return compress

def clean_answer(text: str) -> str:
    text = re.sub(r'\[[^\]]+\]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&lt;', '<').replace('&gt;', '>')
    text = re.sub(r'\s+', ' ', text)

    hedges = [
        "the answer is inferred based on the context",
        "note:",
        "in conclusion",
        "therefore,",
        "the passage provided does not explicitly state",
        "it can be inferred that",
        "based on the context",
        "the answer is",
        "conclusion:"
    ]
    for hedge in hedges:
        text = re.split(re.escape(hedge), text, flags=re.IGNORECASE)[0]

    # Remove duplicate sentences
    sentences = []
    seen = set()
    for s in re.split(r'(?<=[.!?]) +', text):
        s_clean = s.strip().lower()
        if s_clean and s_clean not in seen:
            sentences.append(s.strip())
            seen.add(s_clean)
    cleaned = ' '.join(sentences).strip()

    # Truncate to the last complete sentence
    match = re.match(r'^(.*[.!?])', cleaned)
    if match:
        return match.group(1).strip()
    else:
        return cleaned.strip()

def main(question, index_dir, model_path, compact_model=None):
    embedder = SentenceTransformer(EMBED_MODEL)
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    idx, meta = load_index(index_dir)
    _, I = idx.search(q_emb[None, :], TOP_K)
    ctx   = [meta[i]["text"] for i in I[0]]

    if compact_model:
        device = "cuda" if False else "cpu"
        compactor = get_compactor(compact_model, device)
        comp_start = time.time()
        context_block = compactor(question, ctx)
        comp_end = time.time()
    else:
        comp_start = comp_end = time.time()
        # Directly concatenate retrieved context
        context_block = "\n\n".join(ctx)
    print(context_block)

    # strict_prompt = (
    #     "### Instruction:\nAnswer the question **using only the information"
    #     " in <context>**. If the answer is not present, say \"I don't know.\"\n\n"
    #     f"<context>\n{context_block}\n</context>\n\n"
    #     f"### Question:\n{question}\n\n### Answer:\n"
    # )

    prompt = (
       "### Instruction:\nAnswer the question using the information in <context>. "
       "If you can't find an exact answer, do your best to infer or summarize based on the context.\n\n"
       f"<context>\n{context_block}\n</context>\n\n"
       f"### Question:\n{question}\n\n### Answer:"
    )

    llama = _GGUF(model_path=model_path, n_ctx=LLAMA_CTX, temperature=0.0)
    ans_start = time.time()
    out = llama(prompt, max_tokens=128)['choices'][0]['text']
    ans_end = time.time()
    cleaned = clean_answer(out)
    print("\nANSWER ↴\n", cleaned)
    if compact_model:
        print(f"Compaction time: {comp_end - comp_start:.2f} seconds")
    else:
        print("Compaction time: 0.00 seconds (skipped)")
    print(f"Answer time:    {ans_end - ans_start:.2f} seconds")
    print(f"Total QA time:   {ans_end - comp_start:.2f} seconds")
    return cleaned, round(comp_end-comp_start,2), round(ans_end-ans_start,2), round(ans_end-comp_start,2)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--compact-model", required=False, default=None, help="Path to compaction model (optional). If not provided, compaction is skipped.")
    args = vars(ap.parse_args())
    main(**args)