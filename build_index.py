"""
build_index.py  –– chunk pdfs ➜ embed ➜ FAISS ➜ save
--------------------------------------------------------
python build_index.py --docs "data/*.pdf" --index-dir index
"""
import argparse, glob, os, pickle, numpy as np, faiss
from typing import List
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import re

CHUNK, OVERLAP = 650, 60
EMBED_MODEL = "all-MiniLM-L6-v2"

def pdf_to_text(path: str) -> str:
    if path.lower().endswith('.pdf'):
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

def chunk(txt:str)->List[str]:
    step = CHUNK - OVERLAP
    return [txt[i:i+CHUNK].strip() for i in range(0,len(txt),step) if txt[i:i+CHUNK].strip()]

def main(docs:str, index_dir:str):
    paths   = glob.glob(docs)
    embedder= SentenceTransformer(EMBED_MODEL)
    chunks, meta = [], []
    for p in paths:
        txt = pdf_to_text(p)
        txt = re.sub(r'\n(?!\n)', ' ', txt)
        for c in chunk(txt):
            chunks.append(c)
            meta.append({"doc":os.path.basename(p),"text":c})
    embs = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    os.makedirs(index_dir, exist_ok=True)
    np.save(f"{index_dir}/embs.npy", embs)
    with open(f"{index_dir}/meta.pkl","wb") as f: pickle.dump(meta,f)
    idx = faiss.IndexFlatL2(embs.shape[1]); idx.add(embs)
    faiss.write_index(idx,f"{index_dir}/faiss.idx")
    print(f"✔ indexed {len(chunks)} passages from {len(paths)} pdf(s) → {index_dir}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--docs",required=True); ap.add_argument("--index-dir",default="index")
    main(**vars(ap.parse_args()))
