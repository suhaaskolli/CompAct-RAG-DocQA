import os, tempfile, time, re, pickle, numpy as np, faiss, uuid, gc
from flask import Flask, request, render_template_string, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama as _GGUF
from pypdf import PdfReader

# --- Config ---
ALLOWED_EXTENSIONS = {'pdf'}
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CTX = 4096
TOP_K = 4
MODEL_PATH = "models/llama-3-8b-instruct.Q4_K_M.gguf"
COMPACT_MODEL_PATH = "models/compact7b/CompAct-7b.Q4_K_S.gguf"

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'

INDEX_STORE = {}

def get_llama_model():
    return _GGUF(model_path=MODEL_PATH, n_ctx=MAX_CTX, temperature=0.0)

def get_compactor_model():
    return _GGUF(model_path=COMPACT_MODEL_PATH, n_ctx=MAX_CTX, n_threads=os.cpu_count() or 4, n_gpu_layers=32, temperature=0.0)

def get_compactor():
    llm = get_compactor_model()
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

# --- HTML Template ---
HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Document QA</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-5">
  <h1 class="mb-4">Document QA</h1>
  <form method="post" enctype="multipart/form-data" class="card p-4 shadow-sm">
    <div class="mb-3">
      <label for="pdf" class="form-label">Upload PDF</label>
      <input class="form-control" type="file" id="pdf" name="pdf" accept="application/pdf">
      {% if pdf_name %}<div class="form-text">Current PDF: <b>{{ pdf_name }}</b></div>{% endif %}
    </div>
    <div class="mb-3 form-check">
      <input type="checkbox" class="form-check-input" id="compact" name="compact" {% if use_compact %}checked{% endif %}>
      <label class="form-check-label" for="compact">Use Compaction</label>
    </div>
    <div class="mb-3">
      <label for="question" class="form-label">Your Question</label>
      <input class="form-control" type="text" id="question" name="question" value="{{ question|default('') }}">
    </div>
    <button type="submit" class="btn btn-primary">Ask</button>
  </form>
  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <div class="alert alert-danger mt-4">{{ messages[0] }}</div>
    {% endif %}
  {% endwith %}
  {% if answer %}
    <div class="card mt-4 p-4">
      <h5>Answer</h5>
      <pre style="white-space: pre-wrap;">{{ answer }}</pre>
      <div class="text-muted small mt-2">Compaction time: {{ comp_time }}s | Answer time: {{ ans_time }}s | Total: {{ total_time }}s</div>
    </div>
  {% endif %}
</div>
</body>
</html>
'''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pdf_to_text(path:str)->str:
    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)

def chunk(txt:str):
    CHUNK, OVERLAP = 650, 60
    step = CHUNK - OVERLAP
    return [txt[i:i+CHUNK].strip() for i in range(0,len(txt),step) if txt[i:i+CHUNK].strip()]

def scrub(text: str) -> str:
    text = re.sub(r'^\s*#{3,}.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'<\|[^>]{0,40}\|>', '', text)
    return text.strip()

def build_index_from_pdf(pdf_path):
    txt = pdf_to_text(pdf_path)
    txt = re.sub(r'\n(?!\n)', ' ', txt)
    chunks = chunk(txt)
    meta = [{"doc":os.path.basename(pdf_path),"text":c} for c in chunks]
    embedder = SentenceTransformer(EMBED_MODEL)
    embs = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    idx = faiss.IndexFlatL2(embs.shape[1]); idx.add(embs)
    return idx, meta

def answer_question(idx, meta, question, use_compact):
    embedder = SentenceTransformer(EMBED_MODEL)
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    if idx.ntotal == 0:
        raise RuntimeError("Index empty – upload a valid PDF.")
    _, I = idx.search(q_emb[None, :], TOP_K)
    ctx = [scrub(meta[i]["text"]) for i in I[0]]
    if use_compact:
        compactor = get_compactor()
        comp_start = time.time()
        context_block = compactor(question, ctx)
        comp_end = time.time()
        del compactor  # Explicitly free compactor model
        gc.collect()
    else:
        comp_start = comp_end = time.time()
        context_block = "\n\n".join(ctx)
    prompt = (
        "### Instruction:\nAnswer the question **using only the information"
        " in <context>**. If the answer is not present, say \"I don't know.\"\n\n"
        f"<context>\n{context_block}\n</context>\n\n"
        f"### Question:\n{question}\n\n### Answer:"
    )
    llama = get_llama_model()
    ans_start = time.time()
    out = llama(prompt, max_tokens=128)['choices'][0]['text']
    ans_end = time.time()
    return out.strip(), round(comp_end-comp_start,2), round(ans_end-ans_start,2), round(ans_end-comp_start,2)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = comp_time = ans_time = total_time = None
    pdf_name = session.get('pdf_name')
    session_id = session.get('id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['id'] = session_id
    idx_meta = INDEX_STORE.get(session_id)
    if request.method == 'POST':
        file = request.files.get('pdf')
        question = request.form.get('question','').strip()
        use_compact = bool(request.form.get('compact'))
        # If a new PDF is uploaded, rebuild index
        if file and file.filename and allowed_file(file.filename):
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = os.path.join(tmpdir, secure_filename(file.filename))
                file.save(pdf_path)
                idx, meta = build_index_from_pdf(pdf_path)
                INDEX_STORE[session_id] = (idx, meta)
                session['pdf_name'] = file.filename
                pdf_name = file.filename
        elif not idx_meta:
            flash('Please upload a PDF first.')
            return render_template_string(HTML, answer=None, pdf_name=None, question=question, use_compact=use_compact)
        else:
            idx, meta = idx_meta
        if not question:
            flash('Please enter a question.')
            return render_template_string(HTML, answer=None, pdf_name=pdf_name, question=question, use_compact=use_compact)
        try:
            answer, comp_time, ans_time, total_time = answer_question(idx, meta, question, use_compact)
        except Exception as e:
            flash(f'Error: {e}')
            return render_template_string(HTML, answer=None, pdf_name=pdf_name, question=question, use_compact=use_compact)
    return render_template_string(HTML, answer=answer, pdf_name=pdf_name, question=None, use_compact=False, comp_time=comp_time, ans_time=ans_time, total_time=total_time)

if __name__ == '__main__':
    app.run(debug=False, port=5000, use_reloader=False) 